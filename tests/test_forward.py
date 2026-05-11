"""Forward-pass smoke test for the Phase C architecture.

Builds an MADDPGController for each of the four E-abl variants and
runs a single ``controller.act(...)`` call. Verifies that the new
type-aware GAT + type-specific encoder paths run end-to-end on CPU
without exceptions, and that the four variants are actually different
objects (different parameter counts).

Run with:
    python tests/test_forward.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from hgam.algorithms.maddpg import MADDPGController


# Variant -> (use_type_aware_bias, share_encoding)
VARIANTS = {
    "hgam-full":       (True,  False),
    "hgam-no-bias":    (False, False),
    "hgam-shared-enc": (True,  True),
    "hgam-vanilla":    (False, True),
}


def _build_controller(use_type_aware_bias: bool, share_encoding: bool):
    device = torch.device("cpu")
    num_uav = 3
    num_charger = 2
    node_types = [0] * num_uav + [1] * num_charger
    dim_obs_list = [180, 40]
    dim_act_list = [2, 2]
    return MADDPGController(
        checkpoint_file=str(Path("checkpoints/_forward_test")),
        checkpoint_dir=str(Path("checkpoints/_forward_test")),
        optimizer="adam",
        critic_lr=1e-3,
        actor_lr=1e-4,
        weight_decay=3e-3,
        rmsprop_alpha=1.0,
        rmsprop_eps=1.0,
        num_UAVAgents=num_uav,
        num_chargerAgents=num_charger,
        node_types=node_types,
        dim_obs_list=dim_obs_list,
        dim_act_list=dim_act_list,
        encoding_output_size=[128, 64],
        graph_hidden_size=[[128], [64]],
        action_hidden_size=[128, 64],
        share_encoding=share_encoding,
        act_encoding="leakyrelu",
        act_comms="leakyrelu",
        act_action="leakyrelu",
        gamma=0.98,
        tau=0.01,
        device=device,
        resume_run=False,
        memory_size=1000,
        full_receptive_field=False,
        gat_n_heads=1,
        gat_average_last=True,
        dropout=0.0,
        add_self_loops=False,
        use_type_aware_bias=use_type_aware_bias,
        num_node_types=2,
    )


def _fake_state_and_adj(num_total_agents: int = 5, obs_dim: int = 180):
    """Build a synthetic state and adjacency tensor that matches the
    shape that SensingEnv.reset() returns.

    state shape: (n_agents, obs_dim)
    adj shape:   (n_agents, 2)  each row holds [nearest_UAV_id, nearest_charger_id]
    """
    state = torch.randn(num_total_agents, obs_dim)
    # nearest_UAV for each agent: pick a different MUAV index (loop)
    # nearest_charger: index 3 (i.e. NUM_DRONE), the first CUAV
    n_uav = 3
    adj = torch.tensor(
        [
            [1, 3],  # MUAV 0's neighbors
            [0, 3],  # MUAV 1
            [0, 3],  # MUAV 2
            [0, 4],  # CUAV 0 (index 3): nearest UAV=0, nearest other CUAV=4
            [0, 3],  # CUAV 1 (index 4): nearest UAV=0
        ],
        dtype=torch.int64,
    )
    return state, adj


def test_each_variant_forwards() -> None:
    num_results = {}
    param_counts = {}
    for name, (bias, share) in VARIANTS.items():
        controller = _build_controller(
            use_type_aware_bias=bias, share_encoding=share
        )
        state, adj = _fake_state_and_adj()

        # controller.act expects state shape (N, obs_dim); episode_num must be
        # >= episode_before_train to avoid the noise-decay branch's edge cases.
        actions = controller.act(
            state_batch=state,
            adj=adj,
            episode_num=10,
            episode_before_train=0,
            if_noise=False,
        )
        assert actions.shape == (5, 2), (
            f"[{name}] expected actions of shape (5, 2), got {actions.shape}"
        )
        assert torch.isfinite(actions).all(), f"[{name}] non-finite actions"

        num_results[name] = actions
        param_counts[name] = sum(
            p.numel()
            for p in controller.UAVAgent.parameters()
        ) + sum(p.numel() for p in controller.chargerAgent.parameters())
        print(f"  [{name:18s}] OK  params={param_counts[name]:,}  "
              f"actions[0]={actions[0].tolist()}")

    # NOTE on what parameters() counts here: MADDPGAgent stores actors as a
    # plain Python list (not nn.ModuleList), so their params are *not*
    # reported by parameters(). The counts we get here therefore reflect
    # only the critics (online + target) — which is fine for distinguishing
    # the ablation variants since each variant changes critic architecture
    # in the same way it changes actor architecture.
    full = param_counts["hgam-full"]
    no_bias = param_counts["hgam-no-bias"]
    shared_enc = param_counts["hgam-shared-enc"]
    vanilla = param_counts["hgam-vanilla"]

    # full has the type-aware bias matrix; no-bias uses the legacy GATModule
    # which has no bias parameter. So full > no-bias.
    assert full > no_bias, (
        f"full ({full}) should have MORE params than no-bias ({no_bias}) — "
        f"type_bias parameter is exclusive to the type-aware path."
    )
    assert shared_enc > vanilla, (
        f"shared-enc ({shared_enc}) should have more params than "
        f"vanilla ({vanilla})"
    )

    # share-encoder variants use a single encoder slot (input dim 180+2=182
    # for critic), while two-encoder variants use both slots (182 + 42).
    # The encoder diff is constant across the bias-on/off axis.
    enc_diff_one = full - shared_enc
    enc_diff_two = no_bias - vanilla
    assert enc_diff_one == enc_diff_two, (
        f"Encoder-driven param diff should be invariant to the bias flag, "
        f"got {enc_diff_one} vs {enc_diff_two}"
    )

    # The bias-on/off param diff should also be invariant to the encoder flag.
    bias_diff_two_enc = full - no_bias
    bias_diff_one_enc = shared_enc - vanilla
    assert bias_diff_two_enc == bias_diff_one_enc, (
        f"Bias-driven param diff should be invariant to encoder flag, "
        f"got {bias_diff_two_enc} vs {bias_diff_one_enc}"
    )
    print(f"\n  parameter-count invariants OK")
    print(f"    encoder diff (per variant family): {enc_diff_one}")
    print(f"    bias    diff (per variant family): {bias_diff_two_enc}")


if __name__ == "__main__":
    print("Phase C forward-pass test — 4 ablation variants on CPU\n")
    try:
        test_each_variant_forwards()
        print("\nAll forward-pass variants passed.")
    except Exception as e:
        print(f"\nFAIL: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
