"""End-to-end pipeline smoke test (Phase D).

Validates that the full training loop runs without exceptions for a few
steps, with the Phase C type-aware GAT defaults active:

* env.reset() produces a usable state
* controller.act(state, adj) returns finite actions
* env.step(action) returns next state + reward + dones
* memory.push(...) stores transitions correctly
* controller.update(...) runs the critic+actor optimization step

A 5-step micro-loop is sufficient: we just need the code paths to fire
without crashing. The actual learning behavior is validated by E0
(70 runs in W1).

Run with:
    python tests/test_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pybullet as p
import torch

from hgam.algorithms.maddpg import MADDPGController
from hgam.env.sensing_env import SensingEnv
from hgam.utils import load_default_config, ensure_runtime_dirs
from hgam.utils.paths import PROJECT_ROOT


def test_pipeline_micro_loop() -> None:
    """Run 5 env steps + 1 update step end-to-end."""
    ensure_runtime_dirs()
    cfg = load_default_config()

    device = torch.device("cpu")  # CPU keeps the test fast and dependency-light
    np.random.seed(cfg["RANDOM_SEED"])
    torch.manual_seed(cfg["RANDOM_SEED"])

    env = SensingEnv(device, render=False)

    n_muav = int(cfg["NUM_DRONE"])
    n_cuav = int(cfg["NUM_CHARGER"])
    node_types = [0] * n_muav + [1] * n_cuav

    ckpt_dir = PROJECT_ROOT / "checkpoints" / "_pipeline_test"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    controller = MADDPGController(
        checkpoint_file=str(ckpt_dir),
        checkpoint_dir=str(ckpt_dir),
        optimizer=cfg["OPTIMIZER"],
        critic_lr=cfg["CRITIC_LR"],
        actor_lr=cfg["ACTOR_LR"],
        weight_decay=cfg["WEIGHT_DECAY"],
        rmsprop_alpha=cfg["RMSPROP_ALPHA"],
        rmsprop_eps=cfg["RMSPROP_EPS"],
        num_UAVAgents=n_muav,
        num_chargerAgents=n_cuav,
        node_types=node_types,
        dim_obs_list=cfg["DIMENSION_OBS"],
        dim_act_list=cfg["DIMENSION_ACTION"],
        encoding_output_size=cfg["encoding_output_size"],
        graph_hidden_size=cfg["graph_module_sizes"],
        action_hidden_size=cfg["action_hidden_size"],
        share_encoding=cfg["SHARE_ENCODING"],
        act_encoding=cfg["ACR_ENCODEING"],
        act_comms=cfg["ACT_COMMS"],
        act_action=cfg["ACT_ACTION"],
        gamma=cfg["GAMMA"],
        tau=cfg["TAU"],
        device=device,
        resume_run=False,
        memory_size=cfg["MEMORY_SIZE"],
        full_receptive_field=cfg["full_receptive_field"],
        gat_n_heads=cfg["gat_n_heads"],
        gat_average_last=cfg["gat_average_last"],
        dropout=cfg["dropout"],
        add_self_loops=cfg["add_loops"],
        use_type_aware_bias=cfg.get("USE_TYPE_AWARE_BIAS", False),
        num_node_types=cfg.get("NUM_NODE_TYPES", 2),
    )
    print(f"  controller built — phase-C flags: "
          f"USE_TYPE_AWARE_BIAS={cfg.get('USE_TYPE_AWARE_BIAS')} "
          f"SHARE_ENCODING={cfg['SHARE_ENCODING']}")

    state, adj = env.reset()
    print(f"  env reset OK — state.shape={tuple(state.shape)}")

    # Initial trajectory buffer mirrors what runner.PybulletRunner.run() does.
    start_pos = [list(p.getBasePositionAndOrientation(r.robot)[0]) for r in env.robot] + \
                [list(p.getBasePositionAndOrientation(c.robot)[0]) for c in env.charger]
    trajectory = [start_pos]

    for i_step in range(5):
        actions = controller.act(
            state_batch=state,
            adj=adj,
            episode_num=100,        # >= EPISODES_BEFORE_TRAIN to avoid noise-only branch
            episode_before_train=50,
            if_noise=True,
        )
        assert actions.shape[0] == n_muav + n_cuav, (
            f"action count mismatch: got {actions.shape}"
        )
        assert torch.isfinite(actions).all(), "non-finite actions"

        for robot in env.robot:
            p.resetBaseVelocity(robot.robot, linearVelocity=[0.0, 0.0, 0.0])
        for charger in env.charger:
            p.resetBaseVelocity(charger.robot, linearVelocity=[0.0, 0.0, 0.0])

        next_state, next_adj, reward, dones, _ = env.step(
            actions, i_step, np.array(trajectory)
        )
        assert len(reward) == n_muav + n_cuav, (
            f"reward shape: {len(reward)} vs expected {n_muav + n_cuav}"
        )

        controller.memory.push(
            state, adj, actions.cpu(), next_state, next_adj, reward, dones, 100
        )
        print(f"  step {i_step}: action_norm={actions.norm().item():.3f} "
              f"reward_sum={sum(reward):.3f} dones={sum(dones)}")

        state, adj = next_state, next_adj

        current_pos = [list(p.getBasePositionAndOrientation(r.robot)[0]) for r in env.robot] + \
                      [list(p.getBasePositionAndOrientation(c.robot)[0]) for c in env.charger]
        trajectory.append(current_pos)

        if sum(dones) > 0:
            print(f"  early termination at step {i_step}")
            break

    # Step 2: trigger one optimization step. We need at least BATCH_SIZE samples
    # in the buffer, so push a few duplicates to satisfy the sampler. This is
    # a smoke test for the update path — we are not training to convergence.
    for _ in range(max(0, int(cfg["BATCH_SIZE"]) - 5)):
        controller.memory.push(
            state, adj, actions.cpu(), state, adj, reward, dones, 100
        )
    print(f"  memory primed with {cfg['BATCH_SIZE']} samples; running update()")

    critic_loss, policy_loss = controller.update(i_step=0, param_dict=cfg)
    assert critic_loss is not None
    print(f"  update OK — critic_loss[0]={critic_loss[0].item():.4f}  "
          f"critic_loss[1]={critic_loss[1].item():.4f}")
    if policy_loss is not None:
        print(f"  policy_loss[0]={policy_loss[0].item():.4f}  "
              f"policy_loss[1]={policy_loss[1].item():.4f}")


if __name__ == "__main__":
    try:
        test_pipeline_micro_loop()
        print("\nPipeline smoke test PASSED.")
    except Exception as e:
        import traceback
        print(f"\nFAIL: {type(e).__name__}: {e}\n")
        traceback.print_exc()
        sys.exit(1)
