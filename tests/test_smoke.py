"""Smoke test for the refactored hgam package.

Checks:
1. All major imports succeed.
2. Config loader returns a dict containing the expected env + method keys.
3. The PyBullet environment can be instantiated and reset (DIRECT mode,
   no GUI) on CPU. We do not assert specific shapes here because the obs
   dim depends on agent count and is computed at runtime; only that the
   reset produces a non-empty observation tensor.

Run with `python tests/test_smoke.py` from the project root.
"""
import sys
from pathlib import Path

# Allow running without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_imports_succeed() -> None:
    import hgam  # noqa: F401
    from hgam.algorithms.maddpg import MADDPGController  # noqa: F401
    from hgam.nn.gat import GATModule  # noqa: F401
    from hgam.nn.encoding import EncoderByType  # noqa: F401
    from hgam.nn.nets import MADDPGAgent, Actor_graph, Critic  # noqa: F401
    from hgam.replay.memory import ReplayMemory  # noqa: F401
    from hgam.env.robot import Drone, ChargeUAV  # noqa: F401
    from hgam.env.sensing_env import SensingEnv  # noqa: F401
    from hgam.utils import load_default_config  # noqa: F401


def test_config_loads() -> None:
    from hgam.utils import load_default_config
    cfg = load_default_config()
    assert "NUM_DRONE" in cfg, (
        f"expected NUM_DRONE in merged config, got keys={list(cfg.keys())[:10]}"
    )
    assert "BATCH_SIZE" in cfg, (
        f"expected BATCH_SIZE in merged config, got keys={list(cfg.keys())[:10]}"
    )
    assert "RANDOM_SEED" in cfg


def test_env_reset() -> None:
    import torch
    from hgam.env.sensing_env import SensingEnv
    from hgam.utils import ensure_runtime_dirs
    ensure_runtime_dirs()
    device = torch.device("cpu")
    env = SensingEnv(device, render=False)
    obs, adj = env.reset()
    assert obs.shape[0] > 0, f"expected non-empty obs, got shape={obs.shape}"
    assert adj.shape[0] == obs.shape[0], (
        f"obs rows {obs.shape[0]} != adj rows {adj.shape[0]}"
    )


if __name__ == "__main__":
    test_imports_succeed()
    print("[1/3] imports OK")
    test_config_loads()
    print("[2/3] config loads OK")
    test_env_reset()
    print("[3/3] env reset OK")
    print("\nAll smoke tests pass.")
