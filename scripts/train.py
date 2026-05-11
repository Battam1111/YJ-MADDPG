#!/usr/bin/env python
"""CLI entry point for HGAM training and evaluation.

Examples
--------
    python scripts/train.py --mode train
    python scripts/train.py --mode train --device cuda --seed 0
    python scripts/train.py --mode train --episodes 60 --max-steps 100 --tag dry_run
    python scripts/train.py --mode test --resume
"""
import argparse
import sys
from pathlib import Path

# Ensure we can import the hgam package when running without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from hgam.utils import ensure_runtime_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or evaluate HGAM")
    parser.add_argument(
        "--mode",
        choices=["train", "test"],
        default="train",
        help="train: full training loop. test: evaluation only (requires --resume).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="torch device string (e.g. cuda, cuda:0, cpu)",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable PyBullet GUI rendering (slow; for local debugging only)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint under checkpoints/",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override N_EPISODES from config. Used for dry runs and E0 launchers.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override MAX_STEPS per episode from config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override RANDOM_SEED from config.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help=(
            "Run identifier; used as the checkpoint subdirectory name. "
            "E0 sweeps will pass tags like 'hgam_local_seed0' etc."
        ),
    )
    args = parser.parse_args()

    ensure_runtime_dirs()

    device = torch.device(args.device)

    # Imported here so --help stays fast even before heavy deps initialize
    from hgam.training.runner import PybulletRunner

    # Build the overrides dict from CLI args. None values are skipped so the
    # YAML defaults still apply when a flag isn't given.
    overrides = {}
    if args.episodes is not None:
        overrides["N_EPISODES"] = args.episodes
    if args.max_steps is not None:
        overrides["MAX_STEPS"] = args.max_steps
    if args.seed is not None:
        overrides["RANDOM_SEED"] = args.seed

    runner = PybulletRunner(
        resume_run=args.resume,
        if_render=args.render,
        device=device,
        overrides=overrides or None,
        run_tag=args.tag,
    )

    if args.mode == "train":
        runner.run()
    else:
        runner.evaluate()


if __name__ == "__main__":
    main()
