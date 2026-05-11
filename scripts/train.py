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
    parser.add_argument(
        "--view",
        choices=["local", "global"],
        default=None,
        help=(
            "Observation view scope. 'local' sets LASER_LENGTH=4.0 (the "
            "paper's local-view baseline, default); 'global' sets "
            "LASER_LENGTH=16.0 (full arena, used for the local-vs-global "
            "paradox analysis in S6.3)."
        ),
    )
    parser.add_argument(
        "--train-interval",
        type=int,
        default=None,
        help=(
            "Override TRAIN_INTERVAL: how many env steps between update() "
            "calls. 1 = paper-faithful (update every step). Setting 2-4 "
            "trades some sample efficiency for proportional wall-clock "
            "speedup; HARL's MADDPG runner defaults to 50."
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
    if args.view is not None:
        # 'local' and 'global' both refer to the LASER_LENGTH knob that controls
        # how far each UAV can sense for neighbour / PoI features. The paper
        # baseline uses 4.0 for local and 16.0 (= full arena width) for global.
        overrides["LASER_LENGTH"] = 4.0 if args.view == "local" else 16.0
    if args.train_interval is not None:
        overrides["TRAIN_INTERVAL"] = args.train_interval

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
