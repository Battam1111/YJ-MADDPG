#!/usr/bin/env python
"""CLI entry point for HGAM training and evaluation.

Examples
--------
    python scripts/train.py --mode train
    python scripts/train.py --mode train --device cuda
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
    args = parser.parse_args()

    ensure_runtime_dirs()

    device = torch.device(args.device)

    # Imported here so --help stays fast even before heavy deps initialize
    from hgam.training.runner import PybulletRunner

    runner = PybulletRunner(
        resume_run=args.resume,
        if_render=args.render,
        device=device,
    )

    if args.mode == "train":
        runner.run()
    else:
        runner.evaluate()


if __name__ == "__main__":
    main()
