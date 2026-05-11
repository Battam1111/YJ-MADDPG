#!/usr/bin/env python
"""E0 sweep launcher — runs (method, view, seed) cells in parallel subprocesses.

The whole job lifecycle (queue, dispatch, wait, aggregate progress) lives in
this single script so it stays auditable from a single file.

Examples
--------
Mini-sweep validation (2 views x 2 seeds = 4 runs, 4 concurrent):
    python scripts/run_e0_sweep.py --views local global --seeds 0 1 \\
                                   --episodes 80 --max-steps 100 --concurrency 4

E0 full HGAM (2 views x 5 seeds = 10 runs, 5 concurrent, paper-faithful):
    python scripts/run_e0_sweep.py --views local global --seeds 0 1 2 3 4 \\
                                   --episodes 5000 --max-steps 700 --concurrency 5

E0 with TRAIN_INTERVAL=2 for ~30% extra wall-clock saving:
    python scripts/run_e0_sweep.py --train-interval 2 ...

Output
------
Each run's stdout/stderr is captured to ``data/logs/<tag>.log``.
The launcher prints a one-line status as each run finishes, plus the
total wall-clock at the end. Exit code is non-zero if any run failed.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from itertools import product
from pathlib import Path


def _tag_for(method: str, view: str, seed: int, episodes: int) -> str:
    return f"{method}_{view}_seed{seed}_e{episodes}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--method",
        default="hgam",
        help=(
            "Method label used in the run tag. Currently only 'hgam' is "
            "implemented; future baselines (greedy, maddpg, maac, mappo, "
            "hatd3, happo) will share this launcher once they land."
        ),
    )
    parser.add_argument(
        "--views",
        nargs="+",
        default=["local", "global"],
        choices=["local", "global"],
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
    )
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument(
        "--train-interval",
        type=int,
        default=1,
        help="Forwarded to scripts/train.py as --train-interval.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help=(
            "Maximum simultaneous subprocesses. The 5-parallel calibration "
            "run on the project's RTX 4090 produced 2.77x throughput; "
            "scaling further was bottlenecked by GPU contention during "
            "update(). Setting this above 8 on a single 24GB GPU is not "
            "recommended."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to invoke scripts/train.py.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the runs that would be launched, then exit.",
    )
    args = parser.parse_args()

    runs = list(product([args.method], args.views, args.seeds))
    print(f"[sweep] {len(runs)} runs total; concurrency={args.concurrency}; "
          f"episodes={args.episodes}; max-steps={args.max_steps}; "
          f"train_interval={args.train_interval}")
    for method, view, seed in runs:
        tag = _tag_for(method, view, seed, args.episodes)
        print(f"  - {tag}")
    if args.dry_run:
        return 0

    # Sanity check: must be invoked from project root so ``scripts/train.py``
    # and relative config paths resolve correctly.
    repo_root = Path.cwd()
    if not (repo_root / "scripts" / "train.py").exists():
        print("[sweep] FATAL: expected to run from repo root with scripts/train.py present", file=sys.stderr)
        return 2
    log_root = repo_root / "data" / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    started = time.time()
    queue: list[tuple[str, str, int]] = list(runs)
    running: list[tuple[subprocess.Popen, str, object, float]] = []
    failures: list[tuple[str, int]] = []

    def _now_min() -> float:
        return (time.time() - started) / 60.0

    while queue or running:
        # Fill up to concurrency
        while len(running) < args.concurrency and queue:
            method, view, seed = queue.pop(0)
            tag = _tag_for(method, view, seed, args.episodes)
            log_path = log_root / f"{tag}.log"
            log_file = open(log_path, "w")
            cmd = [
                args.python, "scripts/train.py",
                "--device", args.device,
                "--episodes", str(args.episodes),
                "--max-steps", str(args.max_steps),
                "--seed", str(seed),
                "--view", view,
                "--tag", tag,
                "--train-interval", str(args.train_interval),
            ]
            print(f"[{_now_min():5.1f}m] LAUNCH {tag}")
            print(f"           cmd: {' '.join(shlex.quote(c) for c in cmd)}")
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
            running.append((proc, tag, log_file, time.time()))

        # Poll
        time.sleep(5.0)
        still_running = []
        for proc, tag, log_file, t0 in running:
            rc = proc.poll()
            if rc is None:
                still_running.append((proc, tag, log_file, t0))
                continue
            elapsed = (time.time() - t0) / 60.0
            log_file.close()
            if rc == 0:
                print(f"[{_now_min():5.1f}m] OK   {tag} ({elapsed:.1f}min)")
            else:
                print(f"[{_now_min():5.1f}m] FAIL {tag} exit={rc} ({elapsed:.1f}min); "
                      f"see data/logs/{tag}.log")
                failures.append((tag, rc))
        running = still_running

    total = _now_min()
    print(f"\n[sweep] {len(runs)} runs done in {total:.1f}min  "
          f"({len(failures)} failed)")
    if failures:
        print("[sweep] failed runs:")
        for tag, rc in failures:
            print(f"  - {tag} (exit {rc})")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
