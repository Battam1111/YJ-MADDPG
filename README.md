# HGAM — Heterogeneous Graph Attention MARL for UAV Coordination

Source code for the IEEE TVT submission *Breaking the Pre-Planning Barrier:
Adaptive Real-Time Coordination of Heterogeneous UAVs* (VT-2025-07369,
Major Revision in progress).

## Layout

```
.
├── hgam/                    main package
│   ├── env/                 PyBullet env (MUAV + CUAV)
│   ├── nn/                  GAT, encoders, action heads
│   ├── algorithms/          MARL algorithms (MADDPG today; MAAC/MAPPO/HATD3/HAPPO pending)
│   ├── replay/              replay buffer, PER, epsilon schedules
│   ├── training/            training and evaluation loops
│   ├── metrics/             fairness, efficiency metrics
│   ├── viz/                 trajectory and attention visualization
│   └── utils/               paths, config, checkpoints
├── configs/                 YAML configuration
│   ├── env/base.yaml        environment params
│   ├── method/hgam.yaml     network and training hyperparams
│   └── _legacy/             legacy single-dir layout for backward compatibility
├── scripts/train.py         CLI entry point
├── tests/test_smoke.py      sanity checks (imports + env reset)
├── data/                    runtime data, signal_points.npy (gitignored)
├── results/                 CSV experiment outputs (gitignored)
├── logs/                    TensorBoard event files (gitignored)
├── checkpoints/             trained models (gitignored)
└── _archive/                pre-refactor codebases, kept for traceability
```

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run smoke test
python tests/test_smoke.py

# 3. Train
python scripts/train.py --mode train --device cuda
```

## Refactor history

This repository was consolidated from 4 redundant codebases
(`hcanet-3.27_maddpg`, `hcanet-3.27_maddpg_old`,
`hcanet-3.27_maddpg-MAAC`, `HGAT-MADDPG_ver2`) totaling 2.8 GB into
a single canonical package. The 3 hcanet variants are frozen under
`_archive/` for traceability. A full pre-refactor snapshot is at
`/home/star/Yanjun/MADrones_backup_2026-05-11.tar.gz` (outside the
repository).

## Branches

- `master`: stable releases (currently equivalent to the pre-refactor backup)
- `backup-pre-refactor-2026-05-11`: frozen snapshot before any refactor work
- `refactor`: active refactor branch (this file lives here)
