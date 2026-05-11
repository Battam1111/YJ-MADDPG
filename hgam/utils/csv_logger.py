"""Per-episode CSV logger for E0 sweeps.

Matches the schema paper-agent specified in
``docs/refactor_brief_for_paper_agent.md``:

    method, seed, config, view, episode, C, omega_new, omega_legacy,
    upsilon, D, F_new, F_legacy, episode_length, collision_count, total_reward

Two parallel fairness columns (``new`` and ``legacy``) are written so the
Response Letter for R2#8 can quantify the divergence introduced by the
Phase B bug fixes on the same trained models (paper-agent request from
``docs/project_paper_agent_decisions_2026-05-11.md`` constraint A).

Usage::

    logger = CSVEpisodeLogger(path="results/E0_hgam_local_seed0.csv",
                              method="hgam", seed=0, config="2m1c",
                              view="local")
    # At the end of every episode:
    logger.log_row(episode=42,
                   C=0.91, omega_new=0.88, omega_legacy=0.79, upsilon=0.61,
                   D=0.74, F_new=0.83, F_legacy=float("nan"),
                   episode_length=512, collision_count=0, total_reward=-12.3)
    logger.close()

The logger flushes after every row, so a crash mid-run still leaves a
useful CSV.  Float-NaN values are written as the literal string ``nan``
(Python default), which pandas reads as ``NaN``.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional


_HEADER = [
    "method",
    "seed",
    "config",
    "view",
    "episode",
    "C",
    "omega_new",
    "omega_legacy",
    "upsilon",
    "D",
    "F_new",
    "F_legacy",
    "episode_length",
    "collision_count",
    "total_reward",
]


class CSVEpisodeLogger:
    """Append-only CSV writer.  Writes a header row if the file did not exist."""

    def __init__(
        self,
        path: str | Path,
        *,
        method: str,
        seed: int,
        config: str,
        view: str,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._method = method
        self._seed = int(seed)
        self._config = str(config)
        self._view = str(view)

        need_header = not self.path.exists()
        self._fp = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fp)
        if need_header:
            self._writer.writerow(_HEADER)
            self._fp.flush()

    def log_row(
        self,
        *,
        episode: int,
        C: float,
        omega_new: float,
        omega_legacy: float,
        upsilon: float,
        D: float,
        F_new: float,
        F_legacy: float,
        episode_length: int,
        collision_count: int,
        total_reward: float,
    ) -> None:
        """Write one row.  All keyword-only to keep the call site self-documenting."""
        self._writer.writerow([
            self._method,
            self._seed,
            self._config,
            self._view,
            int(episode),
            float(C),
            float(omega_new),
            float(omega_legacy),
            float(upsilon),
            float(D),
            float(F_new),
            float(F_legacy),
            int(episode_length),
            int(collision_count),
            float(total_reward),
        ])
        self._fp.flush()

    def close(self) -> None:
        if self._fp and not self._fp.closed:
            self._fp.close()

    def __enter__(self) -> "CSVEpisodeLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
