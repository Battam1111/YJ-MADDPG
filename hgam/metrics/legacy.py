"""Legacy (pre-revision) metric implementations preserved for
old-vs-new comparison.

These functions reproduce the exact behavior of the pre-revision code,
**including** the division-by-zero risks and the ``min([1]*N, list)``
semantic bug. They exist for one reason: producing the comparison data
that paper-agent requested for the Response Letter to R2#8, so the
divergence introduced by the bug fixes can be quantified honestly.

**Do not use these from any production training, evaluation or
checkpoint-loading path.** Production code uses
:mod:`hgam.metrics.fairness`.

A few of the legacy implementations will raise ``ZeroDivisionError``
or return ``inf``/``nan`` on edge inputs. That is intentional —
silencing those would obscure the bug being characterized.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def geographical_fairness_buggy(
    data_orig: Sequence[float],
    data_final: Sequence[float],
) -> float:
    """Legacy omega_T — no epsilon protection. Can produce nan/inf when
    any m_0^p is zero or when all diffs are zero.
    """
    a = np.asarray(data_orig, dtype=np.float64)
    b = np.asarray(data_final, dtype=np.float64)
    num_sp = a.size
    diff = a - b
    norm = diff / a  # divides by zero if any a[i] == 0
    return float(norm.sum() ** 2 / (num_sp * (norm ** 2).sum()))


def charging_fairness_buggy(accumulated_charge_energy: Sequence[float]) -> float:
    """Legacy F — no epsilon, no zero-total guard. Returns nan when no
    charging has happened.
    """
    arr = np.asarray(accumulated_charge_energy, dtype=np.float64)
    total = arr.sum()
    normalized = arr / total  # nan if total == 0
    n = arr.size
    return float(normalized.sum() ** 2 / (n * (normalized ** 2).sum()))


def urgency_factor_buggy(remain_energy_list: Sequence[float], n: int | None = None):
    """Legacy urgency factor — the broken ``min([1]*N, list)`` form.

    In Python, ``min(list_a, list_b)`` does lexicographic comparison
    and returns whichever list is "smaller", **not** an elementwise
    minimum. So when remain_energy_list has its first element below 1.0,
    Python returns the entire remain_energy_list unchanged; when the
    first element is >=1.0, Python returns [1]*N.

    This function reproduces the original behavior exactly. Used only
    for the old-vs-new comparison.
    """
    remain = list(remain_energy_list)
    if n is None:
        n = len(remain)
    ones = [1] * n
    # The original was: min([1]*N, remain_energy_list)
    # Python's min on two lists returns one of the two whole lists.
    clipped = min(ones, remain)
    # The original then squared and Jain-indexed:
    s = sum(clipped)
    sq = sum(c ** 2 for c in clipped)
    return float(s ** 2 / (n * sq))


__all__ = [
    "geographical_fairness_buggy",
    "charging_fairness_buggy",
    "urgency_factor_buggy",
]
