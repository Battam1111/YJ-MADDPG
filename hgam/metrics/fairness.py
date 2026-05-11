"""Canonical, epsilon-safe metric implementations for the revised paper.

All Jain-style fairness indices use an additive epsilon in the denominator
to handle the degenerate case where the sum of squares is zero (e.g. all
agents had zero charge accumulation or all PoIs were fully collected).
The default epsilon is 1e-8.

Notation map (paper → code):

* ``omega_T``           : :func:`geographical_fairness`
* ``F`` (charging)      : :func:`charging_fairness`
* ``C`` (collection)    : :func:`data_collection_ratio`
* ``upsilon``           : :func:`energy_usage_efficiency`
* ``D``                 : :func:`charging_efficiency`

The ``urgency_factor`` helper is the corrected version of a reward-shaping
term used inside the CUAV reward (Phase B fix for the original
``min([1]*N, remain_energy_list)`` semantic bug — list comparison
returns the lexicographically smaller list, not an elementwise minimum.
The corrected form uses ``np.minimum``.)
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


_DEFAULT_EPS = 1e-8


def _jain_index(arr: np.ndarray, eps: float = _DEFAULT_EPS) -> float:
    """Jain's fairness index with additive epsilon guard.

    J(x) = (sum_i x_i)^2 / (N * sum_i x_i^2 + eps)

    When all x_i are zero, returns 1.0 (perfect equality of nothing).
    """
    arr = np.asarray(arr, dtype=np.float64)
    n = arr.size
    s = arr.sum()
    sq = (arr ** 2).sum()
    if s == 0.0 and sq == 0.0:
        return 1.0
    return float(s ** 2 / (n * sq + eps))


def geographical_fairness(
    data_orig: Sequence[float],
    data_final: Sequence[float],
    eps: float = _DEFAULT_EPS,
) -> float:
    """Geographical fairness ``omega_T`` over PoI data collection.

    omega_T = J( (m_0^p - m_T^p) / (m_0^p + eps) )

    where m_0^p is initial data at PoI p and m_T^p is final remaining
    data. The epsilon protects against PoIs that started with zero
    data, and Jain index protects against perfect collection (all zero
    diff).

    Boundary convention: if all PoIs started with zero data
    (degenerate scenario), returns 1.0.
    """
    a = np.asarray(data_orig, dtype=np.float64)
    b = np.asarray(data_final, dtype=np.float64)
    if a.size == 0:
        return 1.0
    diff = a - b
    norm = diff / (a + eps)
    return _jain_index(norm, eps=eps)


def charging_fairness(
    accumulated_charge_energy: Sequence[float],
    eps: float = _DEFAULT_EPS,
) -> float:
    """Charging fairness ``F`` across MUAVs.

    F = J( accumulated_charge_energy_i / sum_j accumulated_charge_energy_j )

    Normalizes accumulated charge per MUAV by the total before computing
    Jain index. When no charging has happened (sum is zero), returns 1.0
    (vacuous equality).
    """
    arr = np.asarray(accumulated_charge_energy, dtype=np.float64)
    total = arr.sum()
    if total == 0.0:
        return 1.0
    normalized = arr / total
    return _jain_index(normalized, eps=eps)


def data_collection_ratio(
    data_orig: Sequence[float],
    data_final: Sequence[float],
) -> float:
    """``C`` — fraction of initial PoI data that has been collected."""
    a = np.asarray(data_orig, dtype=np.float64)
    b = np.asarray(data_final, dtype=np.float64)
    total = a.sum()
    if total == 0.0:
        return 1.0
    collected = (a - b).sum()
    return float(collected / total)


def energy_usage_efficiency(
    sensing_energy_total: float,
    total_energy_total: float,
) -> float:
    """``upsilon`` — sensing-energy share of total energy consumption."""
    if total_energy_total == 0.0:
        return 0.0
    return float(sensing_energy_total / total_energy_total)


def charging_efficiency(
    charge_steps: int,
    episode_length: int,
) -> float:
    """``D`` — fraction of episode steps spent in active charging by CUAVs."""
    if episode_length == 0:
        return 0.0
    return float(charge_steps / episode_length)


def urgency_factor(
    remain_energy_list: Sequence[float],
    cap: float = 1.0,
    eps: float = _DEFAULT_EPS,
) -> float:
    """Corrected urgency factor used inside the CUAV reward shaping.

    The original implementation wrote ``min([1]*N, remain_energy_list)``
    which Python interprets as a lexicographic comparison between two
    lists, returning the lexicographically smaller list rather than the
    elementwise minimum. The corrected form below uses ``np.minimum``.

    Definition:
        clipped = np.minimum(cap, remain_energy_list)
        urgency = Jain(clipped)

    Lower remaining energy (lots of agents needing charge) drives the
    factor toward 0; uniform high energy drives it toward 1.
    """
    arr = np.asarray(remain_energy_list, dtype=np.float64)
    if arr.size == 0:
        return 1.0
    clipped = np.minimum(cap, arr)
    return _jain_index(clipped, eps=eps)
