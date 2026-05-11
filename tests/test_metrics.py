"""Unit tests for hgam.metrics.

Two responsibilities:

1. Confirm the canonical fairness / collection / urgency implementations
   behave correctly in the easy and the degenerate cases (empty input,
   all-zero, perfect equality).
2. Document the divergence between the legacy (buggy) implementations
   and the canonical ones — this is the comparison paper-agent will
   cite in the Response Letter for R2#8.

Run with:
    python tests/test_metrics.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from hgam.metrics import (
    geographical_fairness,
    charging_fairness,
    data_collection_ratio,
    urgency_factor,
)
from hgam.metrics import legacy


# ---------------------------------------------------------------------- #
# Canonical-implementation tests
# ---------------------------------------------------------------------- #

def test_geographical_fairness_perfect_collection() -> None:
    """All PoIs fully collected -> diff/orig = 1 for all -> Jain == 1."""
    orig = [10.0, 20.0, 30.0]
    final = [0.0, 0.0, 0.0]
    assert math.isclose(geographical_fairness(orig, final), 1.0, abs_tol=1e-6)


def test_geographical_fairness_zero_collection() -> None:
    """No PoIs touched -> diff/orig = 0 for all -> Jain returns 1.0 by convention."""
    orig = [10.0, 20.0]
    final = [10.0, 20.0]
    assert math.isclose(geographical_fairness(orig, final), 1.0, abs_tol=1e-6)


def test_geographical_fairness_handles_zero_orig() -> None:
    """A PoI with zero initial data must not crash the metric."""
    orig = [0.0, 10.0, 20.0]
    final = [0.0, 5.0, 10.0]
    val = geographical_fairness(orig, final)
    assert math.isfinite(val)


def test_charging_fairness_no_charging() -> None:
    arr = [0.0, 0.0, 0.0]
    assert math.isclose(charging_fairness(arr), 1.0, abs_tol=1e-6)


def test_charging_fairness_perfectly_uniform() -> None:
    arr = [5.0, 5.0, 5.0]
    assert math.isclose(charging_fairness(arr), 1.0, abs_tol=1e-6)


def test_charging_fairness_skewed() -> None:
    """One agent absorbing all charge -> Jain == 1/N."""
    arr = [10.0, 0.0, 0.0]
    expected = 1.0 / 3.0
    assert math.isclose(charging_fairness(arr), expected, abs_tol=1e-6)


def test_data_collection_ratio() -> None:
    orig = [10.0, 20.0]
    final = [2.0, 5.0]
    # collected = 8 + 15 = 23; total = 30
    assert math.isclose(data_collection_ratio(orig, final), 23.0 / 30.0, abs_tol=1e-9)


def test_urgency_factor_clip_below_one() -> None:
    """Values below 1 stay; clipping has no effect."""
    arr = [0.2, 0.5, 0.8]
    val = urgency_factor(arr)
    # Jain([0.2, 0.5, 0.8]) by hand: s=1.5, sq=0.93, N=3 -> 2.25/2.79 ~= 0.806
    assert 0.7 < val < 0.9


def test_urgency_factor_clip_above_one() -> None:
    """Values above 1 get clipped to 1 by np.minimum."""
    arr = [1.5, 2.0, 3.0]  # all clipped to 1 -> Jain == 1
    assert math.isclose(urgency_factor(arr), 1.0, abs_tol=1e-6)


# ---------------------------------------------------------------------- #
# Legacy vs new — divergence characterization for Response Letter R2#8
# ---------------------------------------------------------------------- #

def test_legacy_urgency_factor_lex_bug() -> None:
    """The legacy urgency_factor uses Python min() on two lists, which is
    lexicographic. Demonstrate the bug exists and the new version differs.
    """
    # Case where first element of remain < 1 -> Python returns remain whole
    remain_small_first = [0.2, 5.0, 5.0]
    legacy_val = legacy.urgency_factor_buggy(remain_small_first)
    new_val = urgency_factor(remain_small_first)

    # The two values must differ because of the bug
    assert not math.isclose(legacy_val, new_val, abs_tol=1e-3), (
        f"legacy={legacy_val} new={new_val} — expected divergence "
        f"due to min-list semantic bug"
    )

    # Specifically: legacy uses [0.2, 5.0, 5.0] verbatim (no clipping).
    # Jain([0.2, 5.0, 5.0]) = (10.2)^2 / (3 * (0.04 + 25 + 25)) = 104.04 / 150.12 ~= 0.693
    s = 0.2 + 5.0 + 5.0
    sq = 0.04 + 25 + 25
    expected_legacy = (s ** 2) / (3 * sq)
    assert math.isclose(legacy_val, expected_legacy, abs_tol=1e-6)

    # New version clips to 1 first: Jain([0.2, 1.0, 1.0]) = (2.2)^2 / (3 * (0.04+1+1)) = 4.84 / 6.12 ~= 0.791
    s2 = 0.2 + 1.0 + 1.0
    sq2 = 0.04 + 1 + 1
    expected_new = (s2 ** 2) / (3 * sq2)
    assert math.isclose(new_val, expected_new, abs_tol=1e-6)


def test_legacy_charging_fairness_divzero() -> None:
    """Legacy charging_fairness produces nan when sum is zero. Confirm and
    contrast with the new version which returns 1.0.
    """
    arr = [0.0, 0.0, 0.0]
    new_val = charging_fairness(arr)
    legacy_val = legacy.charging_fairness_buggy(arr)
    assert new_val == 1.0
    assert math.isnan(legacy_val), f"expected nan from legacy, got {legacy_val}"


def test_legacy_geographical_fairness_divzero_on_zero_orig() -> None:
    """Legacy geographical_fairness divides by data_orig elementwise; if any
    initial value is zero, the legacy version produces inf/nan, while the
    new version returns a finite value.
    """
    orig = [0.0, 10.0]
    final = [0.0, 5.0]
    new_val = geographical_fairness(orig, final)
    legacy_val = legacy.geographical_fairness_buggy(orig, final)
    assert math.isfinite(new_val), f"new={new_val} should be finite"
    assert not math.isfinite(legacy_val), (
        f"legacy should be inf/nan, got {legacy_val}"
    )


# ---------------------------------------------------------------------- #
# Runner
# ---------------------------------------------------------------------- #

if __name__ == "__main__":
    import inspect

    funcs = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failures = 0
    for name, fn in funcs:
        try:
            fn()
            print(f"  pass  {name}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL  {name}: {e}")
    print()
    print(f"{len(funcs) - failures}/{len(funcs)} metrics tests passed.")
    if failures:
        sys.exit(1)
