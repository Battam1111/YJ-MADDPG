"""HGAM evaluation metrics.

Two parallel surfaces are provided:

* :mod:`hgam.metrics.fairness` — the canonical, epsilon-safe and
  semantically-correct implementations used by the revised paper's
  E0+ runs.
* :mod:`hgam.metrics.legacy` — bit-for-bit reproductions of the
  pre-revision metric implementations (with division-by-zero and the
  ``min([1]*N, list)`` semantic bug intact). These exist solely so we
  can compute "old logic" alongside "new logic" on the same trained
  models for the Response Letter's R2#8 comparison table. Do not call
  legacy code from any production training path.
"""

from .fairness import (
    geographical_fairness,
    charging_fairness,
    data_collection_ratio,
    energy_usage_efficiency,
    charging_efficiency,
    urgency_factor,
)
from . import legacy

__all__ = [
    "geographical_fairness",
    "charging_fairness",
    "data_collection_ratio",
    "energy_usage_efficiency",
    "charging_efficiency",
    "urgency_factor",
    "legacy",
]
