"""Section B of the homework: the exploration metric."""

from __future__ import annotations

import numpy as np


def coverage_entropy(
    visit_counts: "np.ndarray", reachable_mask: "np.ndarray"
) -> tuple[float, float]:
    """Return reachable-state coverage and visitation entropy in bits."""
    counts = np.asarray(visit_counts)
    mask = np.asarray(reachable_mask, dtype=bool)
    if counts.shape != mask.shape:
        raise ValueError("visit_counts and reachable_mask must have the same shape")
    if np.any(counts < 0):
        raise ValueError("visit_counts must be non-negative")

    reachable = counts[mask].astype(np.float64, copy=False)
    num_reachable = reachable.size
    if num_reachable == 0:
        return 0.0, 0.0

    coverage = float(np.count_nonzero(reachable) / num_reachable)
    total = float(reachable.sum())
    if total <= 0.0:
        return coverage, 0.0

    probs = reachable[reachable > 0.0] / total
    entropy_bits = float(-np.sum(probs * np.log2(probs)))
    return coverage, entropy_bits
