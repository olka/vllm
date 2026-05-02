# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Otsu's method for automatic threshold selection on bimodal distributions.

Used by KVCacheEvictionManager to derive a per-cycle "dead block"
threshold from the histogram of EMA-smoothed importance scores. KV
cache attention is empirically bimodal (hot vs cold populations);
Otsu's method finds the threshold that best separates the two by
maximizing inter-class variance:

    σ²_b(t) = w₀(t) · w₁(t) · [μ₀(t) - μ₁(t)]²

where w₀, w₁ are class weights (probability mass below / above t) and
μ₀, μ₁ are the corresponding class means.

Reference: N. Otsu, "A Threshold Selection Method from Gray-Level
Histograms", IEEE Trans. on Systems, Man, and Cybernetics, 1979.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def otsu_threshold(
    values: Iterable[float],
    num_bins: int = 256,
) -> float:
    """Compute the Otsu threshold over an iterable of scalar values.

    Returns the value of the histogram bin edge that maximizes
    between-class variance. Falls back to ``min(values)`` for
    degenerate inputs (empty, single-valued, or all-equal).

    Args:
        values: Score samples. Order does not matter.
        num_bins: Histogram resolution. Higher values give a finer
            threshold at the cost of more compute. 256 is the standard
            image-processing default and is more than adequate for
            KV-cache score distributions where the bimodal split is
            many orders of magnitude wide.

    Returns:
        Threshold value. Items strictly below this are "background"
        (cold / dead); items at or above are "foreground" (hot / live).
    """
    arr = np.fromiter(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0

    lo = float(arr.min())
    hi = float(arr.max())
    if lo == hi:
        return lo

    hist, edges = np.histogram(arr, bins=num_bins, range=(lo, hi))
    total = hist.sum()
    if total == 0:
        return lo

    p = hist.astype(np.float64) / total
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    omega = np.cumsum(p)                # cumulative class-0 weight
    mu = np.cumsum(p * bin_centers)     # cumulative class-0 first moment
    mu_total = mu[-1]

    eps = 1e-12
    denom = np.clip(omega * (1.0 - omega), eps, None)
    inter_class_var = (mu_total * omega - mu) ** 2 / denom

    best_idx = int(np.argmax(inter_class_var))
    # The threshold sits at the boundary between bin best_idx and
    # bin best_idx + 1; use the upper edge so values strictly above
    # are foreground.
    return float(edges[best_idx + 1])
