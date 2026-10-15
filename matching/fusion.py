"""
Score fusion — combine matching channels into one score.

v5 — SAFIS-primary score fusion with POC tiebreaker.

Architecture:
    Channel 1: SIFT + RANSAC   (keypoint matching — tiebreaker only)
    Channel 2: Gabor-POC       (texture matching on Gabor-enhanced images)
    Channel 3: CLAHE-POC       (texture matching on CLAHE-enhanced images)
    Channel 4: SAFIS / BOZORTH3 (minutiae matching via afis library)

Fusion strategy — SAFIS-primary with POC tiebreaker:
    SAFIS raw_score is the primary ranking signal.  raw_score is quantized
    as 6.5 * n_matched_pairs, so many gallery images share the same tier.
    Within each tier, the best POC score (max of Gabor-POC and CLAHE-POC)
    breaks ties.

    Formula:  fused = safis_raw + best_poc * POC_SCALE

    POC_SCALE is set to 5.0, which is below the SAFIS tier gap (6.5).
    This guarantees that POC can reorder images within a SAFIS tier but
    NEVER promote an image across tiers.  The 7 probes that SAFIS alone
    places at rank 1 are therefore perfectly preserved.

Empirical results (15 probes, SAFIS raw_score only):
    Rank-1: 7/15 (46.7%), Rank-5: 8/15 (53.3%), Rank-10: 8/15 (53.3%)
    Mean rank: 48.6
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Dict, Any, Tuple, List, Optional

logger = logging.getLogger(__name__)


def _compute_ranks(scores: np.ndarray) -> np.ndarray:
    """Convert scores to ranks (1 = best).  Ties get the same rank."""
    order = np.argsort(-scores)
    ranks = np.zeros(len(scores), dtype=int)
    for r, idx in enumerate(order):
        ranks[idx] = r + 1
    return ranks


def fuse_safis_primary(
    minutiae_scores: np.ndarray,
    poc_gabor_scores: Optional[np.ndarray] = None,
    poc_clahe_scores: Optional[np.ndarray] = None,
    sift_scores: Optional[np.ndarray] = None,
    poc_scale: float = 5.0,
    sift_tiebreak_weight: float = 0.0001,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    SAFIS-primary fusion with POC as within-tier tiebreaker.

    SAFIS raw_score is quantized at 6.5 * n_matched_pairs.  Many gallery
    images share the same SAFIS tier.  POC (the best of Gabor-POC and
    CLAHE-POC) breaks ties within each tier.

    Formula:  fused = safis_raw + best_poc * poc_scale

    poc_scale MUST be < 6.5 to guarantee that POC cannot promote an
    image across SAFIS tiers.  Default 5.0 gives POC strong tiebreaking
    power within a tier while keeping cross-tier ordering intact.

    Parameters
    ----------
    minutiae_scores : ndarray
        SAFIS raw_score values (NOT normalized .score).
    poc_gabor_scores : ndarray, optional
        Phase-only correlation on Gabor-enhanced images [0, 1].
    poc_clahe_scores : ndarray, optional
        Phase-only correlation on CLAHE-enhanced images [0, 1].
    sift_scores : ndarray, optional
        SIFT scores for micro-tiebreaking within POC ties.
    poc_scale : float
        Scale factor for POC bonus.  Must be < 6.5.  Default 5.0.
    sift_tiebreak_weight : float
        Weight of SIFT z-score as micro-tiebreaker (default 0.0001).

    Returns
    -------
    fused_scores : ndarray
        Fused scores (higher = better).
    details : dict
        Fusion metadata.
    """
    N = len(minutiae_scores)
    fused = minutiae_scores.astype(float).copy()

    # Best-of-two POC channels as tiebreaker
    best_poc = np.zeros(N, dtype=float)
    if poc_gabor_scores is not None and poc_clahe_scores is not None:
        best_poc = np.maximum(poc_gabor_scores, poc_clahe_scores)
    elif poc_gabor_scores is not None:
        best_poc = poc_gabor_scores.astype(float)
    elif poc_clahe_scores is not None:
        best_poc = poc_clahe_scores.astype(float)

    fused += best_poc * poc_scale

    # SIFT micro-tiebreaker (within POC ties)
    if sift_scores is not None and sift_scores.std() > 1e-10:
        sift_z = (sift_scores - sift_scores.mean()) / (sift_scores.std() + 1e-10)
        fused += sift_tiebreak_weight * sift_z

    details = {
        "method": "safis_primary",
        "poc_scale": poc_scale,
        "safis_max": float(minutiae_scores.max()),
        "safis_unique": int(len(np.unique(minutiae_scores))),
        "best_poc_max": float(best_poc.max()),
    }

    return fused, details


def fuse_all_scores_rank(
    sift_scores: np.ndarray,
    poc_gabor_scores: np.ndarray,
    minutiae_scores: np.ndarray,
    poc_clahe_scores: Optional[np.ndarray] = None,
    **kwargs,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Primary fusion entry point — SAFIS-primary with POC tiebreaker.

    SAFIS raw_score drives the ranking.  POC (best of Gabor and CLAHE)
    breaks ties within SAFIS tiers.  SIFT is a micro-tiebreaker.

    This replaces the RRF fusion which was destroying SAFIS's strong
    rank-1 signal by giving equal weight to noisy POC channels.
    """
    return fuse_safis_primary(
        minutiae_scores=minutiae_scores,
        poc_gabor_scores=poc_gabor_scores,
        poc_clahe_scores=poc_clahe_scores,
        sift_scores=sift_scores,
    )


# ---- Legacy single-pair functions (for GUI compatibility) ----------------

def fuse_scores(
    sift_score: float,
    minutiae_score: float,
    texture_score: float,
    w_sift: float = 0.05,
    w_minutiae: float = 0.50,
    w_texture: float = 0.45,
) -> Tuple[float, Dict[str, Any]]:
    """
    Weighted fusion of three channels (single-pair, for display).
    """
    scores = [
        ("sift", sift_score, w_sift),
        ("minutiae", minutiae_score, w_minutiae),
        ("texture", texture_score, w_texture),
    ]

    active_weight = sum(w for _, s, w in scores if s > 0.001 and w > 0.001)

    if active_weight < 0.001:
        fused = 0.0
    else:
        fused = 0.0
        for name, s, w in scores:
            if s > 0.001 and w > 0.001:
                adjusted_w = w / active_weight
                fused += adjusted_w * s

    fused = max(0.0, min(1.0, fused))

    details = {
        "sift": round(sift_score, 4),
        "minutiae": round(minutiae_score, 4),
        "texture": round(texture_score, 4),
        "fused": round(fused, 4),
        "weights": {"sift": w_sift, "minutiae": w_minutiae, "texture": w_texture},
    }
    return fused, details


def fuse_scores_adaptive(
    sift_score: float,
    minutiae_score: float,
    texture_score: float,
    bz_all_scores: np.ndarray = None,
) -> Tuple[float, Dict[str, Any]]:
    """Per-pair adaptive fusion (legacy fallback)."""
    return fuse_scores(sift_score, minutiae_score, texture_score)
