"""
Channel 1 — SIFT keypoint matching with RANSAC, plus ORB fallback.

SIFT detects scale/rotation-invariant keypoints.  RANSAC filters
out spurious matches by requiring global geometric consistency.

When SIFT fails (common on degraded latent prints where the ratio
test kills everything because ridges are repetitive), we fall back
to ORB which handles binary ridge patterns more robustly.

IMPORTANT: Feed this channel **lightly enhanced** images (CLAHE
only, NO Gabor).  Gabor homogenises ridge texture and kills
keypoint discriminability — that was why SIFT returned 0.0 for
every comparison in v1.
"""

from __future__ import annotations

import numpy as np
import cv2
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)


class SIFTMatcher:
    """
    SIFT + RANSAC fingerprint matcher with ORB fallback.
    """

    def __init__(
        self,
        n_features: int = 2000,
        ratio_threshold: float = 0.82,
        ransac_reproj: float = 10.0,
        min_inliers: int = 4,
        orb_n_features: int = 3000,
        orb_ratio_threshold: float = 0.85,
    ):
        self.n_features = n_features
        self.ratio_threshold = ratio_threshold
        self.ransac_reproj = ransac_reproj
        self.min_inliers = min_inliers
        self.orb_n_features = orb_n_features
        self.orb_ratio_threshold = orb_ratio_threshold

        self._sift = cv2.SIFT_create(nfeatures=n_features)
        self._bf_l2 = cv2.BFMatcher(cv2.NORM_L2)

        self._orb = cv2.ORB_create(nfeatures=orb_n_features)
        self._bf_ham = cv2.BFMatcher(cv2.NORM_HAMMING)

    # ----- public --------------------------------------------------------

    def match(
        self,
        probe: np.ndarray,
        gallery: np.ndarray,
        probe_mask: Optional[np.ndarray] = None,
        gallery_mask: Optional[np.ndarray] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Match two fingerprint images (should be CLAHE-only enhanced).
        Tries SIFT first; falls back to ORB if SIFT fails.
        """
        score, details = self._match_sift(probe, gallery, probe_mask, gallery_mask)
        if score > 0:
            details["method"] = "sift"
            return score, details

        # SIFT failed — try ORB
        score, details = self._match_orb(probe, gallery, probe_mask, gallery_mask)
        details["method"] = "orb"
        return score, details

    # ----- SIFT -----------------------------------------------------------

    def _match_sift(
        self,
        probe: np.ndarray,
        gallery: np.ndarray,
        probe_mask: Optional[np.ndarray],
        gallery_mask: Optional[np.ndarray],
    ) -> Tuple[float, Dict[str, Any]]:
        kp1, des1 = self._sift.detectAndCompute(probe, probe_mask)
        kp2, des2 = self._sift.detectAndCompute(gallery, gallery_mask)

        n1, n2 = len(kp1), len(kp2)
        logger.debug(f"SIFT: probe={n1} kp, gallery={n2} kp")

        if n1 < self.min_inliers or n2 < self.min_inliers:
            return 0.0, {"reason": "too_few_keypoints", "n_kp1": n1, "n_kp2": n2}

        if des1 is None or des2 is None:
            return 0.0, {"reason": "no_descriptors"}

        raw_matches = self._bf_l2.knnMatch(des1, des2, k=2)

        # Lowe's ratio test
        good = []
        for pair in raw_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio_threshold * n.distance:
                good.append(m)

        logger.debug(f"SIFT: {len(raw_matches)} raw -> {len(good)} after ratio test")

        if len(good) < self.min_inliers:
            return 0.0, {"reason": "too_few_good_matches", "n_good": len(good)}

        return self._ransac_score(kp1, kp2, good, n1, n2)

    # ----- ORB fallback ---------------------------------------------------

    def _match_orb(
        self,
        probe: np.ndarray,
        gallery: np.ndarray,
        probe_mask: Optional[np.ndarray],
        gallery_mask: Optional[np.ndarray],
    ) -> Tuple[float, Dict[str, Any]]:
        kp1, des1 = self._orb.detectAndCompute(probe, probe_mask)
        kp2, des2 = self._orb.detectAndCompute(gallery, gallery_mask)

        n1, n2 = len(kp1), len(kp2)
        logger.debug(f"ORB: probe={n1} kp, gallery={n2} kp")

        if n1 < self.min_inliers or n2 < self.min_inliers:
            return 0.0, {"reason": "orb_too_few_keypoints", "n_kp1": n1, "n_kp2": n2}

        if des1 is None or des2 is None:
            return 0.0, {"reason": "orb_no_descriptors"}

        raw_matches = self._bf_ham.knnMatch(des1, des2, k=2)

        good = []
        for pair in raw_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if n.distance > 0 and m.distance < self.orb_ratio_threshold * n.distance:
                good.append(m)

        logger.debug(f"ORB: {len(raw_matches)} raw -> {len(good)} after ratio test")

        if len(good) < self.min_inliers:
            return 0.0, {"reason": "orb_too_few_good_matches", "n_good": len(good)}

        return self._ransac_score(kp1, kp2, good, n1, n2)

    # ----- shared RANSAC scoring -----------------------------------------

    def _ransac_score(
        self,
        kp1, kp2,
        good_matches,
        n1: int,
        n2: int,
    ) -> Tuple[float, Dict[str, Any]]:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.estimateAffinePartial2D(
            src_pts, dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_reproj,
        )

        if mask is None:
            return 0.0, {"reason": "ransac_failed"}

        inlier_mask = mask.ravel().astype(bool)
        n_inliers = int(inlier_mask.sum())

        logger.debug(f"RANSAC: {n_inliers}/{len(good_matches)} inliers")

        if n_inliers < self.min_inliers:
            return 0.0, {"reason": "too_few_inliers", "n_inliers": n_inliers}

        inlier_matches = [m for m, ok in zip(good_matches, inlier_mask) if ok]
        distances = np.array([m.distance for m in inlier_matches])

        # Component 1: inlier ratio
        min_kp = min(n1, n2)
        inlier_ratio = min(1.0, n_inliers / max(min_kp, 1))

        # Component 2: descriptor quality
        avg_dist = float(distances.mean())
        max_dist = 256.0 if distances.max() < 260 else 400.0
        desc_quality = max(0.0, 1.0 - avg_dist / max_dist)

        # Component 3: absolute inlier count (log-scaled)
        count_bonus = min(1.0, np.log1p(n_inliers) / np.log1p(40))

        score = (
            0.35 * inlier_ratio
            + 0.35 * desc_quality
            + 0.30 * count_bonus
        )
        score = float(np.clip(score, 0.0, 1.0))

        details = {
            "n_kp1": n1,
            "n_kp2": n2,
            "n_good": len(good_matches),
            "n_inliers": n_inliers,
            "inlier_ratio": round(inlier_ratio, 4),
            "avg_distance": round(avg_dist, 2),
            "desc_quality": round(desc_quality, 4),
            "affine_matrix": M.tolist() if M is not None else None,
        }

        logger.debug(
            f"Score={score:.4f}  inlier_ratio={inlier_ratio:.3f}  "
            f"desc_q={desc_quality:.3f}  count={count_bonus:.3f}"
        )
        return score, details

    def get_alignment_matrix(
        self,
        probe: np.ndarray,
        gallery: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Return the affine alignment matrix, or None."""
        _, details = self.match(probe, gallery)
        return (
            np.array(details["affine_matrix"])
            if details.get("affine_matrix") is not None
            else None
        )
