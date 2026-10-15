"""
Channel 2 — Minutiae extraction + alignment-free matching.

Extracts ridge endings and bifurcations via the crossing-number method
on a skeletonised image, then matches using TWO alignment-free strategies:

1. **Hough voting (Generalized Hough Transform):**
   For every pair (probe_minutia, gallery_minutia), compute the rigid
   transform (dx, dy, dtheta) that would align them.  Accumulate votes
   in a discretised transform space.  The bin with the most votes
   corresponds to the best alignment, and the vote count tells us how
   many minutiae are consistent with it.  True matches create a sharp
   peak; random pairs scatter across many bins.

2. **Local neighbourhood descriptors:**
   For each minutia, encode the relative positions and angles of its
   K nearest neighbours.  This descriptor is invariant to global
   translation and rotation.  Match descriptors between probe and
   gallery to count structurally-similar minutiae.

The best score from both strategies is returned.  If a SIFT affine
alignment is available, aligned Hungarian matching is also tried as
a third strategy.
"""

from __future__ import annotations

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import logging
from typing import List, Tuple, Dict, Any, Optional, NamedTuple

logger = logging.getLogger(__name__)


class Minutia(NamedTuple):
    x: float
    y: float
    angle: float           # radians, [0, 2*pi)
    mtype: str             # "ending" or "bifurcation"


class MinutiaeMatcher:
    """
    Extract and match minutiae using alignment-free methods.

    Usage:
        mm = MinutiaeMatcher()
        score, details = mm.match(enhanced_probe, enhanced_gallery,
                                  probe_mask, gallery_mask)
    """

    def __init__(
        self,
        # Extraction parameters
        border_margin: int = 15,
        min_minutiae: int = 5,
        min_distance: int = 10,
        # Hough voting parameters
        hough_tx_bin: float = 15.0,
        hough_ty_bin: float = 15.0,
        hough_r_bin: float = np.pi / 12,   # 15 degrees
        # Local structure parameters
        local_k: int = 5,                  # neighbors for descriptor
        local_desc_thresh: float = 1.5,    # max descriptor distance for match
        # Aligned matching parameters (when SIFT affine available)
        max_spatial_dist: float = 25.0,
        max_angle_diff: float = np.pi / 6,
    ):
        self.border_margin = border_margin
        self.min_minutiae = min_minutiae
        self.min_distance = min_distance

        self.hough_tx_bin = hough_tx_bin
        self.hough_ty_bin = hough_ty_bin
        self.hough_r_bin = hough_r_bin

        self.local_k = local_k
        self.local_desc_thresh = local_desc_thresh

        self.max_spatial_dist = max_spatial_dist
        self.max_angle_diff = max_angle_diff

    # ----- public --------------------------------------------------------

    def match(
        self,
        probe_img: np.ndarray,
        gallery_img: np.ndarray,
        probe_mask: Optional[np.ndarray] = None,
        gallery_mask: Optional[np.ndarray] = None,
        alignment_matrix: Optional[np.ndarray] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Match probe against gallery using minutiae.

        Three strategies are tried; the best score wins:
        1. Hough voting (alignment-free)
        2. Local structure descriptors (alignment-free)
        3. Aligned Hungarian matching (if alignment_matrix provided)
        """
        m1 = self._extract(probe_img, probe_mask)
        m2 = self._extract(gallery_img, gallery_mask)

        logger.debug(f"Minutiae: probe={len(m1)}, gallery={len(m2)}")

        if len(m1) < self.min_minutiae or len(m2) < self.min_minutiae:
            return 0.0, {
                "reason": "too_few_minutiae",
                "n_probe": len(m1),
                "n_gallery": len(m2),
            }

        # Strategy 1: Hough voting (alignment-free)
        s_hough, n_hough = self._hough_match(m1, m2)

        # Strategy 2: Local structure descriptors (alignment-free)
        s_local, n_local = self._local_structure_match(m1, m2)

        # Strategy 3: Aligned Hungarian (only if SIFT provided alignment)
        s_aligned = 0.0
        if alignment_matrix is not None:
            m1t = self._transform_minutiae(m1, alignment_matrix)
            s_aligned, _ = self._aligned_hungarian(m1t, m2)

        # Take the best score
        score = max(s_hough, s_local, s_aligned)
        best_method = "hough"
        if score == s_local:
            best_method = "local_structure"
        if score == s_aligned and alignment_matrix is not None:
            best_method = "aligned"

        details = {
            "n_probe": len(m1),
            "n_gallery": len(m2),
            "hough_score": round(s_hough, 4),
            "hough_matched": n_hough,
            "local_score": round(s_local, 4),
            "local_matched": n_local,
            "aligned_score": round(s_aligned, 4),
            "best_method": best_method,
        }

        logger.debug(
            f"Minutiae: hough={s_hough:.4f}({n_hough}) "
            f"local={s_local:.4f}({n_local}) "
            f"aligned={s_aligned:.4f} -> best={score:.4f} ({best_method})"
        )
        return score, details

    # ----- Strategy 1: Hough voting --------------------------------------

    def _hough_match(
        self,
        m1: List[Minutia],
        m2: List[Minutia],
    ) -> Tuple[float, int]:
        """
        Generalized Hough Transform for minutiae matching.

        For each (probe, gallery) minutia pair, compute the rigid transform
        that would align them, and vote in a discretised transform space.
        The peak in the accumulator = best alignment = most consistent matches.

        Returns: (score, n_matched)
        """
        n1, n2 = len(m1), len(m2)

        tx_bin = self.hough_tx_bin
        ty_bin = self.hough_ty_bin
        r_bin = self.hough_r_bin

        accumulator: Dict[Tuple[int, int, int], List[Tuple[int, int]]] = defaultdict(list)

        for i in range(n1):
            for j in range(n2):
                # Compute rigid transform aligning m1[i] -> m2[j]
                dtheta = _angle_diff(m2[j].angle, m1[i].angle)

                # Rotate probe minutia position by dtheta
                cos_t = np.cos(dtheta)
                sin_t = np.sin(dtheta)
                rx = m1[i].x * cos_t - m1[i].y * sin_t
                ry = m1[i].x * sin_t + m1[i].y * cos_t

                dx = m2[j].x - rx
                dy = m2[j].y - ry

                # Discretise into bins
                bx = int(round(dx / tx_bin))
                by = int(round(dy / ty_bin))
                br = int(round(dtheta / r_bin))

                accumulator[(bx, by, br)].append((i, j))

        if not accumulator:
            return 0.0, 0

        # Find the peak bin
        best_bin = max(accumulator, key=lambda k: len(accumulator[k]))

        # Collect votes from the peak and its 3x3x3 neighbourhood
        bx0, by0, br0 = best_bin
        all_pairs: List[Tuple[int, int]] = []
        for dbx in [-1, 0, 1]:
            for dby in [-1, 0, 1]:
                for dbr in [-1, 0, 1]:
                    key = (bx0 + dbx, by0 + dby, br0 + dbr)
                    if key in accumulator:
                        all_pairs.extend(accumulator[key])

        # Greedy one-to-one assignment (keep first seen)
        used_probe = set()
        used_gallery = set()
        unique_pairs: List[Tuple[int, int]] = []
        for pi, gj in all_pairs:
            if pi not in used_probe and gj not in used_gallery:
                unique_pairs.append((pi, gj))
                used_probe.add(pi)
                used_gallery.add(gj)

        n_matched = len(unique_pairs)

        if n_matched < 3:
            return 0.0, n_matched

        # Verify matched pairs: refine alignment and check residuals
        n_verified = self._verify_hough_matches(m1, m2, unique_pairs)

        if n_verified < 3:
            return 0.0, n_verified

        # Score
        min_count = min(n1, n2)
        match_ratio = n_verified / max(min_count, 1)
        count_bonus = min(1.0, np.log1p(n_verified) / np.log1p(20))

        # Discrimination bonus: how concentrated are the votes?
        peak_votes = len(all_pairs)
        total_votes = n1 * n2
        concentration = peak_votes / max(total_votes, 1)
        # A true match has high concentration; random has low
        conc_bonus = min(1.0, concentration * 50)

        score = (
            0.35 * match_ratio
            + 0.30 * count_bonus
            + 0.35 * conc_bonus
        )
        score = float(np.clip(score, 0.0, 1.0))

        return score, n_verified

    def _verify_hough_matches(
        self,
        m1: List[Minutia],
        m2: List[Minutia],
        pairs: List[Tuple[int, int]],
    ) -> int:
        """
        Verify matched pairs by estimating the best rigid transform
        from them and counting how many pairs have small residuals.
        """
        if len(pairs) < 2:
            return len(pairs)

        # Estimate average transform from all pairs
        dthetas = []
        dxs = []
        dys = []
        for pi, gj in pairs:
            dt = _angle_diff(m2[gj].angle, m1[pi].angle)
            dthetas.append(dt)

        avg_theta = float(np.mean(dthetas))
        cos_t = np.cos(avg_theta)
        sin_t = np.sin(avg_theta)

        for pi, gj in pairs:
            rx = m1[pi].x * cos_t - m1[pi].y * sin_t
            ry = m1[pi].x * sin_t + m1[pi].y * cos_t
            dxs.append(m2[gj].x - rx)
            dys.append(m2[gj].y - ry)

        avg_dx = float(np.median(dxs))
        avg_dy = float(np.median(dys))

        # Count verified: transform probe minutia and check distance
        verified = 0
        for pi, gj in pairs:
            rx = m1[pi].x * cos_t - m1[pi].y * sin_t + avg_dx
            ry = m1[pi].x * sin_t + m1[pi].y * cos_t + avg_dy
            dist = np.sqrt((rx - m2[gj].x) ** 2 + (ry - m2[gj].y) ** 2)
            angle_err = abs(_angle_diff(m1[pi].angle + avg_theta, m2[gj].angle))
            if dist < self.max_spatial_dist and angle_err < self.max_angle_diff:
                verified += 1

        return verified

    # ----- Strategy 2: Local structure descriptors -----------------------

    def _local_structure_match(
        self,
        m1: List[Minutia],
        m2: List[Minutia],
    ) -> Tuple[float, int]:
        """
        Match minutiae by comparing local neighbourhood descriptors.

        For each minutia, compute a descriptor based on the relative
        positions and ridge directions of its K nearest neighbours.
        These descriptors are invariant to global translation and rotation.

        Returns: (score, n_matched)
        """
        K = self.local_k

        desc1 = self._compute_local_descriptors(m1, K)
        desc2 = self._compute_local_descriptors(m2, K)

        # Filter out None descriptors (too few neighbours)
        valid1 = [(i, d) for i, d in enumerate(desc1) if d is not None]
        valid2 = [(j, d) for j, d in enumerate(desc2) if d is not None]

        if len(valid1) < 3 or len(valid2) < 3:
            return 0.0, 0

        # Build descriptor matrices
        idx1 = [i for i, _ in valid1]
        mat1 = np.array([d for _, d in valid1])
        idx2 = [j for j, _ in valid2]
        mat2 = np.array([d for _, d in valid2])

        # Compute pairwise distances
        # mat1 is (N1, D), mat2 is (N2, D)
        # Use broadcasting: (N1, 1, D) - (1, N2, D) -> (N1, N2, D)
        diff = mat1[:, None, :] - mat2[None, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))  # (N1, N2)

        # Greedy matching: for each probe descriptor, find best gallery match
        matched = 0
        used_gallery = set()
        total_quality = 0.0

        # Sort by best match distance
        flat_idx = np.argsort(dists, axis=None)
        for flat in flat_idx:
            i = int(flat // dists.shape[1])
            j = int(flat % dists.shape[1])
            if dists[i, j] > self.local_desc_thresh:
                break
            if i in {m[0] for m in []} or j in used_gallery:
                continue
            # Check if this probe minutia already matched
            # (use a set for probe too)
            pass

        # Simpler approach: Hungarian on descriptor distances
        if dists.shape[0] > 0 and dists.shape[1] > 0:
            row_idx, col_idx = linear_sum_assignment(dists)
            for r, c in zip(row_idx, col_idx):
                if dists[r, c] < self.local_desc_thresh:
                    matched += 1
                    total_quality += 1.0 - dists[r, c] / self.local_desc_thresh

        if matched < 3:
            return 0.0, matched

        min_count = min(len(valid1), len(valid2))
        match_ratio = matched / max(min_count, 1)
        avg_quality = total_quality / matched if matched > 0 else 0.0
        count_bonus = min(1.0, np.log1p(matched) / np.log1p(20))

        score = (
            0.35 * match_ratio
            + 0.35 * avg_quality
            + 0.30 * count_bonus
        )
        score = float(np.clip(score, 0.0, 1.0))

        return score, matched

    def _compute_local_descriptors(
        self,
        minutiae: List[Minutia],
        K: int,
    ) -> List[Optional[np.ndarray]]:
        """
        For each minutia, compute a rotation/translation-invariant
        descriptor based on its K nearest neighbours.

        The descriptor consists of, for each neighbour (sorted by
        relative angle):
            - Normalised distance (distance / median_distance)
            - Relative angle (direction to neighbour minus minutia's
              own ridge direction)
            - Relative ridge direction (neighbour's direction minus
              minutia's direction)
        """
        n = len(minutiae)
        if n < K + 1:
            return [None] * n

        # Precompute pairwise distances
        positions = np.array([[m.x, m.y] for m in minutiae])
        # (N, N) distance matrix
        diff = positions[:, None, :] - positions[None, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))

        descriptors: List[Optional[np.ndarray]] = []

        for i in range(n):
            # Sort neighbours by distance
            dists = dist_matrix[i].copy()
            dists[i] = np.inf  # exclude self
            nearest_idx = np.argsort(dists)[:K]

            if dists[nearest_idx[-1]] == np.inf:
                # Not enough valid neighbours
                descriptors.append(None)
                continue

            # Median distance for normalisation
            neighbor_dists = dists[nearest_idx]
            median_dist = float(np.median(neighbor_dists))
            if median_dist < 1.0:
                median_dist = 1.0

            features = []
            for j_idx in nearest_idx:
                m_j = minutiae[j_idx]
                m_i = minutiae[i]

                # Normalised distance
                rel_dist = float(dists[j_idx]) / median_dist

                # Angle from minutia i to neighbour j, relative to i's direction
                direction_to_j = np.arctan2(m_j.y - m_i.y, m_j.x - m_i.x)
                rel_angle = (direction_to_j - m_i.angle) % (2 * np.pi)

                # Relative ridge direction
                rel_dir = (m_j.angle - m_i.angle) % (2 * np.pi)

                features.append((rel_angle, rel_dist, rel_dir))

            # Sort by relative angle for canonical ordering
            features.sort(key=lambda f: f[0])

            descriptor = np.array([f for feat in features for f in feat],
                                  dtype=np.float64)
            descriptors.append(descriptor)

        return descriptors

    # ----- Strategy 3: Aligned Hungarian (when SIFT provides alignment) --

    def _aligned_hungarian(
        self,
        m1: List[Minutia],
        m2: List[Minutia],
    ) -> Tuple[float, Dict]:
        """
        Classical Hungarian matching on aligned minutiae.
        m1 should already be transformed into gallery coordinates.
        """
        n1, n2 = len(m1), len(m2)
        BIG = 1e6

        C = np.full((n1, n2), BIG, dtype=np.float64)

        for i in range(n1):
            for j in range(n2):
                dx = m1[i].x - m2[j].x
                dy = m1[i].y - m2[j].y
                spatial = np.sqrt(dx * dx + dy * dy)

                if spatial > self.max_spatial_dist:
                    continue

                ad = abs(_angle_diff(m1[i].angle, m2[j].angle))
                if ad > self.max_angle_diff:
                    continue

                C[i, j] = (
                    spatial / self.max_spatial_dist * 0.6
                    + ad / self.max_angle_diff * 0.4
                )

        row_idx, col_idx = linear_sum_assignment(C)

        good_mask = C[row_idx, col_idx] < (BIG * 0.5)
        matched_costs = C[row_idx[good_mask], col_idx[good_mask]]
        n_matched = int(good_mask.sum())

        if n_matched < 3:
            return 0.0, {}

        avg_quality = float(1.0 - np.mean(matched_costs))
        match_ratio = n_matched / min(n1, n2)
        count_bonus = min(1.0, np.log1p(n_matched) / np.log1p(30))

        score = (
            0.40 * avg_quality
            + 0.30 * match_ratio
            + 0.30 * count_bonus
        )
        return float(np.clip(score, 0.0, 1.0)), {"n_matched": n_matched}

    # ----- extraction ----------------------------------------------------

    def _extract(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> List[Minutia]:
        """Extract minutiae from an enhanced fingerprint image."""
        # Binarize with Otsu
        _, bw = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Skeletonise
        skel = self._skeletonize(bw)

        h, w = skel.shape
        margin = self.border_margin

        minutiae: List[Minutia] = []
        padded = np.pad(skel, 1, constant_values=0)

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if skel[y, x] == 0:
                    continue
                if x < margin or x >= w - margin or y < margin or y >= h - margin:
                    continue
                if mask is not None and not mask[y, x]:
                    continue

                py, px = y + 1, x + 1
                neighbors = [
                    padded[py - 1, px],
                    padded[py - 1, px + 1],
                    padded[py, px + 1],
                    padded[py + 1, px + 1],
                    padded[py + 1, px],
                    padded[py + 1, px - 1],
                    padded[py, px - 1],
                    padded[py - 1, px - 1],
                ]

                cn = sum(
                    1 for k in range(8)
                    if neighbors[k] == 0 and neighbors[(k + 1) % 8] == 1
                )

                mtype = None
                if cn == 1:
                    mtype = "ending"
                elif cn == 3:
                    mtype = "bifurcation"

                if mtype is not None:
                    angle = self._estimate_angle(skel, x, y, mtype)
                    minutiae.append(Minutia(float(x), float(y), angle, mtype))

        # Proximity filter — keep best-spaced subset
        minutiae = self._filter_proximity(minutiae)

        return minutiae

    def _skeletonize(self, bw: np.ndarray) -> np.ndarray:
        """Zhang-Suen thinning with multiple fallbacks."""
        try:
            skel = cv2.ximgproc.thinning(
                bw, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
            )
            return (skel > 0).astype(np.uint8)
        except (AttributeError, cv2.error):
            pass

        try:
            from skimage.morphology import skeletonize
            binary = bw > 0
            skel = skeletonize(binary).astype(np.uint8)
            return skel
        except ImportError:
            pass

        # Last resort: iterative morphological skeleton via OpenCV
        img = bw.copy()
        skel = np.zeros_like(img)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
            temp = cv2.subtract(img, opened)
            skel = cv2.bitwise_or(skel, temp)
            img = cv2.erode(img, element)
            if cv2.countNonZero(img) == 0:
                break
        return (skel > 0).astype(np.uint8)

    def _estimate_angle(
        self,
        skel: np.ndarray,
        x: int,
        y: int,
        mtype: str,
    ) -> float:
        """Trace the ridge a few pixels to estimate the minutia angle."""
        h, w = skel.shape
        cx, cy = x, y
        visited = {(cx, cy)}
        trace_len = 12

        for _ in range(trace_len):
            found = False
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = cx + dx, cy + dy
                    if (0 <= nx < w and 0 <= ny < h
                            and skel[ny, nx] > 0
                            and (nx, ny) not in visited):
                        visited.add((nx, ny))
                        cx, cy = nx, ny
                        found = True
                        break
                if found:
                    break
            if not found:
                break

        if len(visited) >= 4:
            return float(np.arctan2(y - cy, x - cx)) % (2 * np.pi)
        return 0.0

    @staticmethod
    def _transform_minutiae(
        minutiae: List[Minutia],
        M: np.ndarray,
    ) -> List[Minutia]:
        """Apply a 2x3 affine transform to minutiae positions and angles."""
        if not minutiae:
            return minutiae

        pts = np.array([[m.x, m.y] for m in minutiae], dtype=np.float64)
        ones = np.ones((len(pts), 1), dtype=np.float64)
        pts_h = np.hstack([pts, ones])
        transformed = pts_h @ M.T

        rot_angle = float(np.arctan2(M[1, 0], M[0, 0]))

        result = []
        for i, m in enumerate(minutiae):
            new_angle = (m.angle + rot_angle) % (2 * np.pi)
            result.append(Minutia(
                x=float(transformed[i, 0]),
                y=float(transformed[i, 1]),
                angle=new_angle,
                mtype=m.mtype,
            ))
        return result

    def _filter_proximity(self, minutiae: List[Minutia]) -> List[Minutia]:
        """Remove minutiae that are too close together."""
        accepted: List[Minutia] = []
        for m in minutiae:
            too_close = False
            for a in accepted:
                if np.sqrt((m.x - a.x) ** 2 + (m.y - a.y) ** 2) < self.min_distance:
                    too_close = True
                    break
            if not too_close:
                accepted.append(m)
        return accepted


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _angle_diff(a: float, b: float) -> float:
    """Signed angle difference in [-pi, pi]."""
    d = a - b
    while d > np.pi:
        d -= 2 * np.pi
    while d < -np.pi:
        d += 2 * np.pi
    return d
