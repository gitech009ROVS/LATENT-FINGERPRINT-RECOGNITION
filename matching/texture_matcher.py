"""
Channel 2 — Texture similarity via Phase-Only Correlation (POC).

v5: Dual-POC — computes POC on both Gabor-enhanced and CLAHE-only images.
Different probes respond to different preprocessing: Gabor-POC excels on
probes with clear ridge structure, CLAHE-POC excels on high-contrast probes.
Min-rank fusion across these channels is handled by the fusion module.

Strategy:
1. **Gabor-POC:** Phase-only correlation on Gabor-enhanced images.
   Best for probes with visible ridge patterns (most probes).

2. **CLAHE-POC:** Phase-only correlation on CLAHE-only enhanced images.
   Complementary to Gabor-POC — picks up different probes.

3. **SSIM on aligned overlap (optional boost):**
   When SIFT provides affine alignment, SSIM on overlap adds
   discriminative power (applied to Gabor-POC score only).
"""

from __future__ import annotations

import numpy as np
import cv2
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)


class TextureMatcher:
    """
    Phase correlation texture matching — dual-POC variant.
    """

    def __init__(
        self,
        # POC parameters
        poc_band_low: float = 0.05,
        poc_band_high: float = 0.45,
        poc_peak_radius: int = 5,
        # SSIM parameters
        ssim_win_size: int = 11,
        min_overlap_fraction: float = 0.10,
    ):
        self.poc_band_low = poc_band_low
        self.poc_band_high = poc_band_high
        self.poc_peak_radius = poc_peak_radius
        self.ssim_win_size = ssim_win_size
        self.min_overlap_fraction = min_overlap_fraction

    def match(
        self,
        probe_gabor: np.ndarray,
        gallery_gabor: np.ndarray,
        affine_matrix: Optional[np.ndarray] = None,
        probe_mask: Optional[np.ndarray] = None,
        gallery_mask: Optional[np.ndarray] = None,
        probe_clahe: Optional[np.ndarray] = None,
        gallery_clahe: Optional[np.ndarray] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute texture similarity using dual POC.

        Parameters
        ----------
        probe_gabor, gallery_gabor : Gabor-enhanced images (primary POC).
        affine_matrix : SIFT affine for SSIM alignment (optional).
        probe_mask, gallery_mask : ROI masks (optional).
        probe_clahe, gallery_clahe : CLAHE-only enhanced images (secondary POC).

        Returns
        -------
        score : float
            Gabor-POC score (primary, backward compatible).
        details : dict
            Contains poc_score (Gabor), poc_clahe_score, ssim_score.
        """
        # Channel 2a: POC on Gabor images (primary)
        s_poc = self._poc_match(probe_gabor, gallery_gabor)

        # Channel 2b: POC on CLAHE images (complementary)
        s_poc_clahe = 0.0
        if probe_clahe is not None and gallery_clahe is not None:
            s_poc_clahe = self._poc_match(probe_clahe, gallery_clahe)

        # SSIM boost if alignment available (applied to Gabor-POC)
        s_ssim = 0.0
        if affine_matrix is not None:
            s_ssim = self._ssim_match(probe_gabor, gallery_gabor,
                                       affine_matrix, probe_mask, gallery_mask)

        # Primary score = Gabor-POC (backward compatible)
        score = s_poc

        # SSIM boost when alignment quality is good
        if s_ssim > 0.3:
            score_with_ssim = 0.75 * s_poc + 0.25 * s_ssim
            score = max(score, score_with_ssim)  # never let SSIM hurt

        score = float(np.clip(score, 0.0, 1.0))

        details = {
            "poc_score": round(s_poc, 4),
            "poc_clahe_score": round(s_poc_clahe, 4),
            "ssim_score": round(s_ssim, 4),
            "method": "dual_poc",
        }

        logger.debug(
            f"Texture: gabor_poc={s_poc:.4f} clahe_poc={s_poc_clahe:.4f} "
            f"ssim={s_ssim:.4f} -> {score:.4f}"
        )
        return score, details

    # ----- Phase-Only Correlation -------------------------------------------

    def _poc_match(
        self,
        probe: np.ndarray,
        gallery: np.ndarray,
    ) -> float:
        """
        Phase-Only Correlation with band-pass filtering.
        PCE measures peak sharpness — sharp for true matches, flat for non-matches.
        """
        h = min(probe.shape[0], gallery.shape[0])
        w = min(probe.shape[1], gallery.shape[1])
        h = h & ~1
        w = w & ~1

        if h < 32 or w < 32:
            return 0.0

        p = cv2.resize(probe, (w, h), interpolation=cv2.INTER_AREA).astype(np.float64)
        g = cv2.resize(gallery, (w, h), interpolation=cv2.INTER_AREA).astype(np.float64)

        # Hanning window to reduce spectral leakage
        win_r = np.hanning(h).reshape(-1, 1)
        win_c = np.hanning(w).reshape(1, -1)
        window = win_r * win_c

        p = (p - p.mean()) * window
        g = (g - g.mean()) * window

        F1 = np.fft.fft2(p)
        F2 = np.fft.fft2(g)

        # Band-pass: keep ridge-frequency range
        fy = np.fft.fftfreq(h).reshape(-1, 1)
        fx = np.fft.fftfreq(w).reshape(1, -1)
        freq_mag = np.sqrt(fx**2 + fy**2)

        # Smooth Gaussian band-pass
        center = (self.poc_band_low + self.poc_band_high) / 2
        width = (self.poc_band_high - self.poc_band_low) / 2
        bp = np.exp(-((freq_mag - center) ** 2) / (2 * width ** 2))
        bp[freq_mag < self.poc_band_low * 0.5] = 0
        bp[freq_mag > self.poc_band_high * 1.5] = 0

        # Phase-only cross-power spectrum
        cross = F1 * np.conj(F2)
        mag = np.abs(cross)
        mag[mag < 1e-10] = 1e-10
        poc_spectrum = (cross / mag) * bp

        poc = np.real(np.fft.ifft2(poc_spectrum))

        # PCE: peak-to-correlation-energy ratio
        peak_val = poc.max()
        py, px = np.unravel_index(poc.argmax(), poc.shape)

        r = self.poc_peak_radius
        mask = np.ones_like(poc, dtype=bool)
        y_lo = max(0, py - r)
        y_hi = min(h, py + r + 1)
        x_lo = max(0, px - r)
        x_hi = min(w, px + r + 1)
        mask[y_lo:y_hi, x_lo:x_hi] = False

        noise = poc[mask]
        noise_std = noise.std()
        noise_mean = noise.mean()

        if noise_std < 1e-10:
            return 0.0

        pce = (peak_val - noise_mean) / noise_std

        # Sigmoid mapping: same finger PCE ~10-50, different ~2-6
        score = 1.0 / (1.0 + np.exp(-(pce - 6.0) / 2.0))
        return float(np.clip(score, 0.0, 1.0))

    # ----- SSIM (when alignment available) -----------------------------------

    def _ssim_match(
        self,
        probe: np.ndarray,
        gallery: np.ndarray,
        M: np.ndarray,
        probe_mask: Optional[np.ndarray],
        gallery_mask: Optional[np.ndarray],
    ) -> float:
        """Warp probe -> gallery frame, compute SSIM on overlap."""
        try:
            from skimage.metrics import structural_similarity as ssim
        except ImportError:
            return 0.0

        h, w = gallery.shape[:2]
        warped_probe = cv2.warpAffine(probe, M, (w, h))

        probe_valid = cv2.warpAffine(
            np.ones(probe.shape[:2], dtype=np.uint8) * 255,
            M, (w, h),
        ) > 0

        if probe_mask is not None:
            pm = probe_mask.astype(np.uint8) if probe_mask.dtype == bool else probe_mask
            warped_pmask = cv2.warpAffine(pm * 255, M, (w, h)) > 0
            probe_valid = probe_valid & warped_pmask

        gallery_valid = np.ones((h, w), dtype=bool)
        if gallery_mask is not None:
            gallery_valid = gallery_mask.astype(bool) if gallery_mask.dtype != bool else gallery_mask

        overlap = probe_valid & gallery_valid
        overlap_frac = overlap.sum() / (h * w)

        if overlap_frac < self.min_overlap_fraction:
            return 0.0

        rows = np.any(overlap, axis=1)
        cols = np.any(overlap, axis=0)
        if not rows.any() or not cols.any():
            return 0.0
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        crop_w = warped_probe[rmin:rmax + 1, cmin:cmax + 1]
        crop_g = gallery[rmin:rmax + 1, cmin:cmax + 1]

        if crop_w.shape[0] < self.ssim_win_size or crop_w.shape[1] < self.ssim_win_size:
            return 0.0

        try:
            score_val = ssim(
                crop_w, crop_g,
                win_size=self.ssim_win_size,
                data_range=255,
            )
        except Exception:
            return 0.0

        score = float(np.clip((score_val + 1.0) / 2.0, 0.0, 1.0))
        overlap_weight = min(1.0, overlap_frac / 0.3)
        return score * overlap_weight
