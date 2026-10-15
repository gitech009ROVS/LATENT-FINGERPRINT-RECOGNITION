"""
Fingerprint image enhancement pipeline.

Applies a chain of classical enhancement steps that improve ridge
visibility in both clean rolled prints and noisy latent prints:

    1. Normalize to zero-mean / unit-variance.
    2. CLAHE (Contrast-Limited Adaptive Histogram Equalization).
    3. Gabor filter bank tuned to the dominant ridge frequency.

The enhanced image is what every downstream matcher receives.
"""

from __future__ import annotations

import numpy as np
import cv2
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enhance(
    image: np.ndarray,
    clahe_clip: float = 3.0,
    clahe_grid: int = 8,
    gabor_ksize: int = 21,
    gabor_frequencies: Tuple[float, ...] = (0.08, 0.10, 0.12),
    gabor_n_orientations: int = 8,
) -> np.ndarray:
    """
    Full enhancement pipeline — returns a single-channel uint8 image.

    Args:
        image:  Grayscale input (any depth, any range).
        clahe_clip:  CLAHE clip limit.
        clahe_grid:  CLAHE tile grid size.
        gabor_ksize:  Gabor kernel size.
        gabor_frequencies:  Set of spatial frequencies (cycles/pixel) to
            try.  The filter response is max-pooled across them so we
            don't need to estimate the exact ridge frequency first.
        gabor_n_orientations:  Number of Gabor orientations (evenly
            spaced over [0, pi)).

    Returns:
        Enhanced grayscale image (uint8, 0–255).
    """
    gray = _to_gray(image)

    # Step 1 — global normalization
    norm = _normalize(gray)

    # Step 2 — CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip,
        tileGridSize=(clahe_grid, clahe_grid),
    )
    cl = clahe.apply(norm)

    # Step 3 — Gabor filter bank
    enhanced = _gabor_enhance(
        cl,
        ksize=gabor_ksize,
        frequencies=gabor_frequencies,
        n_orientations=gabor_n_orientations,
    )

    logger.debug(
        f"enhance: input {gray.shape}, output range "
        f"[{enhanced.min()}, {enhanced.max()}]"
    )
    return enhanced


def enhance_light(
    image: np.ndarray,
    clahe_clip: float = 3.0,
    clahe_grid: int = 8,
) -> np.ndarray:
    """
    Light enhancement — normalize + CLAHE only (NO Gabor).

    Used for SIFT/ORB keypoint detection because the Gabor bank
    homogenises ridge texture and destroys the unique local features
    that keypoint matchers need.
    """
    gray = _to_gray(image)
    norm = _normalize(gray)
    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip,
        tileGridSize=(clahe_grid, clahe_grid),
    )
    out = clahe.apply(norm)
    logger.debug(f"enhance_light: {gray.shape}, range [{out.min()}, {out.max()}]")
    return out


def segment_roi(
    image: np.ndarray,
    block_size: int = 16,
    min_coverage: float = 0.15,
) -> np.ndarray:
    """
    Adaptive ROI segmentation.

    Returns a binary mask (True = fingerprint region).  Uses gradient-
    magnitude with an Otsu-style adaptive threshold so it works on both
    high-contrast rolled prints and faint latent prints.
    """
    gray = _to_gray(image).astype(np.float64) / 255.0
    h, w = gray.shape

    # Gradient magnitude — captures edges even in low-contrast images
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)

    nr = h // block_size
    nc = w // block_size
    block_energy = np.zeros((nr, nc), dtype=np.float64)

    for i in range(nr):
        for j in range(nc):
            r0, c0 = i * block_size, j * block_size
            block = mag[r0:r0 + block_size, c0:c0 + block_size]
            block_energy[i, j] = np.mean(block)

    # Otsu threshold on block energies
    flat = block_energy.ravel()
    if flat.max() - flat.min() < 1e-10:
        return np.ones((h, w), dtype=bool)

    # Use OpenCV's Otsu on the block energy histogram
    energy_u8 = np.clip(
        (flat - flat.min()) / (flat.max() - flat.min()) * 255, 0, 255
    ).astype(np.uint8)
    thr_val, _ = cv2.threshold(energy_u8, 0, 255, cv2.THRESH_OTSU)
    thr = flat.min() + (thr_val / 255.0) * (flat.max() - flat.min())

    # Build pixel mask
    mask = np.zeros((h, w), dtype=bool)
    for i in range(nr):
        for j in range(nc):
            if block_energy[i, j] > thr:
                r0, c0 = i * block_size, j * block_size
                mask[r0:r0 + block_size, c0:c0 + block_size] = True

    # Morphological cleanup
    kern = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (block_size * 2, block_size * 2)
    )
    mask_u8 = (mask.astype(np.uint8) * 255)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kern)
    mask_u8 = cv2.dilate(
        mask_u8,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (block_size, block_size)),
    )
    mask = mask_u8 > 0

    coverage = mask.sum() / (h * w)
    logger.debug(f"segment_roi: coverage={coverage:.1%}")

    # Fallback: if still too small, accept everything
    if coverage < min_coverage:
        logger.debug("segment_roi: coverage below minimum — using full image")
        return np.ones((h, w), dtype=bool)

    return mask


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _normalize(gray: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance, rescaled to 0–255 uint8."""
    f = gray.astype(np.float64)
    mu, sigma = f.mean(), f.std()
    if sigma < 1e-6:
        return gray
    f = (f - mu) / sigma
    lo, hi = f.min(), f.max()
    if hi - lo < 1e-6:
        return gray
    f = (f - lo) / (hi - lo) * 255
    return f.astype(np.uint8)


def _gabor_enhance(
    image: np.ndarray,
    ksize: int = 21,
    frequencies: Tuple[float, ...] = (0.08, 0.10, 0.12),
    n_orientations: int = 8,
) -> np.ndarray:
    """
    Gabor filter bank: max-pool over orientations and frequencies.

    This avoids having to estimate the ridge orientation and frequency
    fields accurately — the bank covers the plausible range and takes
    the best response at each pixel.
    """
    img = image.astype(np.float64)
    best = np.zeros_like(img)

    for freq in frequencies:
        lam = 1.0 / freq  # wavelength
        for k in range(n_orientations):
            theta = k * np.pi / n_orientations
            kern = cv2.getGaborKernel(
                (ksize, ksize), sigma=4.0, theta=theta,
                lambd=lam, gamma=0.5, psi=0,
            )
            resp = cv2.filter2D(img, cv2.CV_64F, kern)
            best = np.maximum(best, resp)

    # Normalize to uint8
    lo, hi = best.min(), best.max()
    if hi - lo < 1e-6:
        return image
    out = ((best - lo) / (hi - lo) * 255).astype(np.uint8)
    return out
