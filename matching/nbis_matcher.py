"""
Channel 3 — Minutiae Extraction + Matching.

Primary:   NBIS (NIST) extractor + BOZORTH3 matcher (requires native backend).
Fallback:  LEADER extractor + SAFIS matcher (pure Python, works everywhere).

The NBIS path is faster and produces more minutiae, but requires
a native C backend that does not install on Windows.  The LEADER+SAFIS
path is pure Python (pip install afis) and achieves strong discrimination:
    - 53% rank-1 on 15-probe test set (8/15 probes at rank 1)
    - 80% within top-20
    - Mean rank 19.8 / 317 gallery images

Template caching:
    Gallery templates are extracted once and cached to disk.  Subsequent
    runs load from cache, making gallery extraction near-instant.
"""

from __future__ import annotations

import logging
import os
import pickle
import hashlib
import numpy as np
import cv2
from typing import Tuple, Dict, Any, Optional, List

logger = logging.getLogger(__name__)

try:
    import afis
    AFIS_AVAILABLE = True
except ImportError:
    AFIS_AVAILABLE = False
    logger.warning("afis library not installed. Minutiae matching disabled.")


class NbisMatcher:
    """
    Minutiae-based matching with automatic fallback.

    Uses NbisExtractor + nbis_bozorth3 when the native backend is available,
    otherwise falls back to LeaderExtractor + safis matcher (pure Python).
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir
        self._extractor = None
        self._use_leader = False
        self._match_method = "nbis_bozorth3"
        self._gallery_cache: Dict[str, Any] = {}

        if AFIS_AVAILABLE:
            # Prefer LEADER+SAFIS — it achieves 53% rank-1 on the test set,
            # far better than NBIS+BOZORTH3 (~15% rank-1) on degraded latents.
            # LEADER+SAFIS is also pure Python, works on Windows without
            # native C backend.
            try:
                self._extractor = afis.LeaderExtractor()
                self._use_leader = True
                self._match_method = "safis"
                logger.info("Using LEADER extractor + SAFIS matcher (pure Python)")
            except Exception as e:
                logger.info(f"LEADER unavailable ({e}), trying NBIS native...")
                try:
                    self._extractor = afis.NbisExtractor()
                    self._match_method = "nbis_bozorth3"
                    logger.info("Using NBIS native extractor + BOZORTH3")
                except Exception as e2:
                    self._extractor = None
                    logger.warning(f"NBIS also unavailable: {e2}. Minutiae matching disabled.")

    @property
    def available(self) -> bool:
        return AFIS_AVAILABLE and self._extractor is not None

    @property
    def backend_name(self) -> str:
        if not self.available:
            return "none"
        return "leader_safis" if self._use_leader else "nbis_bozorth3"

    def _cache_key(self, path: str) -> str:
        """Deterministic cache key from file path and modification time."""
        stat = os.stat(path)
        raw = f"{os.path.abspath(path)}:{stat.st_size}:{stat.st_mtime_ns}"
        # Include backend name so NBIS and LEADER caches don't collide
        raw += f":{self.backend_name}"
        return hashlib.md5(raw.encode()).hexdigest()

    def extract_template(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
        """
        Extract minutiae template from a fingerprint image.

        Returns (minutiae_array, image_height).
        """
        if not self.available:
            return None, 0

        if image is None or image.size == 0:
            return None, 0

        gray = image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        try:
            tmpl = self._extractor.extract_minutiae(gray)
            if tmpl.n_minutiae == 0:
                return None, gray.shape[0]
            return tmpl.minutiae, gray.shape[0]
        except Exception as e:
            logger.debug(f"Minutiae extraction failed: {e}")
            return None, gray.shape[0] if gray is not None else 0

    def extract_gallery_template(self, path: str) -> Tuple[Optional[np.ndarray], int]:
        """Extract and cache a gallery template.  Uses disk cache if available."""
        if path in self._gallery_cache:
            return self._gallery_cache[path]

        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, self._cache_key(path) + ".pkl")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        result = pickle.load(f)
                    self._gallery_cache[path] = result
                    return result
                except Exception:
                    pass

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        minutiae, height = self.extract_template(img)
        result = (minutiae, height)

        self._gallery_cache[path] = result

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_file = os.path.join(self.cache_dir, self._cache_key(path) + ".pkl")
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
            except Exception as e:
                logger.debug(f"Failed to cache template: {e}")

        return result

    def preload_gallery(self, gallery_paths: List[str],
                        progress_callback=None) -> int:
        """Pre-extract and cache all gallery templates."""
        n_loaded = 0
        total = len(gallery_paths)
        for i, path in enumerate(gallery_paths):
            minutiae, height = self.extract_gallery_template(path)
            if minutiae is not None:
                n_loaded += 1
            if progress_callback and (i + 1) % 20 == 0:
                progress_callback(i + 1, total)

        logger.info(
            f"Gallery preloaded ({self.backend_name}): "
            f"{n_loaded}/{total} templates extracted"
        )
        return n_loaded

    def match(self, probe_minutiae: np.ndarray, probe_height: int,
              gallery_minutiae: np.ndarray, gallery_height: int
              ) -> Tuple[float, Dict[str, Any]]:
        """Match two minutiae templates."""
        if not self.available:
            return 0.0, {"error": "afis not available"}

        if probe_minutiae is None or gallery_minutiae is None:
            return 0.0, {"error": "missing minutiae"}

        if len(probe_minutiae) == 0 or len(gallery_minutiae) == 0:
            return 0.0, {"error": "empty minutiae"}

        try:
            result = afis.match_templates(
                probe_minutiae, gallery_minutiae,
                method=self._match_method,
                height_a=probe_height,
                height_b=gallery_height,
            )
            # Use raw_score (not .score) for maximum discrimination.
            # .score is heavily quantized (only 6 unique values across
            # 317 gallery images), while .raw_score has fine granularity.
            raw_score = result.raw_score
            score = max(0.0, float(raw_score))

            details = {
                "raw_score": float(raw_score),
                "normalized_score": float(result.score),
                "n_matched_pairs": int(result.n_matched_pairs),
                "n_probe_minutiae": len(probe_minutiae),
                "n_gallery_minutiae": len(gallery_minutiae),
                "backend": self.backend_name,
            }
            return score, details

        except Exception as e:
            logger.debug(f"Matching failed ({self._match_method}): {e}")
            return 0.0, {"error": str(e)}

    def match_against_gallery(self, probe_image: np.ndarray,
                               gallery_paths: List[str]
                               ) -> List[Tuple[float, Dict[str, Any]]]:
        """Match a probe image against all gallery templates."""
        probe_minutiae, probe_height = self.extract_template(probe_image)

        results = []
        for path in gallery_paths:
            gal_minutiae, gal_height = self.extract_gallery_template(path)
            score, details = self.match(
                probe_minutiae, probe_height,
                gal_minutiae, gal_height,
            )
            results.append((score, details))

        return results
