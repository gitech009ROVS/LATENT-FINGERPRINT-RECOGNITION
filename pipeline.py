"""
Parallel matching pipeline — v4 with min-rank fusion.

Orchestrates a 4-channel matching system:
    Channel 1: SIFT + RANSAC   (keypoint matching — tiebreaker only)
    Channel 2: Gabor-POC       (phase correlation on Gabor-enhanced images)
    Channel 3: CLAHE-POC       (phase correlation on CLAHE-enhanced images)
    Channel 4: SAFIS/BOZORTH3  (minutiae matching via afis library)

SIFT and both POC channels run in parallel workers.  The minutiae channel
(SAFIS or BOZORTH3) runs on the main thread using pre-cached gallery
templates.  Final fusion uses min-rank: for each gallery image, take its
best rank across channels 2-4 (the channel where it ranks highest), with
SIFT z-score as a minor tiebreaker.

Projected accuracy (15 probes, 317 gallery):
    Rank-1: 60%   Rank-10: 73%   Rank-20: 87%   Rank-50: 100%
"""

from __future__ import annotations

import os
import time
import logging
import numpy as np
import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple

from .enhance import enhance, enhance_light, segment_roi
from .matching.sift_matcher import SIFTMatcher
from .matching.texture_matcher import TextureMatcher
from .matching.nbis_matcher import NbisMatcher
from .matching.fusion import fuse_all_scores_rank

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker setup (runs once per process)
# ---------------------------------------------------------------------------

_worker_sift: Optional[SIFTMatcher] = None
_worker_text: Optional[TextureMatcher] = None


def _init_worker(log_path: Optional[str] = None):
    """Initialise per-process singletons and logging."""
    global _worker_sift, _worker_text
    _worker_sift = SIFTMatcher()
    _worker_text = TextureMatcher()

    if log_path:
        root = logging.getLogger()
        root.setLevel(logging.DEBUG)
        for h in root.handlers[:]:
            root.removeHandler(h)
            h.close()
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        root.addHandler(fh)


def _match_sift_poc(args: Tuple) -> Dict[str, Any]:
    """
    Worker function: match one gallery image with SIFT + dual-POC.

    Returns a dict with gallery_id, sift_score, gabor_poc_score,
    clahe_poc_score, and detail dicts.
    """
    (probe_light, probe_gabor, probe_clahe,
     probe_mask_bytes, probe_shape,
     gallery_path, gallery_id) = args

    probe_mask = None
    if probe_mask_bytes is not None:
        probe_mask = np.frombuffer(probe_mask_bytes, dtype=bool).reshape(
            probe_shape
        )

    t0 = time.perf_counter()

    try:
        gallery_raw = cv2.imread(gallery_path, cv2.IMREAD_GRAYSCALE)
        if gallery_raw is None:
            return {
                "gallery_id": gallery_id,
                "gallery_path": gallery_path,
                "sift_score": 0.0,
                "gabor_poc_score": 0.0,
                "clahe_poc_score": 0.0,
                "error": f"Cannot read {gallery_path}",
            }

        gallery_light = enhance_light(gallery_raw)
        gallery_gabor = enhance(gallery_raw)
        gallery_mask = segment_roi(gallery_raw)
        gallery_mask_u8 = gallery_mask.astype(np.uint8) * 255

        probe_mask_u8 = None
        if probe_mask is not None:
            probe_mask_u8 = probe_mask.astype(np.uint8) * 255

        # Channel 1: SIFT on CLAHE images
        s_sift, d_sift = 0.0, {}
        affine = None
        try:
            s_sift, d_sift = _worker_sift.match(
                probe_light, gallery_light, probe_mask_u8, gallery_mask_u8
            )
            affine = (
                np.array(d_sift["affine_matrix"])
                if d_sift.get("affine_matrix") is not None
                else None
            )
        except Exception as e:
            logger.warning(f"[{gallery_id}] SIFT error: {e}")

        # Channels 2a+2b: Dual POC (Gabor + CLAHE)
        s_gabor_poc, s_clahe_poc = 0.0, 0.0
        d_text = {}
        try:
            _, d_text = _worker_text.match(
                probe_gabor, gallery_gabor, affine,
                probe_mask, gallery_mask,
                probe_clahe=probe_clahe,
                gallery_clahe=gallery_light,   # gallery CLAHE = enhance_light
            )
            s_gabor_poc = d_text.get("poc_score", 0.0)
            s_clahe_poc = d_text.get("poc_clahe_score", 0.0)
        except Exception as e:
            logger.warning(f"[{gallery_id}] POC error: {e}")

        elapsed = time.perf_counter() - t0

        return {
            "gallery_id": gallery_id,
            "gallery_path": gallery_path,
            "sift_score": s_sift,
            "gabor_poc_score": s_gabor_poc,
            "clahe_poc_score": s_clahe_poc,
            "sift_details": d_sift,
            "texture_details": d_text,
            "elapsed": elapsed,
        }

    except Exception as e:
        logger.exception(f"[{gallery_id}] ERROR: {e}")
        return {
            "gallery_id": gallery_id,
            "gallery_path": gallery_path,
            "sift_score": 0.0,
            "gabor_poc_score": 0.0,
            "clahe_poc_score": 0.0,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Public pipeline
# ---------------------------------------------------------------------------

class MatchingPipeline:
    """
    4-channel probe-vs-gallery matching with parallel workers.

    Usage:
        pipeline = MatchingPipeline()
        results = pipeline.run(
            probe_path="path/to/latent.png",
            gallery_dir="path/to/gallery/",
        )
    """

    def __init__(
        self,
        n_workers: int = 4,
        log_path: Optional[str] = None,
        nbis_cache_dir: Optional[str] = None,
    ):
        self.n_workers = n_workers
        self.log_path = log_path
        self.nbis_cache_dir = nbis_cache_dir
        self._nbis_matcher = NbisMatcher(cache_dir=nbis_cache_dir)
        self._gallery_preloaded = False

    def preload_gallery(self, gallery_dir: str,
                        progress_callback=None) -> int:
        """
        Pre-extract minutiae templates for all gallery images.

        This is optional — the pipeline will extract templates on-the-fly
        if not called, but preloading is much faster for repeated matching.
        """
        gallery_dir = Path(gallery_dir)
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        gallery_files = sorted(
            str(p) for p in gallery_dir.iterdir()
            if p.suffix.lower() in exts and p.is_file()
        )

        if not gallery_files:
            return 0

        n = self._nbis_matcher.preload_gallery(gallery_files, progress_callback)
        self._gallery_preloaded = True
        return n

    def run(
        self,
        probe_path: str,
        gallery_dir: str,
        progress_callback=None,
    ) -> List[Dict[str, Any]]:
        """
        Match one probe against every image in the gallery directory.

        Steps:
            1. Compute minutiae (SAFIS/BZ) scores for probe vs all gallery
               (sequential, using cached gallery templates).
            2. Run SIFT + Gabor-POC + CLAHE-POC for each gallery image
               in parallel workers.
            3. Min-rank fusion across minutiae, Gabor-POC, CLAHE-POC
               with SIFT as tiebreaker.

        Returns:
            List of result dicts sorted by fused_score (best first).
        """
        # 1. Load and enhance the probe
        logger.info(f"Loading probe: {probe_path}")
        probe_raw = cv2.imread(probe_path, cv2.IMREAD_GRAYSCALE)
        if probe_raw is None:
            raise FileNotFoundError(f"Cannot read probe image: {probe_path}")

        probe_light = enhance_light(probe_raw)
        probe_gabor = enhance(probe_raw)
        probe_clahe = probe_light  # CLAHE-only = enhance_light output
        probe_mask = segment_roi(probe_raw)
        probe_mask_bytes = probe_mask.tobytes()
        probe_shape = probe_raw.shape[:2]

        # 2. Discover gallery images
        gallery_dir_path = Path(gallery_dir)
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        gallery_files = sorted(
            p for p in gallery_dir_path.iterdir()
            if p.suffix.lower() in exts and p.is_file()
        )

        if not gallery_files:
            raise ValueError(f"No images found in {gallery_dir}")

        total = len(gallery_files)
        logger.info(f"Matching against {total} gallery images (4 channels)")

        # 3. Minutiae channel (sequential, fast with cache)
        minutiae_available = self._nbis_matcher.available
        if minutiae_available:
            logger.info(f"Computing minutiae scores ({self._nbis_matcher.backend_name})...")
            t0_min = time.perf_counter()

            # Pre-extract gallery if not done yet
            if not self._gallery_preloaded:
                self._nbis_matcher.preload_gallery(
                    [str(gf) for gf in gallery_files],
                    progress_callback=None,
                )
                self._gallery_preloaded = True

            # Compute scores for this probe against all gallery
            min_results = self._nbis_matcher.match_against_gallery(
                probe_raw, [str(gf) for gf in gallery_files]
            )
            minutiae_scores_array = np.array([r[0] for r in min_results])

            min_elapsed = time.perf_counter() - t0_min
            logger.info(
                f"Minutiae done in {min_elapsed:.1f}s, "
                f"max={minutiae_scores_array.max():.4f}, "
                f"backend={self._nbis_matcher.backend_name}"
            )
        else:
            logger.warning(
                "Minutiae matching not available — running in 2-channel mode "
                "(Gabor-POC + CLAHE-POC only). Install afis for best accuracy."
            )
            min_results = [(0.0, {"error": "minutiae_unavailable"})] * total
            minutiae_scores_array = np.zeros(total)

        # 4. SIFT + dual-POC channels (parallel)
        logger.info("Computing SIFT + dual-POC scores in parallel...")
        work_items = []
        for gf in gallery_files:
            gid = gf.stem
            work_items.append((
                probe_light, probe_gabor, probe_clahe,
                probe_mask_bytes, probe_shape,
                str(gf), gid
            ))

        sift_poc_results: Dict[str, Dict] = {}
        done = 0

        # Use ThreadPoolExecutor instead of ProcessPoolExecutor.
        # ProcessPoolExecutor uses 'spawn' on Windows, which re-imports
        # the entire module tree in each child process — this deadlocks
        # with PyQt5.  ThreadPoolExecutor avoids that entirely, and still
        # parallelises well because OpenCV/numpy release the GIL.
        _init_worker(self.log_path)   # init once in the main worker thread context
        with ThreadPoolExecutor(
            max_workers=self.n_workers,
        ) as pool:
            futures = {
                pool.submit(_match_sift_poc, item): item[6]
                for item in work_items
            }
            for future in as_completed(futures):
                result = future.result()
                sift_poc_results[result["gallery_id"]] = result
                done += 1
                if progress_callback:
                    progress_callback(done, total)

        # 5. Collect all channel scores into arrays
        sift_scores_array = np.zeros(total)
        gabor_poc_array = np.zeros(total)
        clahe_poc_array = np.zeros(total)
        gallery_ids = []

        for i, gf in enumerate(gallery_files):
            gid = gf.stem
            gallery_ids.append(gid)
            sp = sift_poc_results.get(gid, {})
            sift_scores_array[i] = sp.get("sift_score", 0.0)
            gabor_poc_array[i] = sp.get("gabor_poc_score", 0.0)
            clahe_poc_array[i] = sp.get("clahe_poc_score", 0.0)

        # 6. Min-rank fusion across all channels
        logger.info("Performing min-rank fusion...")
        fused_scores, fusion_details = fuse_all_scores_rank(
            sift_scores_array,
            gabor_poc_array,
            minutiae_scores_array,
            poc_clahe_scores=clahe_poc_array,
        )

        # 7. Build result dicts
        results: List[Dict[str, Any]] = []
        for i, gf in enumerate(gallery_files):
            gid = gallery_ids[i]
            sp = sift_poc_results.get(gid, {})

            results.append({
                "gallery_id": gid,
                "gallery_path": str(gf),
                "fused_score": float(fused_scores[i]),
                "sift_score": float(sift_scores_array[i]),
                "minutiae_score": float(minutiae_scores_array[i]),
                "texture_score": float(gabor_poc_array[i]),
                "clahe_poc_score": float(clahe_poc_array[i]),
                "sift_details": sp.get("sift_details", {}),
                "minutiae_details": min_results[i][1] if i < len(min_results) else {},
                "texture_details": sp.get("texture_details", {}),
                "fusion_details": fusion_details,
            })

        # 8. Sort by fused score (best first)
        results.sort(key=lambda r: r["fused_score"], reverse=True)

        # Log top-5
        logger.info("===== Top 5 matches =====")
        for i, r in enumerate(results[:5]):
            logger.info(
                f"  Rank {i+1}: {r['gallery_id']}  "
                f"fused={r['fused_score']:.4f}  "
                f"SIFT={r.get('sift_score', 0):.4f}  "
                f"MIN={r.get('minutiae_score', 0):.4f}  "
                f"G-POC={r.get('texture_score', 0):.4f}  "
                f"C-POC={r.get('clahe_poc_score', 0):.4f}"
            )

        return results