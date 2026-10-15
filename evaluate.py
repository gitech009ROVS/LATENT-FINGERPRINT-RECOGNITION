#!/usr/bin/env python3
"""
Batch evaluation script for the Latent Fingerprint Recognition System.

Usage:
    python -m fingerprint_matcher_v2.evaluate --probe-dir <latent_folder> --gallery-dir <gallery_folder>

The probe directory should contain subfolders, each with a "latent/" subfolder
holding the probe images. The gallery directory should contain clean fingerprint
images (.png).

Ground truth mapping:
    The correct gallery match for each probe is determined by the naming
    convention. For example, probe "1_5.png" in folder "VC2000DbsDb2a1"
    matches gallery image "VC2000DbsDb2a1_1_2.png" (same subject prefix).

Example:
    python -m fingerprint_matcher_v2.evaluate \
        --probe-dir "FingerDatasets/Generated_Latent_Fingerprint_Dataset-main/FVC-Latent_Test_Dataset/data" \
        --gallery-dir "FingerDatasets/clean" \
        --workers 4
"""

import argparse
import os
import sys
import time
import re
import numpy as np
from pathlib import Path

from .pipeline import MatchingPipeline


# Known probe-to-gallery ground truth mappings
GROUND_TRUTH = {
    # (folder, probe_file) -> gallery_id (without .png)
    ("VC2000DbsDb2a1",  "1_5.png"):  "VC2000DbsDb2a1_1_2",
    ("VC2000DbsDb2a4",  "4_3.png"):  "VC2000DbsDb2a4_4_1",
    ("VC2000DbsDb2a6",  "6_4.png"):  "VC2000DbsDb2a6_6_2",
    ("VC2000DbsDb2a8",  "8_8.png"):  "VC2000DbsDb2a8_8_2",
    ("VC2000DbsDb2a10", "10_6.png"): "VC2000DbsDb2a10_10_7",
    ("VC2000DbsDb2a15", "15_8.png"): "VC2000DbsDb2a15_15_4",
    ("VC2000DbsDb2a30", "30_8.png"): "VC2000DbsDb2a30_30_3",
    ("VC2000DbsDb2a50", "50_4.png"): "VC2000DbsDb2a50_50_8",
    ("VC2000DbsDb2a75", "75_2.png"): "VC2000DbsDb2a75_75_6",
    ("VC2000DbsDb4a4",  "4_7.png"):  "VC2000DbsDb4a4_4_8",
    ("VC2000DbsDb4a25", "25_8.png"): "VC2000DbsDb4a25_25_6",
    ("VC2000DbsDb4a60", "60_1.png"): "VC2000DbsDb4a60_60_3",
    ("VC2000DbsDb4a90", "90_2.png"): "VC2000DbsDb4a90_90_5",
    ("VC2002DbsDb1a5",  "5_4.png"):  "VC2002DbsDb1a5_5_6",
    ("VC2002DbsDb1a30", "30_6.png"): "VC2002DbsDb1a30_30_2",
}


def find_probes(probe_dir: str):
    """
    Discover all probe images under probe_dir.

    Expected structure:
        probe_dir/
            <FolderName>/
                latent/
                    <probe>.png

    Returns list of (folder_name, probe_filename, probe_full_path).
    """
    probes = []
    probe_dir = Path(probe_dir)

    if not probe_dir.exists():
        print(f"ERROR: Probe directory not found: {probe_dir}")
        sys.exit(1)

    for folder in sorted(probe_dir.iterdir()):
        if not folder.is_dir():
            continue
        latent_dir = folder / "latent"
        if not latent_dir.exists():
            continue
        for img in sorted(latent_dir.iterdir()):
            if img.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                probes.append((folder.name, img.name, str(img)))

    return probes


def get_correct_gallery_id(folder_name: str, probe_file: str) -> str:
    """
    Get the correct gallery match ID for a probe.

    First checks the hardcoded ground truth table.
    Falls back to heuristic: folder prefix + subject number from probe filename.
    """
    key = (folder_name, probe_file)
    if key in GROUND_TRUTH:
        return GROUND_TRUTH[key]

    # Heuristic fallback: extract subject number from probe filename
    # e.g., probe "1_5.png" in folder "VC2000DbsDb2a1" -> look for "VC2000DbsDb2a1_1_"
    match = re.match(r"(\d+)_\d+\.png", probe_file)
    if match:
        subject_num = match.group(1)
        prefix = f"{folder_name}_{subject_num}_"
        return prefix  # Will use startswith matching

    return ""


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate latent fingerprint recognition accuracy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python -m fingerprint_matcher_v2.evaluate \\
      --probe-dir "path/to/FVC-Latent_Test_Dataset/data" \\
      --gallery-dir "path/to/clean" \\
      --workers 4
        """
    )
    parser.add_argument(
        "--probe-dir", required=True,
        help="Directory containing probe subfolders (each with latent/ subfolder)"
    )
    parser.add_argument(
        "--gallery-dir", required=True,
        help="Directory containing gallery (clean) fingerprint images"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="Directory for caching gallery templates (speeds up repeated runs)"
    )

    args = parser.parse_args()

    # Discover probes
    probes = find_probes(args.probe_dir)
    if not probes:
        print(f"ERROR: No probe images found in {args.probe_dir}")
        print("Expected structure: <probe-dir>/<FolderName>/latent/<image>.png")
        sys.exit(1)

    print(f"{'='*70}")
    print(f"  Latent Fingerprint Recognition — Batch Evaluation")
    print(f"{'='*70}")
    print(f"  Probes found:   {len(probes)}")
    print(f"  Gallery dir:    {args.gallery_dir}")
    print(f"  Workers:        {args.workers}")
    print(f"  Cache dir:      {args.cache_dir or '(none)'}")
    print(f"{'='*70}\n")

    # Initialize pipeline
    pipeline = MatchingPipeline(
        n_workers=args.workers,
        nbis_cache_dir=args.cache_dir,
    )
    print(f"Minutiae backend: {pipeline._nbis_matcher.backend_name}")

    # Pre-load gallery templates
    print("Pre-loading gallery templates...")
    t0 = time.perf_counter()
    n_templates = pipeline.preload_gallery(args.gallery_dir)
    gallery_time = time.perf_counter() - t0
    print(f"  {n_templates} templates loaded in {gallery_time:.1f}s\n")

    # Evaluate each probe
    all_ranks = []
    total_time = 0

    print(f"{'Probe':<12} {'Folder':<20} {'Rank':>6} {'SAFIS':>8} {'G-POC':>8} {'C-POC':>8} {'Time':>6}")
    print("-" * 70)

    for folder_name, probe_file, probe_path in probes:
        correct_id = get_correct_gallery_id(folder_name, probe_file)

        t0 = time.perf_counter()
        results = pipeline.run(probe_path, args.gallery_dir)
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        # Find the rank of the correct match
        rank = None
        for i, r in enumerate(results):
            gid = r["gallery_id"]
            if gid == correct_id:
                rank = i + 1
                break
            # Fallback: prefix matching
            if correct_id and (gid.startswith(correct_id) or correct_id.startswith(gid)):
                rank = i + 1
                break

        if rank is None:
            rank = len(results)
            status = "?"
        elif rank == 1:
            status = "+"
        elif rank <= 10:
            status = "~"
        else:
            status = "-"

        all_ranks.append(rank)

        # Get correct match details
        correct_result = results[rank - 1] if rank <= len(results) else results[-1]

        print(f"{status} {probe_file:<10} {folder_name:<20} {rank:>4} "
              f"{correct_result['minutiae_score']:>8.1f} "
              f"{correct_result['texture_score']:>8.3f} "
              f"{correct_result.get('clahe_poc_score', 0):>8.3f} "
              f"{elapsed:>5.0f}s")

    # Summary
    ranks_arr = np.array(all_ranks)
    N = len(ranks_arr)

    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY ({N} probes)")
    print(f"{'='*70}")
    print()

    for k in [1, 5, 10, 20, 50, 100]:
        count = int(np.sum(ranks_arr <= k))
        pct = 100 * count / N
        bar = "#" * int(pct / 2) + "." * (50 - int(pct / 2))
        print(f"  Rank-{k:<3d}  {count:>3d}/{N}  ({pct:5.1f}%)  [{bar}]")

    print()
    print(f"  Mean rank:    {ranks_arr.mean():.1f}")
    print(f"  Median rank:  {np.median(ranks_arr):.0f}")
    print(f"  Best rank:    {ranks_arr.min()}")
    print(f"  Worst rank:   {ranks_arr.max()}")
    print(f"  Total time:   {total_time:.0f}s ({total_time/N:.0f}s per probe)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
