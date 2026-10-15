# Latent Fingerprint Recognition System

**Assignment 4 — Biometrics**

A Python application that matches degraded latent fingerprint images against a gallery of clean rolled/plain prints. The system uses a 4-channel matching pipeline with SAFIS-primary score fusion and provides a PyQt5 GUI for interactive use.

---

## Architecture

### Enhancement Pipeline

Every fingerprint image undergoes preprocessing before matching:

| Stage | Method | Purpose |
|-------|--------|---------|
| Normalization | Zero-mean, unit-variance rescaling | Standardize intensity range |
| CLAHE | Contrast-Limited Adaptive Histogram Equalization (clip=3.0, grid=8x8) | Enhance local contrast |
| Gabor Bank | 8 orientations x 3 frequencies (0.08, 0.10, 0.12 cycles/px) | Ridge structure enhancement |
| ROI Segmentation | Gradient-magnitude + Otsu threshold | Isolate fingerprint region |

Three enhancement levels are used:
- **Light (CLAHE only)** — preserves local texture for SIFT keypoint detection and CLAHE-POC
- **Full (CLAHE + Gabor)** — enhances ridge patterns for Gabor-POC texture correlation

### Matching Channels

#### Channel 1: SIFT + RANSAC (Micro-tiebreaker)

Detects scale- and rotation-invariant keypoints using SIFT (2000 features) on CLAHE-enhanced images. Matches are filtered using:

1. **Lowe's ratio test** (threshold = 0.82) — removes ambiguous matches
2. **RANSAC** (reprojection threshold = 10.0 px) — enforces geometric consistency
3. **ORB fallback** (3000 features, ratio = 0.85) — when SIFT yields too few inliers

Score is computed from three components: inlier ratio (35%), descriptor quality (35%), inlier count bonus (30%). Used only as a micro-tiebreaker (weight 0.0001) since degraded latent prints yield 0 SIFT inliers on most correct-match pairs.

#### Channel 2: Gabor-POC (Texture — tiebreaker)

Phase-only correlation (POC) on Gabor-enhanced images:

1. **Hanning window** — reduces spectral leakage
2. **Band-pass filtering** — Gaussian band-pass (0.05–0.45 normalized frequency) isolates ridge frequencies
3. **Phase correlation** — cross-power spectrum normalized by magnitude
4. **PCE scoring** — Peak-to-Correlation-Energy ratio measures peak sharpness
5. **Sigmoid mapping** — PCE mapped through sigmoid (threshold=6.0, scale=2.0) to [0, 1]

#### Channel 3: CLAHE-POC (Complementary texture — tiebreaker)

Same POC algorithm applied to CLAHE-only enhanced images (no Gabor). Provides complementary texture signal — some probes respond better to CLAHE-POC than Gabor-POC.

#### Channel 4: LEADER + SAFIS (Primary discriminator)

Uses the `afis` library's LEADER extractor (pure Python) for minutiae extraction and SAFIS for matching:

1. **LEADER extraction** — pure Python, works on all platforms (no native C backend required)
2. **SAFIS matching** — rotation/translation invariant minutiae matching
3. **Raw score** — uses `result.raw_score` (= 6.5 × n_matched_pairs) for maximum discrimination; the normalized `.score` attribute is heavily quantized (only 6 unique values)
4. **Template caching** — gallery templates are extracted once and cached to disk for instant re-use

### Score Fusion: SAFIS-Primary with POC Tiebreaker

```
fused_score = safis_raw_score + best_poc × 5.0
```

Where `best_poc = max(Gabor_POC, CLAHE_POC)`.

**Key insight**: SAFIS `raw_score` is quantized at intervals of 6.5 (= 6.5 × n_matched_pairs). Many gallery images share the same SAFIS tier. The POC scale factor (5.0) is deliberately set below 6.5 so that POC can **only reorder images within the same SAFIS tier** — it can never promote an image across tiers. This preserves SAFIS's strong rank-1 signal while letting POC break ties.

**Why SAFIS-primary?** SAFIS alone achieves 46.7% rank-1 accuracy. Earlier fusion approaches (RRF, min-rank, z-score sum) that treated POC channels equally with SAFIS *destroyed* this signal — noisy POC rankings pulled correct matches off rank 1. The SAFIS-primary approach preserves all rank-1 hits and uses POC only where SAFIS cannot discriminate (tied tiers).

---

## Experimental Results

Tested on the FVC-Latent Test Dataset (15 latent probes across 3 FVC databases, 317 gallery images):

### v5 Results (4-Channel SAFIS-Primary Fusion)

| Metric | Value |
|--------|-------|
| **Rank-1 Rate** | **53.3% (8/15)** |
| **Rank-5 Rate** | **53.3% (8/15)** |
| **Rank-10 Rate** | **60.0% (9/15)** |
| Rank-20 Rate | 60.0% (9/15) |
| Mean Rank | 29.5 |

### Per-Probe Results (All 15 Probes)

| Probe | Database | Rank | SAFIS raw | Gabor-POC | CLAHE-POC |
|-------|----------|------|-----------|-----------|-----------|
| 1_5   | Db2      | **1** | 65.0 | 0.963 | 0.781 |
| 4_3   | Db2      | **1** | 45.5 | 0.846 | 0.640 |
| 6_4   | Db2      | **1** | 162.5 | 0.983 | 0.745 |
| 8_8   | Db2      | **1** | 149.5 | 0.967 | 0.733 |
| 4_7   | Db4      | **1** | 45.5 | 0.975 | 0.779 |
| 60_1  | Db4      | **1** | 39.0 | 0.959 | 0.864 |
| 5_4   | Db1      | **1** | 149.5 | 0.754 | 0.822 |
| 30_6  | Db1      | **1** | 123.5 | 0.981 | 0.719 |
| 25_8  | Db4      | 6    | 39.0 | 0.999 | 0.891 |
| 90_2  | Db4      | 21   | 32.5 | 0.856 | 0.867 |
| 15_8  | Db2      | 24   | 26.0 | 0.913 | 0.692 |
| 30_8  | Db2      | 30   | 32.5 | 0.953 | 0.705 |
| 75_2  | Db2      | 69   | 19.5 | 0.778 | 0.793 |
| 10_6  | Db2      | 117  | 19.5 | 0.989 | 0.541 |
| 50_4  | Db2      | 168  | 26.0 | 0.882 | 0.786 |

### Version Comparison

| Version | Channels | Fusion | Rank-1 | Rank-10 | Mean Rank |
|---------|----------|--------|--------|---------|-----------|
| v2.1 | SIFT + POC | Weighted 10/90 | 0/13 (0%) | 1/13 (8%) | — |
| v3.0 | SIFT + POC + NBIS | Two-tier rank bonus | 2/13 (15%) | 7/13 (54%) | 31.2 |
| **v5.0** | **SIFT + Gabor-POC + CLAHE-POC + SAFIS** | **SAFIS-primary** | **8/15 (53%)** | **9/15 (60%)** | **29.5** |

---

## Project Structure

```
fingerprint_matcher_v2/
    __init__.py              # Package metadata (v5.0.0)
    __main__.py              # Entry point — launches PyQt5 GUI
    evaluate.py              # Batch evaluation script (CMC metrics)
    enhance.py               # Image enhancement (CLAHE, Gabor, ROI segmentation)
    pipeline.py              # 4-channel parallel matching orchestration
    requirements.txt         # Python dependencies
    matching/
        __init__.py
        sift_matcher.py      # Channel 1: SIFT + RANSAC + ORB fallback
        texture_matcher.py   # Channels 2-3: Gabor-POC + CLAHE-POC
        nbis_matcher.py      # Channel 4: LEADER/NBIS extraction + SAFIS matching
        fusion.py            # SAFIS-primary score fusion
    gui/
        __init__.py
        main_window.py       # PyQt5 GUI with progress bar and result table
```

---

## Installation

### Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

### Setup

```bash
pip install -r requirements.txt
```

> **Note**: `opencv-contrib-python` is required for SIFT (patented algorithm). The `afis` library provides LEADER+SAFIS fingerprint matching capabilities (pure Python — works on all platforms including Windows).

---

## Usage

### Batch Evaluation (Get Overall Metrics)

```bash
python -m fingerprint_matcher_v2.evaluate \
    --probe-dir "FingerDatasets/Generated_Latent_Fingerprint_Dataset-main/FVC-Latent_Test_Dataset/data" \
    --gallery-dir "FingerDatasets/clean" \
    --workers 4
```

This runs all probes against the gallery and prints a per-probe results table plus CMC metrics (Rank-1, Rank-5, Rank-10, etc.), mean rank, and timing information.

Optional: `--cache-dir <path>` to cache gallery minutiae templates for faster repeated runs.

### GUI Mode

```bash
python -m fingerprint_matcher_v2
```

1. Click **"Single Image..."** to select one latent probe, or **"Folder..."** to batch-process a folder of latent probes.
2. Click **"Browse..."** to select the gallery folder containing clean fingerprint images.
3. Set the number of parallel workers (default: 4).
4. Click **"Run Matching"**.
5. Results appear in a ranked table with fused, SIFT, minutiae, and texture scores.
6. Click any row to preview the probe and gallery images side-by-side.

### Dataset Format

- **Probe images**: Latent fingerprint images (`.png`, `.jpg`, `.bmp`, `.tif`)
- **Gallery images**: Clean rolled/plain prints (`.png`, `.jpg`, `.bmp`, `.tif`)
- Images should be grayscale or will be converted automatically

---

## Known Limitations

- Extremely degraded latent prints with very few extractable minutiae (< 5 matched pairs) remain challenging — these probes have SAFIS raw scores ≤ 26, making them indistinguishable from random matches.
- Processing time is approximately 85–100 seconds per probe against 317 gallery images (4 workers), dominated by POC computation.
- The system uses classical computer vision techniques; deep learning approaches would likely improve performance but are beyond the scope of this assignment.

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >= 1.24 | Array operations |
| opencv-python | >= 4.8 | Image processing |
| opencv-contrib-python | >= 4.8 | SIFT, ORB, thinning |
| scipy | >= 1.11 | Signal processing |
| scikit-image | >= 0.21 | SSIM computation |
| PyQt5 | >= 5.15 | GUI framework |
| afis | >= 0.1 | LEADER extraction + SAFIS matching |

---

## References

1. D. Maltoni, D. Maio, A.K. Jain, S. Prabhakar, *Handbook of Fingerprint Recognition*, Springer, 2009.
2. D. Lowe, "Distinctive Image Features from Scale-Invariant Keypoints," *IJCV*, 2004.
3. K. Ito, H. Nakajima, K. Kobayashi, T. Aoki, T. Higuchi, "A Fingerprint Matching Algorithm Using Phase-Only Correlation," *IEICE Trans.*, 2004.
4. NIST Biometric Image Software (NBIS) — MINDTCT/BOZORTH3 minutiae tools.
5. FVC2000/2002/2004 Fingerprint Verification Competition datasets.
