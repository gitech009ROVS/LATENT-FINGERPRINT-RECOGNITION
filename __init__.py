"""
Latent Fingerprint Recognition System v5.

Matches degraded latent fingerprint images against a gallery of clean
rolled/plain prints using a 4-channel approach with SAFIS-primary score
fusion:

  Channel 1 -- SIFT keypoints + RANSAC geometric verification (micro-tiebreaker)
      Detects scale/rotation-invariant keypoints on CLAHE-enhanced images.
      Lowe's ratio test (0.82) filters ambiguous matches; RANSAC enforces
      geometric consistency.  Falls back to ORB when SIFT yields too few
      inliers.

  Channel 2 -- Gabor-POC: Phase-Only Correlation on Gabor-enhanced images
      Band-pass filtered phase correlation in the frequency domain measures
      ridge-pattern similarity.  PCE ratio mapped through a sigmoid to [0,1].

  Channel 3 -- CLAHE-POC: Phase-Only Correlation on CLAHE-only images
      Complementary texture signal — some probes respond better to CLAHE-POC
      than Gabor-POC.

  Channel 4 -- LEADER + SAFIS minutiae matching (primary discriminator)
      Pure Python minutiae extraction (LEADER) and matching (SAFIS).
      Uses raw_score (6.5 * n_matched_pairs) for maximum discrimination.
      Gallery templates are cached to disk for instant re-use.

Enhancement pipeline:
  - enhance_light() -- Normalize + CLAHE only (for SIFT and CLAHE-POC)
  - enhance()       -- Normalize + CLAHE + Gabor bank (8 orientations x
                       3 frequencies) for Gabor-POC texture matching
  - segment_roi()   -- Gradient-magnitude ROI segmentation with Otsu

Fusion: SAFIS-primary with POC tiebreaker.
  fused = safis_raw + max(gabor_poc, clahe_poc) * 5.0
  POC scale (5.0) is below the SAFIS tier gap (6.5), guaranteeing POC
  can only reorder within tiers, never across them.

Results (15 probes, 317 gallery):
  Rank-1: 53.3% (8/15), Rank-10: 60.0% (9/15), Mean rank: 29.5
"""

__version__ = "5.0.0"
