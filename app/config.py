"""Central configuration/constants for the diff pipeline.

Kept in one place so CLI, API, and tests all agree on thresholds without
having to thread parameters through every function call.
"""
import os

# --- Ingestion ---
PDF_RASTER_DPI = 300
SUPPORTED_RASTER_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SUPPORTED_PDF_EXTS = {".pdf"}

# --- Normalization ---
# Both images are resized so their longest side does not exceed this,
# using the smaller of the two images' natural scale as the common target.
MAX_WORKING_DIMENSION = 2000

# --- Registration ---
# Minimum fraction of the frame area a contour must cover to be considered
# a candidate drawing border / title-block frame.
BORDER_CONTOUR_MIN_AREA_RATIO = 0.35
# Approx-poly epsilon as a fraction of contour perimeter.
BORDER_APPROX_EPSILON_RATIO = 0.02

# ORB / feature-matching fallback
ORB_MAX_FEATURES = 5000
LOWE_RATIO_TEST = 0.75
MIN_GOOD_MATCHES = 12
RANSAC_REPROJ_THRESHOLD = 5.0

# Alignment sanity-check: normalized cross-correlation of edge maps after
# warping must reach at least this value or registration is reported failed.
MIN_ALIGNMENT_SCORE = 0.35

# --- Difference detection ---
ADAPTIVE_THRESH_BLOCK_SIZE = 25
ADAPTIVE_THRESH_C = 10
# A pixel only counts as changed if the other image has no foreground within
# this radius -- absorbs 1-2px residual registration jitter that would
# otherwise produce ghost outlines along every line of the drawing.
LINE_TOLERANCE_PX = 3
MORPH_KERNEL_SIZE = 5
MORPH_DILATE_ITERATIONS = 2
MORPH_ERODE_ITERATIONS = 1
SSIM_DIFF_THRESHOLD = 0.15  # 1 - SSIM local score above this counts as changed

# --- Region extraction ---
MIN_REGION_AREA_PX = 60  # discard smaller blobs as noise
# Same threshold expressed as a fraction of total image area is also
# supported for images of very different resolutions.
MIN_REGION_AREA_RATIO = 0.00005

# --- Visualization ---
BBOX_COLOR = (0, 0, 255)  # red, BGR
BBOX_THICKNESS = 2
HEATMAP_COLORMAP = "COLORMAP_JET"
ADDED_COLOR = (0, 0, 255)    # red = content present only in B (new/added)
REMOVED_COLOR = (255, 0, 0)  # blue = content present only in A (removed)

# --- LLM summary ---
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
SUMMARY_MAX_TOKENS = 400
