"""Stage 4: Difference detection.

Two complementary detection modes are supported:

- "linework" (default, best for CAD/architectural drawings): binarize
  both aligned images to near-black-and-white, XOR them, then clean the
  result with morphological closing so nearby differing pixels merge
  into coherent blobs instead of scattered single-pixel noise.
- "photo" (SSIM-based): structural similarity diff map, useful as a
  sanity check and as the primary signal for general photographic
  inputs where a binary line-art threshold doesn't make sense.

Both modes operate on already-registered (same coordinate space) images.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from app.config import (
    ADAPTIVE_THRESH_BLOCK_SIZE,
    ADAPTIVE_THRESH_C,
    LINE_TOLERANCE_PX,
    MORPH_KERNEL_SIZE,
    MORPH_DILATE_ITERATIONS,
    MORPH_ERODE_ITERATIONS,
    SSIM_DIFF_THRESHOLD,
)

DiffMode = Literal["linework", "photo"]


@dataclass
class DiffResult:
    diff_mask: np.ndarray       # uint8, 0/255, cleaned changed-region mask
    binary_a: np.ndarray        # uint8, 0/255 binarized image A (linework mode only)
    binary_b: np.ndarray        # uint8, 0/255 binarized image B (linework mode only)
    mode: DiffMode
    raw_score: float            # mean SSIM (photo mode) or fraction changed (linework)
    added_mask: np.ndarray | None = None    # foreground only in B (tolerance applied)
    removed_mask: np.ndarray | None = None  # foreground only in A (tolerance applied)


def _binarize(gray: np.ndarray) -> np.ndarray:
    """Adaptive-threshold a grayscale image to near-binary black/white.

    Adaptive thresholding (rather than a single global cutoff) is used
    because anti-aliasing and rendering differences between the two
    source files mean raw pixel intensities are not directly comparable.
    """
    block_size = ADAPTIVE_THRESH_BLOCK_SIZE
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        ADAPTIVE_THRESH_C,
    )


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """Merge nearby differing pixels into coherent blobs via morphology."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
    )
    dilated = cv2.dilate(mask, kernel, iterations=MORPH_DILATE_ITERATIONS)
    cleaned = cv2.erode(dilated, kernel, iterations=MORPH_ERODE_ITERATIONS)
    return cleaned


def compute_diff_linework(image_a: np.ndarray, image_b: np.ndarray) -> DiffResult:
    """Tolerant binary diff tuned for line-art / CAD-style drawings.

    Rather than a raw XOR (which flags a ghost outline along every line
    whenever registration is off by even a pixel), a foreground pixel in
    one image only counts as changed if the *other* image has no
    foreground within LINE_TOLERANCE_PX of it.
    """
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)

    binary_a = _binarize(gray_a)
    binary_b = _binarize(gray_b)

    tol = 2 * LINE_TOLERANCE_PX + 1
    tol_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tol, tol))
    a_neighborhood = cv2.dilate(binary_a, tol_kernel)
    b_neighborhood = cv2.dilate(binary_b, tol_kernel)

    added_only = cv2.bitwise_and(binary_b, cv2.bitwise_not(a_neighborhood))
    removed_only = cv2.bitwise_and(binary_a, cv2.bitwise_not(b_neighborhood))
    xor_mask = cv2.bitwise_or(added_only, removed_only)
    cleaned_mask = _clean_mask(xor_mask)

    fraction_changed = float(np.count_nonzero(cleaned_mask)) / cleaned_mask.size

    return DiffResult(
        diff_mask=cleaned_mask,
        binary_a=binary_a,
        binary_b=binary_b,
        mode="linework",
        raw_score=fraction_changed,
        added_mask=added_only,
        removed_mask=removed_only,
    )


def compute_diff_photo(image_a: np.ndarray, image_b: np.ndarray) -> DiffResult:
    """SSIM-based diff for general photographic (non-line-art) inputs."""
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)

    mean_score, diff = ssim(gray_a, gray_b, full=True)
    dissimilarity = 1.0 - diff  # high value = more different

    changed = (dissimilarity > SSIM_DIFF_THRESHOLD).astype(np.uint8) * 255
    cleaned_mask = _clean_mask(changed)

    return DiffResult(
        diff_mask=cleaned_mask,
        binary_a=gray_a,
        binary_b=gray_b,
        mode="photo",
        raw_score=float(mean_score),
    )


def compute_diff(
    image_a: np.ndarray, image_b: np.ndarray, mode: DiffMode = "linework"
) -> DiffResult:
    """Dispatch to the requested diff mode."""
    if mode == "photo":
        return compute_diff_photo(image_a, image_b)
    return compute_diff_linework(image_a, image_b)
