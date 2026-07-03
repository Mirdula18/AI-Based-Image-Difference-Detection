"""Stage 2: Normalization.

Resolves gross size mismatches between the two input images by resizing
both to a common working resolution, preserving aspect ratio. This step
intentionally does NOT attempt alignment/registration -- it only ensures
downstream stages operate on comparably-scaled images.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.config import MAX_WORKING_DIMENSION


@dataclass
class NormalizedPair:
    image_a: np.ndarray
    image_b: np.ndarray
    target_size: tuple[int, int]  # (width, height)


def _fit_scale(w: int, h: int, max_dim: int) -> float:
    longest = max(w, h)
    if longest <= max_dim:
        return 1.0
    return max_dim / float(longest)


def normalize_pair(
    image_a: np.ndarray,
    image_b: np.ndarray,
    max_dim: int = MAX_WORKING_DIMENSION,
) -> NormalizedPair:
    """Resize both images to a shared working resolution.

    The common target is derived from whichever image, once capped to
    `max_dim` on its longest side, has the smaller resulting scale --
    i.e. we standardize on the *smaller* of the two natural resolutions
    so we never upscale a low-res input past the other image's detail.
    """
    ha, wa = image_a.shape[:2]
    hb, wb = image_b.shape[:2]

    scale_a = _fit_scale(wa, ha, max_dim)
    scale_b = _fit_scale(wb, hb, max_dim)

    # Resulting (capped) longest-side length for each image.
    longest_a = max(wa, ha) * scale_a
    longest_b = max(wb, hb) * scale_b
    target_longest = min(longest_a, longest_b)

    def resize_to_longest(img: np.ndarray, target_longest: float) -> np.ndarray:
        h, w = img.shape[:2]
        scale = target_longest / max(w, h)
        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        return cv2.resize(img, (new_w, new_h), interpolation=interp)

    resized_a = resize_to_longest(image_a, target_longest)
    resized_b = resize_to_longest(image_b, target_longest)

    # The two results may still differ by a pixel or two due to rounding
    # of differing aspect ratios; pad to a common canvas size so every
    # downstream stage can assume identical array shapes.
    common_h = max(resized_a.shape[0], resized_b.shape[0])
    common_w = max(resized_a.shape[1], resized_b.shape[1])

    def pad_to(img: np.ndarray, h: int, w: int) -> np.ndarray:
        ph = h - img.shape[0]
        pw = w - img.shape[1]
        if ph == 0 and pw == 0:
            return img
        return cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    padded_a = pad_to(resized_a, common_h, common_w)
    padded_b = pad_to(resized_b, common_h, common_w)

    return NormalizedPair(image_a=padded_a, image_b=padded_b, target_size=(common_w, common_h))
