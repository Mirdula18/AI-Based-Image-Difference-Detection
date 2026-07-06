"""Stage 1: ROI region masking helper.

Masks out sheet margins, borders, title blocks, revision panels, numbering,
and legends so they never participate in comparison differencing.
"""
from __future__ import annotations

import cv2
import numpy as np
from app.registration import get_drawing_roi_mask

def apply_comparison_mask(image_mask: np.ndarray, content_mask: np.ndarray) -> np.ndarray:
    """Applies the drawing content ROI mask to mask out non-engineering areas."""
    return cv2.bitwise_and(image_mask, content_mask)
