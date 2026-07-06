"""Stage 6: Visualization.

Produces the visual artifacts a reviewer actually looks at:
- annotated copy of image B with bounding boxes around changed regions
- a heatmap overlay of the diff mask
- an added/removed color-coded overlay (more informative than a generic
  heatmap for drawing-type inputs: distinguishes "only in A" from
  "only in B")
- a side-by-side composite of the two original, unmodified inputs
"""
from __future__ import annotations

import cv2
import numpy as np

from app.config import (
    BBOX_COLOR,
    BBOX_THICKNESS,
    ADDED_COLOR,
    REMOVED_COLOR,
)
from app.regions import Region


def draw_bounding_boxes(image_b: np.ndarray, regions: list[Region]) -> np.ndarray:
    """Return a copy of image B annotated with a box around each region."""
    annotated = image_b.copy()
    for region in regions:
        x, y, w, h = region.bbox
        cv2.rectangle(annotated, (x, y), (x + w, y + h), BBOX_COLOR, BBOX_THICKNESS)
        label = f"{region.area_px}px"
        cv2.putText(
            annotated,
            label,
            (x, max(0, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            BBOX_COLOR,
            1,
            cv2.LINE_AA,
        )
    return annotated


def generate_heatmap(diff_mask: np.ndarray, base_image: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay a JET colormap of the diff mask on top of the base image."""
    blurred = cv2.GaussianBlur(diff_mask, (15, 15), 0)
    heatmap = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(base_image, 1 - alpha, heatmap, alpha, 0)
    return overlay


def generate_added_removed_overlay(
    added_only: np.ndarray, removed_only: np.ndarray, base_image: np.ndarray, alpha: float = 0.6
) -> np.ndarray:
    """Color-code content that is exclusively in A ("removed") vs B ("added").

    added_only / removed_only are 0/255 masks (produced by the diff engine
    with registration-jitter tolerance already applied). This is more
    informative than a plain heatmap for drawing-type inputs because it
    tells the viewer the *direction* of the change, not just that
    something changed.
    """
    color_layer = np.zeros_like(base_image)
    color_layer[removed_only > 0] = REMOVED_COLOR
    color_layer[added_only > 0] = ADDED_COLOR

    mask = (removed_only > 0) | (added_only > 0)
    overlay = base_image.copy()
    overlay[mask] = cv2.addWeighted(base_image, 1 - alpha, color_layer, alpha, 0)[mask]
    return overlay


def generate_side_by_side(image_a: np.ndarray, image_b: np.ndarray, gap_px: int = 12) -> np.ndarray:
    """Compose the two original (unmodified) images side by side."""
    h = max(image_a.shape[0], image_b.shape[0])

    def pad_height(img: np.ndarray) -> np.ndarray:
        if img.shape[0] == h:
            return img
        pad = h - img.shape[0]
        return cv2.copyMakeBorder(img, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    padded_a = pad_height(image_a)
    padded_b = pad_height(image_b)
    gap = np.full((h, gap_px, 3), 200, dtype=np.uint8)
    return np.hstack([padded_a, gap, padded_b])


def generate_revision_overlay(
    image_a: np.ndarray,
    image_b: np.ndarray,
    change_objects: list[dict],
    alpha: float = 0.4
) -> np.ndarray:
    """Creates a multi-colored revision overlay based on change classification.

    Color coding:
    - Green (0, 200, 0): Added elements
    - Red (0, 0, 220): Removed elements
    - Orange (0, 140, 255): Modified physical objects (Doors, Windows)
    - Blue (220, 0, 0): Annotation changes / Room renames
    - Purple (200, 0, 200): Dimension changes
    """
    # Start with a faded/light grayscale version of the aligned image A as base
    gray = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # Blend with white to wash it out slightly so colors stand out
    base = cv2.addWeighted(base, 0.4, np.full_like(base, 255), 0.6, 0)

    overlay = base.copy()
    
    for ch in change_objects:
        x, y, w, h = ch["bbox"]
        c_type = ch["type"]
        delta = str(ch.get("delta", ""))
        
        # Select color based on classification rules
        if "Added" in delta or "added" in c_type.lower():
            color = (0, 200, 0)      # Green
        elif "Removed" in delta or "removed" in c_type.lower():
            color = (0, 0, 220)      # Red
        elif c_type == "Dimension Change":
            color = (200, 0, 200)    # Purple
        elif c_type in ["Annotation Change", "Title Block Change", "Room Change"]:
            color = (220, 0, 0)      # Blue
        else:
            color = (0, 140, 255)    # Orange (Door Change, Window Change, Wall Extension, Geometry Change)
            
        # Draw filled transparent rectangle
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        # Draw solid border
        cv2.rectangle(base, (x, y), (x + w, y + h), color, 2)
        
    return cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0)

