"""Stage 5: Region extraction.

Turns a cleaned diff mask into a list of discrete changed regions with
bounding box, area, centroid, and a human-readable location descriptor
(e.g. "upper-left"), filtering out anything below a configurable minimum
area so rendering/compression noise doesn't show up as a "change".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from app.config import MIN_REGION_AREA_PX, MIN_REGION_AREA_RATIO


@dataclass
class Region:
    bbox: tuple[int, int, int, int]  # x, y, w, h
    area_px: int
    centroid: tuple[float, float]
    location: str


def _location_descriptor(cx: float, cy: float, width: int, height: int) -> str:
    """Describe a centroid's rough position within the image frame."""
    col = "left" if cx < width / 3 else ("right" if cx > 2 * width / 3 else "center")
    row = "upper" if cy < height / 3 else ("lower" if cy > 2 * height / 3 else "middle")

    if row == "middle" and col == "center":
        return "center"
    if row == "middle":
        return f"middle-{col}"
    if col == "center":
        return f"{row}-center"
    return f"{row}-{col}"


def extract_regions(
    diff_mask: np.ndarray,
    min_area_px: int = MIN_REGION_AREA_PX,
    min_area_ratio: float = MIN_REGION_AREA_RATIO,
) -> list[Region]:
    """Find distinct changed regions in a cleaned binary diff mask.

    A region must clear both the absolute pixel-area floor and the
    ratio-of-frame floor, so the same config works reasonably across
    very different working resolutions.
    """
    height, width = diff_mask.shape[:2]
    frame_area = height * width
    effective_min_area = max(min_area_px, frame_area * min_area_ratio)

    contours, _ = cv2.findContours(
        diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    regions: list[Region] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < effective_min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        else:
            cx, cy = x + w / 2.0, y + h / 2.0

        regions.append(
            Region(
                bbox=(x, y, w, h),
                area_px=int(area),
                centroid=(round(cx, 1), round(cy, 1)),
                location=_location_descriptor(cx, cy, width, height),
            )
        )

    regions.sort(key=lambda r: r.area_px, reverse=True)
    return regions
