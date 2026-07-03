"""Stage 7: Statistics.

Packages region/diff results into a structured, JSON-serializable object.
This -- not raw pixels -- is what gets handed to the LLM summarizer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from app.diff_engine import DiffResult
from app.regions import Region
from app.registration import RegistrationResult


@dataclass
class RegionStat:
    index: int
    bbox: dict[str, int]
    area_px: int
    centroid: dict[str, float]
    location: str


@dataclass
class ComparisonStats:
    registration_success: bool
    registration_method: str
    registration_message: str
    alignment_score: float
    diff_mode: str
    image_width: int
    image_height: int
    total_region_count: int
    percent_area_changed: float
    regions: list[RegionStat]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_stats(
    registration: RegistrationResult,
    diff_result: DiffResult,
    regions: list[Region],
    image_shape: tuple[int, int],
) -> ComparisonStats:
    """Assemble the full structured stats object for a comparison run."""
    height, width = image_shape[:2]
    total_area = height * width
    changed_area = sum(r.area_px for r in regions)
    percent_changed = round(100.0 * changed_area / total_area, 3) if total_area else 0.0

    region_stats = [
        RegionStat(
            index=i,
            bbox={"x": r.bbox[0], "y": r.bbox[1], "w": r.bbox[2], "h": r.bbox[3]},
            area_px=r.area_px,
            centroid={"x": r.centroid[0], "y": r.centroid[1]},
            location=r.location,
        )
        for i, r in enumerate(regions)
    ]

    return ComparisonStats(
        registration_success=registration.success,
        registration_method=registration.method,
        registration_message=registration.message,
        alignment_score=round(registration.alignment_score, 3),
        diff_mode=diff_result.mode,
        image_width=width,
        image_height=height,
        total_region_count=len(regions),
        percent_area_changed=percent_changed,
        regions=region_stats,
    )
