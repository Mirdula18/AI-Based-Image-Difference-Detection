"""Orchestrates all pipeline stages in order.

This is the single entry point used by both the CLI (cli.py) and the
FastAPI route (routers/compare.py) so the two never drift out of sync.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

from app.ingest import load_pair
from app.preprocess import normalize_pair
from app.registration import register_images
from app.diff_engine import compute_diff, DiffMode
from app.regions import extract_regions
from app.visualize import (
    draw_bounding_boxes,
    generate_heatmap,
    generate_added_removed_overlay,
    generate_side_by_side,
)
from app.stats import build_stats
from app.summarizer import summarize


@dataclass
class PipelineOutput:
    stats: dict[str, Any]
    summary: str
    image_paths: dict[str, str]


def run_pipeline(
    path_a: str,
    path_b: str,
    output_dir: str,
    diff_mode: DiffMode = "linework",
    generate_summary: bool = True,
) -> PipelineOutput:
    """Run ingestion through summary and write all visual artifacts to disk.

    Returns a PipelineOutput with the structured stats dict, the summary
    paragraph, and a mapping of artifact name -> file path.
    """
    os.makedirs(output_dir, exist_ok=True)

    ingested_a, ingested_b = load_pair(path_a, path_b)
    normalized = normalize_pair(ingested_a.image, ingested_b.image)

    registration = register_images(normalized.image_a, normalized.image_b)

    diff_result = compute_diff(normalized.image_a, registration.aligned_b, mode=diff_mode)
    regions = extract_regions(diff_result.diff_mask)

    stats = build_stats(
        registration=registration,
        diff_result=diff_result,
        regions=regions,
        image_shape=normalized.image_a.shape,
    )

    annotated = draw_bounding_boxes(registration.aligned_b, regions)
    heatmap = generate_heatmap(diff_result.diff_mask, normalized.image_a)
    side_by_side = generate_side_by_side(ingested_a.image, ingested_b.image)

    if diff_result.mode == "linework" and diff_result.added_mask is not None:
        added_removed = generate_added_removed_overlay(
            diff_result.added_mask, diff_result.removed_mask, normalized.image_a
        )
    else:
        added_removed = heatmap

    image_paths = {
        "original_a": os.path.join(output_dir, "original_a.png"),
        "original_b": os.path.join(output_dir, "original_b.png"),
        "aligned_b": os.path.join(output_dir, "aligned_b.png"),
        "annotated_regions": os.path.join(output_dir, "annotated_regions.png"),
        "heatmap": os.path.join(output_dir, "heatmap.png"),
        "added_removed_overlay": os.path.join(output_dir, "added_removed_overlay.png"),
        "side_by_side": os.path.join(output_dir, "side_by_side.png"),
    }

    cv2.imwrite(image_paths["original_a"], ingested_a.image)
    cv2.imwrite(image_paths["original_b"], ingested_b.image)
    cv2.imwrite(image_paths["aligned_b"], registration.aligned_b)
    cv2.imwrite(image_paths["annotated_regions"], annotated)
    cv2.imwrite(image_paths["heatmap"], heatmap)
    cv2.imwrite(image_paths["added_removed_overlay"], added_removed)
    cv2.imwrite(image_paths["side_by_side"], side_by_side)

    stats_dict = stats.to_dict()
    summary = summarize(stats) if generate_summary else ""

    return PipelineOutput(stats=stats_dict, summary=summary, image_paths=image_paths)
