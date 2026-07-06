"""Orchestrates all pipeline stages in order.

This is the single entry point used by both the CLI (cli.py) and the
FastAPI route (routers/compare.py) so the two drift out of sync.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

from app.ingest import load_pair_pages
from app.preprocess import normalize_pair
from app.registration import register_images, RegistrationError, get_drawing_roi_mask
from app.diff_engine import compute_diff, DiffMode
from app.regions import extract_regions, Region, _location_descriptor
from app.visualize import (
    draw_bounding_boxes,
    generate_heatmap,
    generate_added_removed_overlay,
    generate_side_by_side,
    generate_revision_overlay,
)
from app.stats import build_stats
from app.summarizer import summarize

# Import new AI/Semantic drawing modules
from app.semantic.ocr_analyzer import run_ocr, compare_annotations
from app.semantic.detector import classify_drawing_category, detect_entities, compare_geometry
from app.semantic.fusion import fuse_and_classify_changes
from app.semantic.masking import apply_comparison_mask
from app.report.pdf_generator import generate_pdf_report


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

    # 1. Ingestion: Load page list for both drawings
    ingested_a_pages, ingested_b_pages = load_pair_pages(path_a, path_b)
    ingested_a = ingested_a_pages[0]
    ingested_b = ingested_b_pages[0]

    # 2. Normalization: Standardize resolution and calculate pixel_ratio mapping
    normalized = normalize_pair(ingested_a.image, ingested_b.image, page_a=ingested_a)

    # 3. Registration: Find H matrix and warp Version B to align with A
    try:
        registration = register_images(normalized.image_a, normalized.image_b)
    except RegistrationError as err:
        # Stage 4: Catch low-quality registration and abort comparison
        stats_dict = {
            "status": "aborted",
            "error": "Comparison aborted. Alignment confidence too low. Please review inputs.",
            "registration_success": False,
            "alignment_score": 0.0,
            "total_region_count": 0,
            "percent_area_changed": 0.0,
            "regions": [],
            "total_changes": 0,
            "change_objects": [],
            "drawing_type": "general"
        }
        return PipelineOutput(
            stats=stats_dict,
            summary="Comparison aborted. Alignment confidence too low. Please review inputs.",
            image_paths={}
        )

    # 4. Content ROI Masking
    content_mask, roi_bbox = get_drawing_roi_mask(normalized.image_a)

    # 5. OCR and Annotation Analysis: Extract text from aligned sheets and compare
    ocr_a = run_ocr(normalized.image_a)
    ocr_b = run_ocr(registration.aligned_b)
    ocr_changes = compare_annotations(ocr_a, ocr_b)

    # 6. Semantic Decomposition: Classify drawing and detect domain objects (doors, windows, etc.)
    ocr_texts = [item["text"] for item in ocr_a] + [item["text"] for item in ocr_b]
    drawing_type = classify_drawing_category(ocr_texts)
    
    cv_entities_a = detect_entities(normalized.image_a, drawing_type, ocr_a)
    cv_entities_b = detect_entities(registration.aligned_b, drawing_type, ocr_b)

    # 7. Geometry Analysis: Compare lines/vectors
    geometry_changes = compare_geometry(normalized.image_a, registration.aligned_b)

    # 8. Fusion Engine: Merge evidence and compute confidence/explanations
    def location_helper(cx: float, cy: float) -> str:
        return _location_descriptor(cx, cy, normalized.target_size[0], normalized.target_size[1])

    change_objects = fuse_and_classify_changes(
        normalized.image_a,
        registration.aligned_b,
        cv_entities_a,
        cv_entities_b,
        ocr_changes,
        geometry_changes,
        normalized.pixel_ratio,
        location_helper,
        alignment_score=registration.alignment_score
    )

    # --- Check report constraints: if change_count > 100, flag as unreliable ---
    if len(change_objects) > 100:
        stats_dict = {
            "status": "unreliable",
            "error": "Comparison quality insufficient. Too many candidate changes detected. Likely registration failure.",
            "registration_success": True,
            "alignment_score": registration.alignment_score,
            "total_region_count": len(change_objects),
            "percent_area_changed": 0.0,
            "regions": [],
            "total_changes": len(change_objects),
            "change_objects": [],
            "drawing_type": drawing_type
        }
        return PipelineOutput(
            stats=stats_dict,
            summary="Comparison quality insufficient. Too many candidate changes detected. Likely registration failure.",
            image_paths={}
        )

    # 9. Diff Engine (Classical fallback comparison / background validation)
    diff_result = compute_diff(normalized.image_a, registration.aligned_b, mode=diff_mode)
    
    # Apply Drawing Area mask to raw diffs before region extraction to suppress border & margins artifacts
    diff_result.diff_mask = apply_comparison_mask(diff_result.diff_mask, content_mask)
    if diff_result.added_mask is not None:
        diff_result.added_mask = apply_comparison_mask(diff_result.added_mask, content_mask)
    if diff_result.removed_mask is not None:
        diff_result.removed_mask = apply_comparison_mask(diff_result.removed_mask, content_mask)
    
    # Map change_objects back to Regions for compatibility with legacy visualization methods
    regions = []
    for idx, ch in enumerate(change_objects):
        x, y, w, h = ch["bbox"]
        regions.append(
            Region(
                bbox=(x, y, w, h),
                area_px=w * h,
                centroid=(x + w / 2.0, y + h / 2.0),
                location=ch["location"]
            )
        )
    # If no semantic changes were found, fall back to pixel diff regions inside content mask
    if not regions:
        regions = extract_regions(diff_result.diff_mask)

    # 10. Statistics: Build final stats dictionary
    stats = build_stats(
        registration=registration,
        diff_result=diff_result,
        regions=regions,
        image_shape=normalized.image_a.shape,
        change_objects=change_objects,
        drawing_type=drawing_type,
    )

    # 11. Visualization overlays
    annotated = draw_bounding_boxes(registration.aligned_b, regions)
    heatmap = generate_heatmap(diff_result.diff_mask, normalized.image_a)
    side_by_side = generate_side_by_side(ingested_a.image, ingested_b.image)
    added_removed = generate_revision_overlay(normalized.image_a, registration.aligned_b, change_objects)

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

    # 12. PDF Report Generator
    pdf_report_path = os.path.join(output_dir, "revision_report.pdf")
    generate_pdf_report(pdf_report_path, stats, image_paths)
    image_paths["pdf_report"] = pdf_report_path

    stats_dict = stats.to_dict()
    summary = summarize(stats) if generate_summary else ""

    return PipelineOutput(stats=stats_dict, summary=summary, image_paths=image_paths)


