import os
import cv2
import numpy as np
import pytest
from app.semantic.detector import classify_drawing_category, detect_entities, compare_geometry, Entity
from app.semantic.ocr_analyzer import compare_annotations
from app.semantic.fusion import fuse_and_classify_changes, generate_explanation
from app.stats import build_stats
from app.registration import RegistrationResult
from app.diff_engine import DiffResult
from app.regions import Region
from app.report.pdf_generator import generate_pdf_report


def test_classify_drawing_category():
    # Architectural keywords
    texts_arch = ["Floor Plan Level 1", "Main entrance door", "Bedroom closet"]
    assert classify_drawing_category(texts_arch) == "architectural"

    # Mechanical keywords
    texts_mech = ["Drill hole size", "Chamfer tolerance 0.1", "Assembly view"]
    assert classify_drawing_category(texts_mech) == "mechanical"

    # Fallback default
    assert classify_drawing_category([]) == "architectural"


def test_detect_entities_empty():
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    entities = detect_entities(img, "architectural", [])
    assert isinstance(entities, list)


def test_compare_geometry_identical():
    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    cv2.line(img, (20, 20), (180, 20), (0, 0, 0), 2)
    changes = compare_geometry(img, img.copy())
    # Identical images should report no geometrical line modifications
    assert len(changes) == 0


def test_compare_annotations():
    ocr_a = [{"text": "Ø25", "bbox": (10, 10, 20, 20), "confidence": 0.9}]
    ocr_b = [{"text": "Ø30", "bbox": (12, 11, 20, 20), "confidence": 0.95}]
    
    changes = compare_annotations(ocr_a, ocr_b)
    assert len(changes) == 1
    assert changes[0]["type"] == "dimension_change"
    assert changes[0]["old_value"] == "Ø25"
    assert changes[0]["new_value"] == "Ø30"
    assert changes[0]["delta"] == "+5"


def test_fuse_changes_and_explain():
    img_a = np.full((200, 200, 3), 255, dtype=np.uint8)
    img_b = np.full((200, 200, 3), 255, dtype=np.uint8)
    
    cv_a = [Entity("Door1", "door", (50, 50, 40, 40), 0.9)]
    cv_b = [Entity("Door1", "door", (50, 50, 60, 40), 0.95)] # Door widened from 40 to 60 pixels
    
    def mock_loc(cx, cy):
        return "center"
        
    fused = fuse_and_classify_changes(
        img_a, img_b, cv_a, cv_b, [], [], pixel_ratio=0.5, location_fn=mock_loc
    )
    
    assert len(fused) == 1
    assert fused[0]["type"] == "Door Change"
    assert "20 mm" in fused[0]["old_value"]
    assert "30 mm" in fused[0]["new_value"]
    assert "+10 mm" in fused[0]["delta"]
    assert "Door width modified" in fused[0]["explanation"]


def test_generate_pdf_report(tmp_path):
    # Mock data for report generation
    pdf_path = os.path.join(tmp_path, "report.pdf")
    
    # Mock images
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    img_path = os.path.join(tmp_path, "img.png")
    cv2.imwrite(img_path, img)
    
    image_paths = {
        "original_a": img_path,
        "aligned_b": img_path,
        "added_removed_overlay": img_path,
        "heatmap": img_path,
        "annotated_regions": img_path,
    }
    
    # Mock stats
    reg = RegistrationResult(img, True, "border", 0.9, "Success", None, 0.1)
    diff = DiffResult(np.zeros((100, 100), dtype=np.uint8), img, img, "linework", 0.0)
    regions = [Region((10, 10, 20, 20), 400, (20, 20), "center")]
    
    change_objs = [{
        "id": "CH-001",
        "type": "Door Change",
        "old_value": "20 mm",
        "new_value": "30 mm",
        "delta": "+10 mm",
        "bbox": (10, 10, 20, 20),
        "real_dimensions": (10.0, 10.0),
        "confidence": 0.95,
        "location": "center",
        "ssim_score": 0.4,
        "explanation": "Door width modified."
    }]
    
    stats = build_stats(reg, diff, regions, img.shape, change_objs, "architectural")
    
    out_pdf = generate_pdf_report(pdf_path, stats, image_paths)
    assert os.path.exists(out_pdf)
    assert os.path.getsize(out_pdf) > 0
