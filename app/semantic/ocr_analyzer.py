import cv2
import numpy as np
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_EASYOCR_READER = None

def get_easyocr_reader():
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        try:
            import easyocr
            # Initialize with English, CPU-only
            _EASYOCR_READER = easyocr.Reader(['en'], gpu=False, verbose=False)
            logger.info("EasyOCR Reader initialized successfully on CPU.")
        except Exception as e:
            logger.warning(f"Could not initialize EasyOCR: {e}")
    return _EASYOCR_READER

def run_easyocr(image: np.ndarray) -> list[dict[str, Any]]:
    reader = get_easyocr_reader()
    if reader is None:
        return []
    try:
        raw_results = reader.readtext(image)
        results = []
        for box, text, conf in raw_results:
            text = text.strip()
            if not text:
                continue
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x, y = int(min(xs)), int(min(ys))
            w, h = int(max(xs) - x), int(max(ys) - y)
            results.append({
                "text": text,
                "bbox": (x, y, w, h),
                "confidence": float(conf)
            })
        return results
    except Exception as e:
        logger.warning(f"EasyOCR execution failed: {e}")
        return []

def run_tesseract_ocr(image: np.ndarray) -> list[dict[str, Any]]:
    try:
        import pytesseract
        from pytesseract import Output
        data = pytesseract.image_to_data(image, output_type=Output.DICT)
        results = []
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            text = data['text'][i].strip()
            if not text:
                continue
            conf_val = data['conf'][i]
            if conf_val == -1 or conf_val is None:
                conf = 0.80
            else:
                conf = float(conf_val) / 100.0
            if conf < 0.2:
                continue
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            results.append({
                "text": text,
                "bbox": (x, y, w, h),
                "confidence": conf
            })
        return results
    except Exception as e:
        logger.warning(f"PyTesseract OCR failed: {e}")
        return []

def run_ocr(image: np.ndarray) -> list[dict[str, Any]]:
    """OCR is skipped / disabled to reduce laptop CPU overhead."""
    return []


def compare_annotations(
    ocr_a: list[dict[str, Any]],
    ocr_b: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Compares annotation collections spatially, identifying modifications."""
    changes = []
    matched_b = set()

    for idx_a, text_a in enumerate(ocr_a):
        ax, ay, aw, ah = text_a["bbox"]
        acx, acy = ax + aw / 2.0, ay + ah / 2.0
        val_a = text_a["text"]

        # Search for a spatially proximate text box in B
        best_match_idx = -1
        best_dist = 60.0  # max spatial matching radius in pixels
        for idx_b, text_b in enumerate(ocr_b):
            if idx_b in matched_b:
                continue
            bx, by, bw, bh = text_b["bbox"]
            bcx, bcy = bx + bw / 2.0, by + bh / 2.0
            dist = np.hypot(bcx - acx, bcy - acy)
            if dist < best_dist:
                best_dist = dist
                best_match_idx = idx_b

        if best_match_idx != -1:
            matched_b.add(best_match_idx)
            val_b = ocr_b[best_match_idx]["text"]
            if val_a != val_b:
                # Text changed
                is_dim = any(c in val_a.lower() or c in val_b.lower() for c in ["mm", "dia", "ø", "mtr", "ft", "'", "\""]) or val_a.isdigit() or val_b.isdigit()
                change_type = "dimension_change" if is_dim else "annotation_change"
                
                # Try to compute difference delta for dimensions
                delta = ""
                try:
                    # Clean strings to parse numbers
                    num_a = float(''.join(c for c in val_a if c.isdigit() or c == '.'))
                    num_b = float(''.join(c for c in val_b if c.isdigit() or c == '.'))
                    diff = num_b - num_a
                    delta = f"{diff:+.1f}".rstrip('0').rstrip('.')
                except Exception:
                    delta = f"{val_a} -> {val_b}"

                changes.append({
                    "type": change_type,
                    "old_value": val_a,
                    "new_value": val_b,
                    "delta": delta,
                    "bbox": ocr_b[best_match_idx]["bbox"],
                    "confidence": min(text_a["confidence"], ocr_b[best_match_idx]["confidence"])
                })
        else:
            # Deleted annotation
            is_dim = any(c in val_a.lower() for c in ["mm", "dia", "ø", "mtr", "ft", "'", "\""]) or val_a.isdigit()
            changes.append({
                "type": "dimension_removed" if is_dim else "annotation_removed",
                "old_value": val_a,
                "new_value": "",
                "delta": f"Removed {val_a}",
                "bbox": text_a["bbox"],
                "confidence": text_a["confidence"]
            })

    # Find added annotations (unmatched in B)
    for idx_b, text_b in enumerate(ocr_b):
        if idx_b not in matched_b:
            val_b = text_b["text"]
            is_dim = any(c in val_b.lower() for c in ["mm", "dia", "ø", "mtr", "ft", "'", "\""]) or val_b.isdigit()
            changes.append({
                "type": "dimension_added" if is_dim else "annotation_added",
                "old_value": "",
                "new_value": val_b,
                "delta": f"Added {val_b}",
                "bbox": text_b["bbox"],
                "confidence": text_b["confidence"]
            })

    return changes
