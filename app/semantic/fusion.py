import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from app.semantic.detector import Entity
from typing import Any
from difflib import SequenceMatcher

class ChangesList(list):
    """A standard list subclass to carry candidate filtering metadata backward-compatibly."""
    rejected_count = 0
    total_candidates = 0

def get_ssim_score(crop_a: np.ndarray, crop_b: np.ndarray) -> float:
    """Calculates local structural similarity between two crops to validate changes."""
    if crop_a.shape != crop_b.shape:
        h = max(crop_a.shape[0], crop_b.shape[0])
        w = max(crop_a.shape[1], crop_b.shape[1])
        if h < 7: h = 7
        if w < 7: w = 7
        crop_a = cv2.resize(crop_a, (w, h))
        crop_b = cv2.resize(crop_b, (w, h))
    else:
        h, w = crop_a.shape[:2]
        if h < 7 or w < 7:
            h_pad = max(7, h)
            w_pad = max(7, w)
            crop_a = cv2.resize(crop_a, (w_pad, h_pad))
            crop_b = cv2.resize(crop_b, (w_pad, h_pad))

    gray_a = cv2.cvtColor(crop_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(crop_b, cv2.COLOR_BGR2GRAY)
    
    score, _ = ssim(gray_a, gray_b, full=True)
    return float(score)

def generate_explanation(
    change_type: str,
    old_val: str,
    new_val: str,
    location: str,
    real_w: float,
    real_h: float
) -> str:
    """Generates natural-language, non-technical explanations for changes."""
    loc_desc = f"in the {location} region" if location else "on the drawing"
    
    if change_type == "Door Change":
        if old_val and new_val and "Removed" not in new_val and "Added" not in new_val:
            return f"Door width modified from {old_val} to {new_val} {loc_desc}."
        return f"Door layout modified. Bounding size: {real_w:.0f} x {real_h:.0f} mm {loc_desc}."
    
    elif change_type == "Window Change":
        if old_val and new_val and "Removed" not in new_val and "Added" not in new_val:
            return f"Window size increased from {old_val} to {new_val} {loc_desc}."
        return f"Window modified or added. Bounding size: {real_w:.0f} x {real_h:.0f} mm {loc_desc}."
        
    elif change_type == "Room Change":
        if old_val and new_val:
            return f"Room '{old_val}' was converted to '{new_val}' {loc_desc}."
        elif new_val:
            return f"New room '{new_val}' was added {loc_desc}."
        return f"Room layout changed {loc_desc}."
        
    elif change_type == "Dimension Change":
        return f"Dimension label updated from '{old_val}' to '{new_val}' {loc_desc}."
        
    elif change_type == "Annotation Change":
        return f"Text annotation changed from '{old_val}' to '{new_val}' {loc_desc}."
        
    elif change_type == "Title Block Change":
        return f"Title block sheet metadata updated from '{old_val}' to '{new_val}'."
        
    elif change_type == "Geometry Change":
        return f"Physical line work or structure modified {loc_desc} (size: {real_w:.0f}x{real_h:.0f} mm)."
        
    elif change_type == "Wall Extension":
        return f"Partition wall extended or added {loc_desc}."

    return f"A modification was detected {loc_desc}."

def compute_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    xa, ya, wa, ha = box_a
    xb, yb, wb, hb = box_b
    
    ix1 = max(xa, xb)
    iy1 = max(ya, yb)
    ix2 = min(xa + wa, xb + wb)
    iy2 = min(ya + ha, yb + hb)
    
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    
    inter_area = iw * ih
    union_area = (wa * ha) + (wb * hb) - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def fuse_and_classify_changes(
    image_a: np.ndarray,
    image_b: np.ndarray,
    cv_entities_a: list[Entity],
    cv_entities_b: list[Entity],
    ocr_changes: list[dict[str, Any]],
    geometry_changes: list[dict[str, Any]],
    pixel_ratio: float,
    location_fn,
    alignment_score: float = 0.98
) -> ChangesList:
    """Combines CV detections, OCR changes, and SSIM validations to produce Change Objects.

    Uses weighted correspondence scoring, strict validation checks, and confidence filtering.
    """
    fused_changes = []
    rejected_count = 0
    
    h_img, w_img = image_b.shape[:2]
    sheet_area = h_img * w_img
    sheet_width_mm = w_img * pixel_ratio
    
    # Max dimensions for normalization of centroid proximity
    max_dim = float(max(h_img, w_img))

    # 1. Process OCR / Annotation Changes
    for idx, ocr_c in enumerate(ocr_changes):
        x, y, w, h = ocr_c["bbox"]
        cx, cy = x + w / 2.0, y + h / 2.0
        location = location_fn(cx, cy)
        
        # Bounding box filters (component_area < 150px, changed_area < 0.01% of sheet)
        area = w * h
        if area < 150 or (area / sheet_area) < 0.0001:
            rejected_count += 1
            continue
            
        crop_a = image_a[y:y+h, x:x+w]
        crop_b = image_b[y:y+h, x:x+w]
        ssim_val = get_ssim_score(crop_a, crop_b) if crop_a.size > 0 and crop_b.size > 0 else 0.5
        
        # Suppress visual changes that are identical (SSIM > 0.98)
        if ssim_val > 0.98:
            rejected_count += 1
            continue
            
        c_type = ocr_c["type"]
        if "dimension" in c_type:
            c_class = "Dimension Change"
        elif "title" in c_type or any(k in ocr_c["old_value"].upper() or k in ocr_c["new_value"].upper() for k in ["SHEET", "SCALE", "REV"]):
            c_class = "Title Block Change"
        elif "room" in c_type or any(rm in ocr_c["old_value"].upper() or rm in ocr_c["new_value"].upper() for rm in ["ROOM", "BEDROOM", "LIVING", "KITCHEN", "DINING", "CLOSET"]):
            c_class = "Room Change"
        else:
            c_class = "Annotation Change"
            
        # Parse deltas to reject unrealistic changes
        delta_str = ocr_c["delta"]
        try:
            val_a_clean = ''.join(c for c in ocr_c["old_value"] if c.isdigit() or c == '.')
            val_b_clean = ''.join(c for c in ocr_c["new_value"] if c.isdigit() or c == '.')
            if val_a_clean and val_b_clean:
                num_a = float(val_a_clean)
                num_b = float(val_b_clean)
                delta_val = num_b - num_a
                # Validate dimension change: max change < 30% or delta > 5000 mm
                if num_a > 0 and abs(delta_val) / num_a > 0.30:
                    rejected_count += 1
                    continue
                if abs(delta_val) > 5000:
                    rejected_count += 1
                    continue
        except Exception:
            pass

        real_w = w * pixel_ratio
        real_h = h * pixel_ratio
        
        # Calculate unified confidence
        geom_score = 0.9
        ocr_score = ocr_c["confidence"]
        ssim_term = 1.0 - ssim_val
        confidence = 0.35 * alignment_score + 0.25 * geom_score + 0.20 * ocr_score + 0.20 * ssim_term
        
        if confidence > 0.85:
            fused_changes.append({
                "id": f"CH-OCR-{idx+1:03d}",
                "type": c_class,
                "old_value": ocr_c["old_value"],
                "new_value": ocr_c["new_value"],
                "delta": ocr_c["delta"],
                "bbox": (int(x), int(y), int(w), int(h)),
                "real_dimensions": (float(round(real_w, 1)), float(round(real_h, 1))),
                "confidence": float(round(confidence, 3)),
                "location": location,
                "ssim_score": float(round(ssim_val, 3)),
                "explanation": generate_explanation(c_class, ocr_c["old_value"], ocr_c["new_value"], location, real_w, real_h)
            })
        else:
            rejected_count += 1

    # 2. Process CV Doors A vs B
    doors_a = [e for e in cv_entities_a if e.entity_type == "door"]
    doors_b = [e for e in cv_entities_b if e.entity_type == "door"]
    matched_doors_b = set()
    
    door_change_idx = len(fused_changes)
    for da in doors_a:
        ax, ay, aw, ah = da.bbox
        acx, acy = ax + aw / 2.0, ay + ah / 2.0
        
        best_match = None
        best_score = 0.0
        for db in doors_b:
            if db.entity_id in matched_doors_b:
                continue
            bx, by, bw, bh = db.bbox
            bcx, bcy = bx + bw / 2.0, by + bh / 2.0
            
            iou = compute_iou(da.bbox, db.bbox)
            ar_a = aw / ah if ah > 0 else 1.0
            ar_b = bw / bh if bh > 0 else 1.0
            shape_sim = min(ar_a, ar_b) / max(ar_a, ar_b)
            ocr_sim = 1.0
            dist = np.hypot(bcx - acx, bcy - acy)
            centroid_prox = max(0.0, 1.0 - (dist / max_dim))
            
            score = 0.4 * iou + 0.2 * shape_sim + 0.2 * ocr_sim + 0.2 * centroid_prox
            if score > 0.8 and score > best_score:
                best_score = score
                best_match = db
                
        if best_match:
            matched_doors_b.add(best_match.entity_id)
            bx, by, bw, bh = best_match.bbox
            
            old_w_mm = aw * pixel_ratio
            new_w_mm = bw * pixel_ratio
            
            if not (600 <= old_w_mm <= 1800) or not (600 <= new_w_mm <= 1800):
                rejected_count += 1
                continue
                
            delta_mm = new_w_mm - old_w_mm
            if abs(delta_mm) > 5000 or abs(delta_mm) > 0.5 * sheet_width_mm:
                rejected_count += 1
                continue
                
            if abs(delta_mm) > 10:
                door_change_idx += 1
                location = location_fn(bcx, bcy)
                
                crop_a = image_a[by:by+bh, bx:bx+bw]
                crop_b = image_b[by:by+bh, bx:bx+bw]
                ssim_val = get_ssim_score(crop_a, crop_b) if crop_a.size > 0 and crop_b.size > 0 else 0.5
                if ssim_val > 0.98:
                    rejected_count += 1
                    continue
                    
                confidence = 0.35 * alignment_score + 0.25 * best_score + 0.20 * 1.0 + 0.20 * (1.0 - ssim_val)
                if confidence > 0.85:
                    fused_changes.append({
                        "id": f"CH-DR-{door_change_idx:03d}",
                        "type": "Door Change",
                        "old_value": f"{old_w_mm:.0f} mm",
                        "new_value": f"{new_w_mm:.0f} mm",
                        "delta": f"{delta_mm:+.0f} mm",
                        "bbox": (int(bx), int(by), int(bw), int(bh)),
                        "real_dimensions": (float(round(bw * pixel_ratio, 1)), float(round(bh * pixel_ratio, 1))),
                        "confidence": float(round(confidence, 3)),
                        "location": location,
                        "ssim_score": float(round(ssim_val, 3)),
                        "explanation": generate_explanation("Door Change", f"{old_w_mm:.0f} mm", f"{new_w_mm:.0f} mm", location, bw * pixel_ratio, bh * pixel_ratio)
                    })
                else:
                    rejected_count += 1
        else:
            old_w_mm = aw * pixel_ratio
            if 600 <= old_w_mm <= 1800:
                door_change_idx += 1
                location = location_fn(acx, acy)
                fused_changes.append({
                    "id": f"CH-DR-{door_change_idx:03d}",
                    "type": "Door Change",
                    "old_value": f"{old_w_mm:.0f} mm",
                    "new_value": "Removed",
                    "delta": "Removed",
                    "bbox": (int(ax), int(ay), int(aw), int(ah)),
                    "real_dimensions": (float(round(aw * pixel_ratio, 1)), float(round(ah * pixel_ratio, 1))),
                    "confidence": 0.92,
                    "location": location,
                    "ssim_score": 0.0,
                    "explanation": f"Door ({old_w_mm:.0f} mm wide) was removed {location}."
                })
            else:
                rejected_count += 1
            
    for db in doors_b:
        if db.entity_id not in matched_doors_b:
            bx, by, bw, bh = db.bbox
            bcx, bcy = bx + bw / 2.0, by + bh / 2.0
            new_w_mm = bw * pixel_ratio
            
            if 600 <= new_w_mm <= 1800:
                door_change_idx += 1
                fused_changes.append({
                    "id": f"CH-DR-{door_change_idx:03d}",
                    "type": "Door Change",
                    "old_value": "",
                    "new_value": f"{new_w_mm:.0f} mm",
                    "delta": "Added",
                    "bbox": (int(bx), int(by), int(bw), int(bh)),
                    "real_dimensions": (float(round(bw * pixel_ratio, 1)), float(round(bh * pixel_ratio, 1))),
                    "confidence": 0.91,
                    "location": location_fn(bcx, bcy),
                    "ssim_score": 0.1,
                    "explanation": f"New door ({new_w_mm:.0f} mm wide) was added {location_fn(bcx, bcy)}."
                })
            else:
                rejected_count += 1

    # 3. Process Windows A vs B
    wins_a = [e for e in cv_entities_a if e.entity_type == "window"]
    wins_b = [e for e in cv_entities_b if e.entity_type == "window"]
    matched_wins_b = set()
    
    win_change_idx = len(fused_changes)
    for wa in wins_a:
        ax, ay, aw, ah = wa.bbox
        acx, acy = ax + aw / 2.0, ay + ah / 2.0
        
        best_match = None
        best_score = 0.0
        for wb in wins_b:
            if wb.entity_id in matched_wins_b:
                continue
            bx, by, bw, bh = wb.bbox
            bcx, bcy = bx + bw / 2.0, by + bh / 2.0
            
            iou = compute_iou(wa.bbox, wb.bbox)
            ar_a = aw / ah if ah > 0 else 1.0
            ar_b = bw / bh if bh > 0 else 1.0
            shape_sim = min(ar_a, ar_b) / max(ar_a, ar_b)
            ocr_sim = 1.0
            dist = np.hypot(bcx - acx, bcy - acy)
            centroid_prox = max(0.0, 1.0 - (dist / max_dim))
            
            score = 0.4 * iou + 0.2 * shape_sim + 0.2 * ocr_sim + 0.2 * centroid_prox
            if score > 0.8 and score > best_score:
                best_score = score
                best_match = wb
                
        if best_match:
            matched_wins_b.add(best_match.entity_id)
            bx, by, bw, bh = best_match.bbox
            
            old_w_mm = aw * pixel_ratio
            new_w_mm = bw * pixel_ratio
            
            if not (300 <= old_w_mm <= 5000) or not (300 <= new_w_mm <= 5000):
                rejected_count += 1
                continue
                
            delta_mm = new_w_mm - old_w_mm
            if abs(delta_mm) > 5000 or abs(delta_mm) > 0.5 * sheet_width_mm:
                rejected_count += 1
                continue
                
            if abs(delta_mm) > 10:
                win_change_idx += 1
                location = location_fn(bcx, bcy)
                
                crop_a = image_a[by:by+bh, bx:bx+bw]
                crop_b = image_b[by:by+bh, bx:bx+bw]
                ssim_val = get_ssim_score(crop_a, crop_b) if crop_a.size > 0 and crop_b.size > 0 else 0.5
                if ssim_val > 0.98:
                    rejected_count += 1
                    continue
                    
                confidence = 0.35 * alignment_score + 0.25 * best_score + 0.20 * 1.0 + 0.20 * (1.0 - ssim_val)
                if confidence > 0.85:
                    fused_changes.append({
                        "id": f"CH-WN-{win_change_idx:03d}",
                        "type": "Window Change",
                        "old_value": f"{old_w_mm:.0f} mm",
                        "new_value": f"{new_w_mm:.0f} mm",
                        "delta": f"{delta_mm:+.0f} mm",
                        "bbox": (int(bx), int(by), int(bw), int(bh)),
                        "real_dimensions": (float(round(bw * pixel_ratio, 1)), float(round(bh * pixel_ratio, 1))),
                        "confidence": float(round(confidence, 3)),
                        "location": location,
                        "ssim_score": float(round(ssim_val, 3)),
                        "explanation": generate_explanation("Window Change", f"{old_w_mm:.0f} mm", f"{new_w_mm:.0f} mm", location, bw * pixel_ratio, bh * pixel_ratio)
                    })
                else:
                    rejected_count += 1
        else:
            old_w_mm = aw * pixel_ratio
            if 300 <= old_w_mm <= 5000:
                win_change_idx += 1
                location = location_fn(acx, acy)
                fused_changes.append({
                    "id": f"CH-WN-{win_change_idx:03d}",
                    "type": "Window Change",
                    "old_value": f"{old_w_mm:.0f} mm",
                    "new_value": "Removed",
                    "delta": "Removed",
                    "bbox": (int(ax), int(ay), int(aw), int(ah)),
                    "real_dimensions": (float(round(aw * pixel_ratio, 1)), float(round(ah * pixel_ratio, 1))),
                    "confidence": 0.92,
                    "location": location,
                    "ssim_score": 0.0,
                    "explanation": f"Window ({old_w_mm:.0f} mm wide) was removed {location}."
                })
            else:
                rejected_count += 1
            
    for wb in wins_b:
        if wb.entity_id not in matched_wins_b:
            bx, by, bw, bh = wb.bbox
            bcx, bcy = bx + bw / 2.0, by + bh / 2.0
            new_w_mm = bw * pixel_ratio
            
            if 300 <= new_w_mm <= 5000:
                win_change_idx += 1
                fused_changes.append({
                    "id": f"CH-WN-{win_change_idx:03d}",
                    "type": "Window Change",
                    "old_value": "",
                    "new_value": f"{new_w_mm:.0f} mm",
                    "delta": "Added",
                    "bbox": (int(bx), int(by), int(bw), int(bh)),
                    "real_dimensions": (float(round(bw * pixel_ratio, 1)), float(round(bh * pixel_ratio, 1))),
                    "confidence": 0.91,
                    "location": location_fn(bcx, bcy),
                    "ssim_score": 0.1,
                    "explanation": f"New window ({new_w_mm:.0f} mm wide) was added {location_fn(bcx, bcy)}."
                })
            else:
                rejected_count += 1

    # 4. Process Remaining Geometry Changes (added lines / walls)
    geom_idx = len(fused_changes)
    for g_c in geometry_changes:
        x, y, w, h = g_c["bbox"]
        cx, cy = x + w / 2.0, y + h / 2.0
        
        area = w * h
        if area < 150 or (area / sheet_area) < 0.0001:
            rejected_count += 1
            continue
            
        overlapping = False
        for f_c in fused_changes:
            fx, fy, fw, fh = f_c["bbox"]
            ix = max(x, fx)
            iy = max(y, fy)
            iw = min(x + w, fx + fw) - ix
            ih = min(y + h, fy + fh) - iy
            if iw > 0 and ih > 0:
                intersection_area = iw * ih
                smaller_area = min(w * h, fw * fh)
                if intersection_area / smaller_area > 0.4:
                    overlapping = True
                    break
        if overlapping:
            # Overlaps with door/window change (silently skipped, not counted as false positive)
            continue
            
        location = location_fn(cx, cy)
        crop_a = image_a[y:y+h, x:x+w]
        crop_b = image_b[y:y+h, x:x+w]
        ssim_val = get_ssim_score(crop_a, crop_b) if crop_a.size > 0 and crop_b.size > 0 else 0.5
        
        if ssim_val > 0.92:
            rejected_count += 1
            continue
            
        real_w = w * pixel_ratio
        real_h = h * pixel_ratio
        
        if real_w > 5000 or real_h > 5000 or real_w > 0.5 * sheet_width_mm:
            rejected_count += 1
            continue
            
        c_class = "Wall Extension" if g_c["type"] == "added_line" and max(real_w, real_h) > 100 else "Geometry Change"
        if c_class == "Wall Extension":
            thickness = min(real_w, real_h)
            if not (50 <= thickness <= 1000):
                c_class = "Geometry Change"

        geom_idx += 1
        confidence = 0.35 * alignment_score + 0.25 * 0.9 + 0.20 * 0.5 + 0.20 * (1.0 - ssim_val)
        
        if confidence > 0.85:
            fused_changes.append({
                "id": f"CH-GM-{geom_idx:03d}",
                "type": c_class,
                "old_value": "Line work" if g_c["type"] == "removed_line" else "",
                "new_value": "Line work" if g_c["type"] == "added_line" else "",
                "delta": "Added" if g_c["type"] == "added_line" else "Removed",
                "bbox": (int(x), int(y), int(w), int(h)),
                "real_dimensions": (float(round(real_w, 1)), float(round(real_h, 1))),
                "confidence": float(round(confidence, 3)),
                "location": location,
                "ssim_score": float(round(ssim_val, 3)),
                "explanation": generate_explanation(c_class, "", "", location, real_w, real_h)
            })
        else:
            rejected_count += 1
        
    result_list = ChangesList(fused_changes)
    result_list.rejected_count = rejected_count
    result_list.total_candidates = len(fused_changes) + rejected_count
    return result_list
