import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class Entity:
    entity_id: str
    entity_type: str
    bbox: tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    attributes: dict[str, Any] = field(default_factory=dict)

def classify_drawing_category(ocr_texts: list[str]) -> str:
    """Classifies the drawing type based on text content."""
    texts_lower = [t.lower() for t in ocr_texts]
    
    arch_kws = ["floor plan", "elevation", "door", "window", "room", "living", "bedroom", "closet", "kitchen", "bathroom", "stairs"]
    mech_kws = ["drill", "thread", "chamfer", "tolerance", "shaft", "bolt", "flange", "assembly", "gear", "coupling"]
    elec_kws = ["wiring", "breaker", "panel", "schematic", "circuit", "volt", "cable", "switch", "transformer", "junction"]
    struct_kws = ["beam", "rebar", "slab", "foundation", "footing", "concrete", "steel", "truss", "girder"]
    piping_kws = ["pipe", "valve", "p&id", "flange", "pump", "fitting", "compressor", "isometric"]

    scores = {
        "architectural": sum(1 for kw in arch_kws if any(kw in t for t in texts_lower)),
        "mechanical": sum(1 for kw in mech_kws if any(kw in t for t in texts_lower)),
        "electrical": sum(1 for kw in elec_kws if any(kw in t for t in texts_lower)),
        "structural": sum(1 for kw in struct_kws if any(kw in t for t in texts_lower)),
        "piping": sum(1 for kw in piping_kws if any(kw in t for t in texts_lower)),
    }
    
    best_type = max(scores, key=scores.get)
    if scores[best_type] == 0:
        return "architectural"  # Default fallback
    return best_type

def detect_entities(
    image: np.ndarray,
    drawing_type: str,
    ocr_results: list[dict[str, Any]]
) -> list[Entity]:
    """Detects semantic entities in the drawing using classical CV and OCR text fusion."""
    entities: list[Entity] = []
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold for finding contours
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    door_count = 0
    window_count = 0
    wall_count = 0
    column_count = 0
    table_count = 0
    cloud_count = 0
    
    # 1. OCR-based Entity Extraction (Rooms, Labels, Title Blocks)
    for idx, ocr in enumerate(ocr_results):
        text = ocr["text"].strip()
        bbox = ocr["bbox"]  # x, y, w, h
        ibox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        conf = ocr.get("confidence", 0.90)
        
        # Check for Room labels
        if any(rm in text.upper() for rm in ["ROOM", "BEDROOM", "LIVING", "KITCHEN", "DINING", "CLOSET", "BATHROOM", "HALL"]):
            entities.append(Entity(
                entity_id=f"Room-{text.replace(' ', '_')}",
                entity_type="room",
                bbox=ibox,
                confidence=conf,
                attributes={"label": text}
            ))
        # Title block sheets text
        elif any(tb in text.upper() for tb in ["SHEET", "SCALE", "PROJECT", "REV", "DRAWN"]):
            entities.append(Entity(
                entity_id=f"TitleText-{idx}",
                entity_type="title_block_text",
                bbox=ibox,
                confidence=conf,
                attributes={"text": text}
            ))
        # Annotations / Dimensions
        elif any(unit in text.lower() for unit in ["mm", "mtr", "ft", "inch", "'", "\"", "dia", "ø"]):
            entities.append(Entity(
                entity_id=f"DimText-{idx}",
                entity_type="dimension_label",
                bbox=ibox,
                confidence=conf,
                attributes={"text": text}
            ))
        else:
            entities.append(Entity(
                entity_id=f"LabelText-{idx}",
                entity_type="text_label",
                bbox=ibox,
                confidence=conf,
                attributes={"text": text}
            ))

    # 2. Contour-based CV Entity Detection
    for c in contours:
        area = cv2.contourArea(c)
        if area < 50:
            continue
            
        x, y, cw, ch = cv2.boundingRect(c)
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
            
        # Compute convexity/circularity/aspect ratio
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        aspect_ratio = float(cw) / ch if ch > 0 else 0
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Detect Title Block (large rectangles near borders)
        if area > (w * h * 0.05) and solidity > 0.9:
            # Check if near right or bottom edge
            if x + cw >= w - 100 or y + ch >= h - 100:
                entities.append(Entity(
                    entity_id="TitleBlock",
                    entity_type="title_block",
                    bbox=(x, y, cw, ch),
                    confidence=0.95,
                    attributes={"area_ratio": area / (w * h)}
                ))
                continue

        # Detect Revision Clouds (high perimeter, low convexity, curvy shape)
        if solidity < 0.70 and perimeter > 300:
            cloud_count += 1
            entities.append(Entity(
                entity_id=f"Cloud-{cloud_count}",
                entity_type="revision_cloud",
                bbox=(x, y, cw, ch),
                confidence=0.85,
                attributes={"solidity": solidity, "perimeter": perimeter}
            ))
            continue

        # Detect Columns (small filled square/circle regions with high solidity)
        if 200 < area < 5000 and solidity > 0.85:
            if 0.8 < aspect_ratio < 1.2:
                column_count += 1
                entities.append(Entity(
                    entity_id=f"Column-{column_count}",
                    entity_type="column",
                    bbox=(x, y, cw, ch),
                    confidence=0.80,
                    attributes={"solidity": solidity}
                ))
                continue

        # Detect Doors (characterized by an arc quadrant + aspect ratio near 1.0)
        # Note: door swings are drawn as curves
        if 500 < area < 20000 and 0.4 < aspect_ratio < 2.5:
            # Check if contour resembles a quarter circle arc (low solidity, moderate circularity)
            if 0.2 < circularity < 0.75 and solidity < 0.75:
                door_count += 1
                entities.append(Entity(
                    entity_id=f"Door-CV-{door_count}",
                    entity_type="door",
                    bbox=(x, y, cw, ch),
                    confidence=0.85,
                    attributes={"aspect_ratio": aspect_ratio}
                ))
                continue

        # Detect Windows (parallel lines inside a long narrow bounding box)
        if 100 < area < 15000 and (aspect_ratio > 3.0 or aspect_ratio < 0.33):
            # Narrow rectangular slot, typical of a window pane
            if solidity > 0.8:
                window_count += 1
                entities.append(Entity(
                    entity_id=f"Window-CV-{window_count}",
                    entity_type="window",
                    bbox=(x, y, cw, ch),
                    confidence=0.80,
                    attributes={"orientation": "horizontal" if aspect_ratio > 1.0 else "vertical"}
                ))
                continue

    # 3. Wall Segment Detection (Hough lines)
    # Detect long line segments and group them
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=50, maxLineGap=10)

    if lines is not None:
        for idx, line in enumerate(lines):
            flat = line.flatten()
            if len(flat) != 4:
                continue
            lx1, ly1, lx2, ly2 = flat
            length = np.hypot(lx2 - lx1, ly2 - ly1)
            if length > 80:
                wall_count += 1
                bx = min(lx1, lx2)
                by = min(ly1, ly2)
                bw = max(2, abs(lx2 - lx1))
                bh = max(2, abs(ly2 - ly1))
                entities.append(Entity(
                    entity_id=f"Wall-{wall_count}",
                    entity_type="wall",
                    bbox=(bx - 2, by - 2, bw + 4, bh + 4),
                    confidence=0.75,
                    attributes={"p1": (lx1, ly1), "p2": (lx2, ly2), "length": length}
                ))

    return entities

def compare_geometry(image_a: np.ndarray, image_b: np.ndarray) -> list[dict[str, Any]]:
    """Compares the line work / vector geometry between A and B, isolating changes.

    Pipeline: Binary Conversion -> Skeletonization -> Line Extraction -> Vector Match.
    """
    from skimage.morphology import skeletonize
    
    changes = []
    h, w = image_a.shape[:2]
    sheet_area = h * w

    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
    
    # Binary conversion (Foreground is inverted to True/1 for skeletonization)
    _, bin_a = cv2.threshold(gray_a, 200, 255, cv2.THRESH_BINARY_INV)
    _, bin_b = cv2.threshold(gray_b, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Skeletonization (reduces thickness and antialiasing artifacts)
    skel_a = skeletonize(bin_a > 0).astype(np.uint8) * 255
    skel_b = skeletonize(bin_b > 0).astype(np.uint8) * 255
    
    # Line Extraction via Hough Lines
    lines_a = cv2.HoughLinesP(skel_a, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=5)
    lines_b = cv2.HoughLinesP(skel_b, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=5)
    
    list_a = []
    if lines_a is not None:
        for l in lines_a:
            flat = l.flatten()
            if len(flat) == 4:
                list_a.append(flat)
                
    list_b = []
    if lines_b is not None:
        for l in lines_b:
            flat = l.flatten()
            if len(flat) == 4:
                list_b.append(flat)
    
    added_lines = []
    removed_lines = []
    
    for lb in list_b:
        bx1, by1, bx2, by2 = lb
        bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
        
        matched = False
        for la in list_a:
            ax1, ay1, ax2, ay2 = la
            acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
            
            # 1. Centroid distance check
            dist = np.hypot(bcx - acx, bcy - acy)
            if dist < 15:
                # 2. Orientation check
                v_a = np.array([ax2 - ax1, ay2 - ay1], dtype=np.float32)
                v_b = np.array([bx2 - bx1, by2 - by1], dtype=np.float32)
                len_a = np.linalg.norm(v_a)
                len_b = np.linalg.norm(v_b)
                
                if len_a > 0 and len_b > 0:
                    cos_theta = abs(np.dot(v_a, v_b) / (len_a * len_b))
                    if cos_theta > 0.95:  # ~18 degrees
                        # 3. Line shift (perpendicular distance) check
                        # Distance from midpoint P(bcx, bcy) to line la
                        num = abs((ax2 - ax1) * (ay1 - bcy) - (ax1 - bcx) * (ay2 - ay1))
                        perp_dist = num / len_a
                        
                        # Ignore line shifts less than 2px
                        if perp_dist < 2.0:
                            matched = True
                            break
                            
        if not matched:
            added_lines.append(lb)
            
    for la in list_a:
        ax1, ay1, ax2, ay2 = la
        acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
        
        matched = False
        for lb in list_b:
            bx1, by1, bx2, by2 = lb
            bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
            
            dist = np.hypot(bcx - acx, bcy - acy)
            if dist < 15:
                v_a = np.array([ax2 - ax1, ay2 - ay1], dtype=np.float32)
                v_b = np.array([bx2 - bx1, by2 - by1], dtype=np.float32)
                len_a = np.linalg.norm(v_a)
                len_b = np.linalg.norm(v_b)
                
                if len_a > 0 and len_b > 0:
                    cos_theta = abs(np.dot(v_a, v_b) / (len_a * len_b))
                    if cos_theta > 0.95:
                        num = abs((ax2 - ax1) * (ay1 - bcy) - (ax1 - bcx) * (ay2 - ay1))
                        perp_dist = num / len_a
                        if perp_dist < 2.0:
                            matched = True
                            break
        if not matched:
            removed_lines.append(la)
            
    for al in added_lines:
        x1, y1, x2, y2 = al
        w_px = max(4, abs(x2 - x1))
        h_px = max(4, abs(y2 - y1))
        area = w_px * h_px
        
        # Ignore changes below component_area < 150px and changed_area < 0.01%
        if area >= 150 and (area / sheet_area) >= 0.0001:
            changes.append({
                "type": "added_line",
                "bbox": (int(min(x1, x2)), int(min(y1, y2)), int(w_px), int(h_px)),
                "attributes": {"p1": (int(x1), int(y1)), "p2": (int(x2), int(y2))}
            })
        
    for rl in removed_lines:
        x1, y1, x2, y2 = rl
        w_px = max(4, abs(x2 - x1))
        h_px = max(4, abs(y2 - y1))
        area = w_px * h_px
        
        if area >= 150 and (area / sheet_area) >= 0.0001:
            changes.append({
                "type": "removed_line",
                "bbox": (int(min(x1, x2)), int(min(y1, y2)), int(w_px), int(h_px)),
                "attributes": {"p1": (int(x1), int(y1)), "p2": (int(x2), int(y2))}
            })
        
    return changes
