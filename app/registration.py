"""Stage 3: Registration -- the critical alignment step.

Naive pixel-by-pixel diffing fails whenever the two input images differ in
zoom, scale, or position. This module warps image B onto image A's
coordinate space before any diffing happens, using two strategies:

1. Border/title-block detection: find the largest rectangular contour in
   each image (the drawing sheet's frame/border) and compute a
   perspective transform mapping B's frame corners onto A's.
2. ORB feature matching + RANSAC homography: a general-purpose fallback
   for images without a clean rectangular border, or when method 1 fails
   to find a confident quadrilateral in both images.

After warping, an alignment sanity check (normalized cross-correlation of
Canny edge maps) is used to decide whether registration actually
succeeded -- callers must check `RegistrationResult.success` rather than
assuming the warp is meaningful.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from app.config import (
    BORDER_CONTOUR_MIN_AREA_RATIO,
    BORDER_APPROX_EPSILON_RATIO,
    ORB_MAX_FEATURES,
    LOWE_RATIO_TEST,
    MIN_GOOD_MATCHES,
    RANSAC_REPROJ_THRESHOLD,
    MIN_ALIGNMENT_SCORE,
)


@dataclass
class RegistrationResult:
    aligned_b: np.ndarray
    success: bool
    method: str  # "border", "feature", or "none"
    alignment_score: float
    message: str


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]       # top-left
    ordered[2] = pts[np.argmax(s)]       # bottom-right
    ordered[1] = pts[np.argmin(diff)]    # top-right
    ordered[3] = pts[np.argmax(diff)]    # bottom-left
    return ordered


def _find_border_quad(image: np.ndarray) -> Optional[np.ndarray]:
    """Locate the largest rectangular contour (drawing border/title block).

    Returns 4 ordered corner points, or None if no confident quadrilateral
    covering a large enough fraction of the frame is found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = image.shape[0] * image.shape[1]
    best_quad = None
    best_area = 0.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < frame_area * BORDER_CONTOUR_MIN_AREA_RATIO:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, BORDER_APPROX_EPSILON_RATIO * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            if area > best_area:
                best_area = area
                best_quad = approx

    if best_quad is None:
        return None
    return _order_quad_points(best_quad)


def _edge_correlation_score(image_a: np.ndarray, warped_b: np.ndarray) -> float:
    """Normalized cross-correlation between Canny edge maps of A and warped B.

    Used as a coordinate-space-agnostic proxy for "did the warp actually
    line the drawings up". Areas warped in as blank (black) padding are
    excluded from both edge maps via a validity mask so they don't
    artificially inflate or deflate the score.
    """
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(warped_b, cv2.COLOR_BGR2GRAY)

    valid_mask = (warped_b.sum(axis=2) > 0)
    if valid_mask.mean() < 0.1:
        return 0.0

    edges_a = cv2.Canny(gray_a, 50, 150).astype(np.float32)
    edges_b = cv2.Canny(gray_b, 50, 150).astype(np.float32)

    edges_a = edges_a[valid_mask]
    edges_b = edges_b[valid_mask]

    if edges_a.std() < 1e-6 or edges_b.std() < 1e-6:
        return 0.0

    correlation = np.corrcoef(edges_a, edges_b)[0, 1]
    if np.isnan(correlation):
        return 0.0
    return float(max(0.0, correlation))


def _register_via_border(
    image_a: np.ndarray, image_b: np.ndarray
) -> Optional[np.ndarray]:
    quad_a = _find_border_quad(image_a)
    quad_b = _find_border_quad(image_b)
    if quad_a is None or quad_b is None:
        return None

    h, w = image_a.shape[:2]
    transform = cv2.getPerspectiveTransform(quad_b, quad_a)
    warped_b = cv2.warpPerspective(image_b, transform, (w, h))
    return warped_b


def _register_via_features(
    image_a: np.ndarray, image_b: np.ndarray
) -> Optional[np.ndarray]:
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=ORB_MAX_FEATURES)
    kp_a, des_a = orb.detectAndCompute(gray_a, None)
    kp_b, des_b = orb.detectAndCompute(gray_b, None)

    if des_a is None or des_b is None or len(kp_a) < 4 or len(kp_b) < 4:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = bf.knnMatch(des_b, des_a, k=2)

    good_matches = []
    for pair in raw_matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < LOWE_RATIO_TEST * n.distance:
            good_matches.append(m)

    if len(good_matches) < MIN_GOOD_MATCHES:
        return None

    pts_b = np.float32([kp_b[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_a = np.float32([kp_a[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(
        pts_b, pts_a, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD
    )
    if homography is None:
        return None

    inlier_count = int(mask.sum()) if mask is not None else 0
    if inlier_count < MIN_GOOD_MATCHES:
        return None

    h, w = image_a.shape[:2]
    warped_b = cv2.warpPerspective(image_b, homography, (w, h))
    return warped_b


def register_images(image_a: np.ndarray, image_b: np.ndarray) -> RegistrationResult:
    """Align image B onto image A's coordinate space.

    Tries border/title-block detection first (fast, robust for CAD-style
    sheets with a fixed frame), then falls back to ORB feature matching +
    RANSAC homography for general photographic inputs. Reports failure
    rather than returning a meaningless warp if neither method produces
    an alignment that passes the edge-correlation sanity check.
    """
    candidates: list[tuple[str, np.ndarray]] = []

    border_result = _register_via_border(image_a, image_b)
    if border_result is not None:
        candidates.append(("border", border_result))

    feature_result = _register_via_features(image_a, image_b)
    if feature_result is not None:
        candidates.append(("feature", feature_result))

    if not candidates:
        return RegistrationResult(
            aligned_b=image_b,
            success=False,
            method="none",
            alignment_score=0.0,
            message=(
                "Registration failed: no drawing border/title-block quadrilateral "
                "could be detected in both images, and feature matching did not "
                "find enough reliable keypoint correspondences. Falling back to "
                "the unaligned image; downstream diff results are not meaningful."
            ),
        )

    best_method, best_warped, best_score = None, None, -1.0
    for method, warped in candidates:
        score = _edge_correlation_score(image_a, warped)
        if score > best_score:
            best_method, best_warped, best_score = method, warped, score

    success = best_score >= MIN_ALIGNMENT_SCORE
    message = (
        f"Registered via '{best_method}' method with alignment score {best_score:.2f}."
        if success
        else (
            f"Registration attempted via '{best_method}' but alignment score "
            f"{best_score:.2f} is below the confidence threshold "
            f"({MIN_ALIGNMENT_SCORE}). Results may be unreliable."
        )
    )

    return RegistrationResult(
        aligned_b=best_warped,
        success=success,
        method=best_method,
        alignment_score=best_score,
        message=message,
    )
