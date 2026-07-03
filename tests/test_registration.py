import cv2
import numpy as np

from app.registration import register_images


def _drawing_with_border(offset=(0, 0), scale=1.0, size=(600, 800)):
    h, w = size
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (40, 40), (w - 40, h - 40), (0, 0, 0), 3)
    cv2.line(img, (200, 40), (200, h - 40), (0, 0, 0), 2)
    cv2.putText(img, "TEST", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    if offset != (0, 0) or scale != 1.0:
        transform = np.array(
            [[scale, 0, offset[0]], [0, scale, offset[1]]], dtype=np.float32
        )
        img = cv2.warpAffine(img, transform, (w, h), borderValue=(255, 255, 255))
    return img


def test_register_identical_images_succeeds():
    img = _drawing_with_border()
    result = register_images(img, img.copy())
    assert result.success
    assert result.alignment_score > 0.9


def test_register_offset_image_succeeds_via_border_or_feature():
    img_a = _drawing_with_border()
    img_b = _drawing_with_border(offset=(15, -10), scale=1.03)
    result = register_images(img_a, img_b)
    assert result.method in ("border", "feature")
    assert result.aligned_b.shape == img_a.shape


def test_register_blank_images_reports_failure():
    blank_a = np.full((400, 400, 3), 255, dtype=np.uint8)
    blank_b = np.full((400, 400, 3), 255, dtype=np.uint8)
    result = register_images(blank_a, blank_b)
    assert result.success is False
    assert "fail" in result.message.lower() or result.method == "none"
