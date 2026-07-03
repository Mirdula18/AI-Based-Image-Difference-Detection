import cv2
import numpy as np

from app.diff_engine import compute_diff


def _blank(size=(400, 400)):
    return np.full((size[1], size[0], 3), 255, dtype=np.uint8)


def test_identical_images_have_minimal_diff():
    img = _blank()
    cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 0), 2)
    result = compute_diff(img, img.copy(), mode="linework")
    assert np.count_nonzero(result.diff_mask) == 0


def test_added_rectangle_is_detected():
    img_a = _blank()
    cv2.rectangle(img_a, (50, 50), (150, 150), (0, 0, 0), 2)

    img_b = img_a.copy()
    cv2.rectangle(img_b, (250, 250), (350, 350), (0, 0, 0), 3)

    result = compute_diff(img_a, img_b, mode="linework")
    assert np.count_nonzero(result.diff_mask) > 0


def test_photo_mode_returns_ssim_score():
    img_a = _blank()
    img_b = _blank()
    cv2.circle(img_b, (200, 200), 50, (0, 0, 0), -1)
    result = compute_diff(img_a, img_b, mode="photo")
    assert 0.0 <= result.raw_score <= 1.0
    assert np.count_nonzero(result.diff_mask) > 0
