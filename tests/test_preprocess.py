import numpy as np

from app.preprocess import normalize_pair


def test_normalize_pair_same_size():
    a = np.zeros((500, 400, 3), dtype=np.uint8)
    b = np.zeros((500, 400, 3), dtype=np.uint8)
    result = normalize_pair(a, b, max_dim=1000)
    assert result.image_a.shape == result.image_b.shape


def test_normalize_pair_downscales_to_smaller_input():
    a = np.zeros((1000, 800, 3), dtype=np.uint8)
    b = np.zeros((500, 400, 3), dtype=np.uint8)
    result = normalize_pair(a, b, max_dim=2000)
    # Common target should follow the smaller image's scale (400x500-ish),
    # not the larger one.
    assert max(result.image_a.shape[:2]) <= 500 + 2
    assert result.image_a.shape == result.image_b.shape


def test_normalize_pair_respects_max_dim():
    a = np.zeros((3000, 2000, 3), dtype=np.uint8)
    b = np.zeros((3000, 2000, 3), dtype=np.uint8)
    result = normalize_pair(a, b, max_dim=1000)
    assert max(result.image_a.shape[:2]) <= 1000 + 2
