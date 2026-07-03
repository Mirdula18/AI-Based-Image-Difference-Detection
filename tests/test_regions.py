import numpy as np

from app.regions import extract_regions


def test_extract_regions_finds_blobs_above_threshold():
    mask = np.zeros((400, 400), dtype=np.uint8)
    mask[20:60, 20:60] = 255  # 40x40 = 1600px, upper-left
    mask[300:340, 300:340] = 255  # lower-right
    regions = extract_regions(mask, min_area_px=100)
    assert len(regions) == 2
    locations = {r.location for r in regions}
    assert "upper-left" in locations
    assert "lower-right" in locations


def test_extract_regions_filters_small_noise():
    mask = np.zeros((400, 400), dtype=np.uint8)
    mask[10:12, 10:12] = 255  # 2x2 = 4px, below threshold
    regions = extract_regions(mask, min_area_px=100)
    assert len(regions) == 0


def test_region_bbox_and_centroid_are_reasonable():
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[50:100, 60:120] = 255
    regions = extract_regions(mask, min_area_px=50)
    assert len(regions) == 1
    x, y, w, h = regions[0].bbox
    assert (x, y) == (60, 50)
    assert w == 60 and h == 50
