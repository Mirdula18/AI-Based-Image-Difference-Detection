from app.diff_engine import DiffResult
from app.registration import RegistrationResult
from app.regions import Region
from app.stats import build_stats
import numpy as np


def test_build_stats_computes_percent_area():
    registration = RegistrationResult(
        aligned_b=np.zeros((100, 100, 3), dtype=np.uint8),
        success=True,
        method="border",
        alignment_score=0.8,
        message="ok",
    )
    diff_result = DiffResult(
        diff_mask=np.zeros((100, 100), dtype=np.uint8),
        binary_a=np.zeros((100, 100), dtype=np.uint8),
        binary_b=np.zeros((100, 100), dtype=np.uint8),
        mode="linework",
        raw_score=0.01,
    )
    regions = [
        Region(bbox=(0, 0, 10, 10), area_px=100, centroid=(5, 5), location="upper-left"),
    ]
    stats = build_stats(registration, diff_result, regions, (100, 100))
    assert stats.total_region_count == 1
    assert stats.percent_area_changed == 1.0
    assert stats.regions[0].location == "upper-left"
    d = stats.to_dict()
    assert d["registration_success"] is True
