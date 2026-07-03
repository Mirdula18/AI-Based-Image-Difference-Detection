from app.stats import ComparisonStats, RegionStat
from app.summarizer import summarize


def _stats(success=True, region_count=1, percent=2.5):
    return ComparisonStats(
        registration_success=success,
        registration_method="border",
        registration_message="ok",
        alignment_score=0.8,
        diff_mode="linework",
        image_width=800,
        image_height=600,
        total_region_count=region_count,
        percent_area_changed=percent,
        regions=[
            RegionStat(
                index=0,
                bbox={"x": 0, "y": 0, "w": 10, "h": 10},
                area_px=100,
                centroid={"x": 5, "y": 5},
                location="upper-left",
            )
        ]
        if region_count
        else [],
    )


def test_summarize_without_api_key_uses_fallback(monkeypatch):
    monkeypatch.setattr("app.summarizer.ANTHROPIC_API_KEY", None)
    summary = summarize(_stats())
    assert isinstance(summary, str) and len(summary) > 0
    assert "upper-left" in summary


def test_summarize_reports_registration_failure(monkeypatch):
    monkeypatch.setattr("app.summarizer.ANTHROPIC_API_KEY", None)
    summary = summarize(_stats(success=False))
    assert "not reliable" in summary.lower() or "did not reach" in summary.lower()


def test_summarize_no_changes(monkeypatch):
    monkeypatch.setattr("app.summarizer.ANTHROPIC_API_KEY", None)
    summary = summarize(_stats(region_count=0, percent=0.0))
    assert "no significant differences" in summary.lower()
