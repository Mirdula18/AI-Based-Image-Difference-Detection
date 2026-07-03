"""Stage 8: AI summary.

Takes the structured stats JSON (never raw pixels -- for speed, cost, and
reliability) and asks an LLM to write one concise paragraph describing
the overall result, major changed regions, approximate locations, and
severity of change.

If no API key is configured, or the API call fails for any reason, a
deterministic template-based summary is returned instead so the rest of
the pipeline keeps working end to end (e.g. for offline CLI testing).
"""
from __future__ import annotations

import json

from app.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, SUMMARY_MAX_TOKENS
from app.stats import ComparisonStats

PROMPT_TEMPLATE = """You are assisting an architect/engineer reviewing an automated \
comparison between two revisions (A = older, B = newer) of a drawing sheet.

Below is a structured JSON description of the detected differences \
(bounding boxes, areas, and rough locations of changed regions). Write ONE \
concise paragraph (3-5 sentences) summarizing:
- the overall result (did registration succeed, how much changed overall)
- the major changed regions and their approximate locations
- the extent/severity of the change (minor tweak vs. substantial revision)

Do not invent details not present in the data. If registration failed, say \
so plainly and note that the diff may not be reliable.

Structured data:
{stats_json}
"""


def _fallback_summary(stats: ComparisonStats) -> str:
    if not stats.registration_success:
        return (
            "Automated alignment between the two drawings did not reach a "
            "confident match (registration method: "
            f"{stats.registration_method}, score {stats.alignment_score}), so "
            "the difference results below may not be reliable. Please verify "
            "manually or supply clearer/higher-resolution source files."
        )

    if stats.total_region_count == 0:
        return (
            "No significant differences were detected between the two "
            "drawing revisions after alignment; the sheets appear "
            "effectively identical within the configured sensitivity "
            "threshold."
        )

    top_regions = stats.regions[:3]
    locations = ", ".join(r.location for r in top_regions)
    severity = (
        "substantial"
        if stats.percent_area_changed > 5
        else "moderate"
        if stats.percent_area_changed > 1
        else "minor"
    )
    return (
        f"Comparison of the two drawing revisions found {stats.total_region_count} "
        f"changed region(s) covering approximately {stats.percent_area_changed}% "
        f"of the sheet area, indicating a {severity} revision. The largest "
        f"changes are located in the {locations} area(s) of the sheet. "
        f"Alignment was performed via the '{stats.registration_method}' method "
        f"with an alignment confidence score of {stats.alignment_score}."
    )


def summarize(stats: ComparisonStats) -> str:
    """Generate a natural-language summary paragraph from structured stats.

    Attempts a live LLM call if ANTHROPIC_API_KEY is configured; otherwise
    (or on any API error) falls back to a deterministic template so the
    pipeline never hard-fails on the summary step.
    """
    if not ANTHROPIC_API_KEY:
        return _fallback_summary(stats)

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        prompt = PROMPT_TEMPLATE.format(
            stats_json=json.dumps(stats.to_dict(), indent=2)
        )
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=SUMMARY_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        text_parts = [block.text for block in response.content if block.type == "text"]
        summary = "".join(text_parts).strip()
        return summary or _fallback_summary(stats)
    except Exception:
        return _fallback_summary(stats)
