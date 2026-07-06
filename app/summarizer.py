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

    if not stats.change_objects:
        if stats.total_region_count == 0:
            return (
                "No significant differences were detected between the two "
                "drawing revisions after alignment; the sheets appear "
                "effectively identical within the configured sensitivity "
                "threshold."
            )
        else:
            # Fallback when there are regions but no detailed change_objects yet
            top_regions = stats.regions[:3]
            locations = ", ".join(r.location for r in top_regions)
            severity = stats.severity_index.lower()
            return (
                f"Comparison of the two drawing revisions found {stats.total_region_count} "
                f"changed region(s) covering approximately {stats.percent_area_changed}% "
                f"of the sheet area, indicating a {severity} revision. The largest "
                f"changes are located in the {locations} area(s) of the sheet."
            )

    # Compile explanations from change objects
    explanations = []
    
    # Group changes by type
    doors = [c for c in stats.change_objects if c["type"] == "Door Change"]
    windows = [c for c in stats.change_objects if c["type"] == "Window Change"]
    rooms = [c for c in stats.change_objects if c["type"] == "Room Change"]
    dims = [c for c in stats.change_objects if c["type"] == "Dimension Change"]
    walls = [c for c in stats.change_objects if c["type"] == "Wall Extension"]
    
    # Compile a few select major modifications
    for r in rooms[:2]:
        if r.get("old_value") and r.get("new_value"):
            explanations.append(f"Room '{r['old_value']}' was converted to '{r['new_value']}'")
        elif r.get("new_value"):
            explanations.append(f"Room '{r['new_value']}' was added")
            
    for d in doors[:2]:
        delta_str = d.get("delta", "")
        if "mm" in delta_str:
            explanations.append(f"Door was adjusted by {delta_str}")
        elif delta_str == "Added":
            explanations.append(f"A new door was added {d.get('location', '')}")
        elif delta_str == "Removed":
            explanations.append(f"A door was removed {d.get('location', '')}")
            
    for w in windows[:2]:
        delta_str = w.get("delta", "")
        if "mm" in delta_str:
            explanations.append(f"Window width was changed by {delta_str}")
        elif delta_str == "Added":
            explanations.append(f"A window was added")
            
    for dm in dims[:2]:
        explanations.append(f"Dimension annotation was updated from '{dm['old_value']}' to '{dm['new_value']}'")

    for wl in walls[:1]:
        explanations.append(f"New wall structures were extended {wl.get('location', '')}")

    # Build summary sentences
    summary_parts = []
    
    total_mods = stats.total_changes
    if total_mods == 1:
        summary_parts.append("One modification was detected.")
    elif total_mods > 1:
        summary_parts.append(f"{total_mods} modifications were detected.")
    else:
        summary_parts.append("No semantic modifications were detected.")

    if explanations:
        # Join explanations with appropriate punctuation
        summary_parts.append(". ".join(explanations) + ".")
        
    summary_parts.append(
        f"Approximately {stats.percent_area_changed}% of the drawing sheet was modified, "
        f"representing a {stats.severity_index.lower()} severity revision."
    )

    return " ".join(summary_parts)


def summarize(stats: ComparisonStats) -> str:
    """Generate a natural-language summary paragraph from structured stats.

    Since the system must run entirely offline, it uses a deterministic
    rule-based generator that synthesizes explanations of semantic changes.
    """
    return _fallback_summary(stats)

