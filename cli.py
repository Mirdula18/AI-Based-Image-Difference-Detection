#!/usr/bin/env python
"""Run the full comparison pipeline on two local files without the API/frontend.

Usage:
    python cli.py path/to/A.png path/to/B.pdf --out ./output --mode linework
"""
from __future__ import annotations

import argparse
import json
import sys

from app.ingest import IngestionError
from app.pipeline import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_a", help="Path to the original/older drawing (JPG/PNG/PDF)")
    parser.add_argument("image_b", help="Path to the revised/newer drawing (JPG/PNG/PDF)")
    parser.add_argument(
        "--out", default="./output", help="Directory to write result images/stats into"
    )
    parser.add_argument(
        "--mode",
        choices=["linework", "photo"],
        default="linework",
        help="Diff mode: 'linework' for CAD/line-art drawings, 'photo' for general images (SSIM-based)",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip the AI summary step (useful for offline / no API key runs)",
    )
    args = parser.parse_args()

    try:
        result = run_pipeline(
            args.image_a,
            args.image_b,
            args.out,
            diff_mode=args.mode,
            generate_summary=not args.no_summary,
        )
    except IngestionError as exc:
        print(f"Ingestion error: {exc}", file=sys.stderr)
        return 1

    print("=== Stats ===")
    print(json.dumps(result.stats, indent=2))
    print("\n=== Summary ===")
    print(result.summary or "(summary skipped)")
    print("\n=== Output images ===")
    for name, path in result.image_paths.items():
        print(f"  {name}: {path}")

    with open(f"{args.out}/stats.json", "w", encoding="utf-8") as f:
        json.dump({"stats": result.stats, "summary": result.summary}, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
