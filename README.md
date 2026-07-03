# AI-Based Drawing / Image Difference Detection

Compares two revisions of an architectural/CAD drawing (or general image) and produces a visual diff report with an AI-generated natural-language summary.

This is a **classical computer-vision pipeline** — no model training, no GPU required. An LLM (Anthropic Claude) is used **only** for the final text-summary step, and the system degrades gracefully to a template summary when no API key is configured.

## How it works

```
A, B (JPG/PNG/PDF)
   │
   ▼
1. Ingestion (app/ingest.py) ── PDF pages rasterized at 300 DPI via PyMuPDF
   ▼
2. Normalization (app/preprocess.py) ── common working resolution, aspect ratio preserved
   ▼
3. Registration (app/registration.py) ── THE critical step
   │    primary:  detect drawing border / title-block quadrilateral in both
   │              images → perspective transform B onto A
   │    fallback: ORB keypoints + BFMatcher ratio test + RANSAC homography
   │    sanity:   edge-map correlation score; failure is REPORTED, never silent
   ▼
4. Difference detection (app/diff_engine.py)
   │    linework mode: adaptive threshold → binary XOR → morphological cleanup
   │    photo mode:    SSIM diff map (for non-line-art inputs)
   ▼
5. Region extraction (app/regions.py) ── contours → bbox, area, centroid,
   │                                     location descriptor, noise filtering
   ▼
6. Visualization (app/visualize.py) ── bounding boxes, heatmap,
   │                                   added/removed overlay, side-by-side
   ▼
7. Statistics (app/stats.py) ── structured JSON (this is what the LLM sees)
   ▼
8. AI summary (app/summarizer.py) ── one-paragraph summary from stats JSON only
```

Naive pixel-by-pixel diffing is deliberately **not** used — the two inputs are never assumed to be pixel-aligned. Registration warps image B into image A's coordinate space first, and the result is validated with an alignment-score check before any diff is trusted.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Optional — for live AI summaries (otherwise a deterministic template summary is used):

```bash
# Windows (PowerShell):
$env:ANTHROPIC_API_KEY = "sk-ant-..."
# macOS/Linux:
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Quick CLI test (no API/frontend needed)

Generate the bundled synthetic sample drawings, then run the pipeline on them:

```bash
python samples/generate_samples.py
python cli.py samples/sample_a.png samples/sample_b.png --out samples/output
```

This prints the stats JSON + summary to the console and writes all visual artifacts (`annotated_regions.png`, `heatmap.png`, `added_removed_overlay.png`, `side_by_side.png`, …) plus `stats.json` into `samples/output/`.

Useful flags:

- `--mode photo` — SSIM-based diff for general photographic images instead of line-art drawings
- `--no-summary` — skip the AI-summary step entirely

Works with PDFs too: `python cli.py revA.pdf revB.pdf --out ./output`

## Running the full API + frontend

**Terminal 1 — FastAPI backend:**

```bash
uvicorn main:app --reload
```

- `POST /compare` — multipart upload of `file_a` and `file_b` (JPG/PNG/PDF), optional `?mode=linework|photo`. Returns JSON with artifact URLs, the stats object, and the summary text.
- `GET /health` — liveness check.
- Interactive docs at http://localhost:8000/docs.

**Terminal 2 — Streamlit frontend:**

```bash
streamlit run frontend/streamlit_app.py
```

Open http://localhost:8501, upload two files, click **Compare**. The page shows: original A, original B, the highlighted-regions diff, the added/removed overlay (red = added in B, blue = removed from A), the heatmap, the statistics, and the AI summary paragraph.

If the API runs on a different host/port, set `DIFF_API_URL` for the frontend.

## Running the tests

```bash
pip install pytest
pytest tests/ -v
```

Each pipeline stage has its own test module (`tests/test_ingest.py`, `test_preprocess.py`, `test_registration.py`, `test_diff_engine.py`, `test_regions.py`, `test_stats.py`, `test_summarizer.py`).

## Failure handling

- Unsupported/corrupt/missing input files → clear `IngestionError` (HTTP 400 from the API).
- Registration failure (no border found **and** insufficient feature matches, or low alignment confidence) → `registration_success: false` in the stats, a warning surfaced in the frontend, and a summary that explicitly says the diff may not be reliable. The pipeline never silently produces a meaningless diff.
- No/failed LLM API → deterministic template summary; the pipeline still completes.

## Project layout

```
app/                 # pipeline stages (each independently testable)
  config.py          # all thresholds/tunables in one place
  ingest.py          # stage 1
  preprocess.py      # stage 2
  registration.py    # stage 3 (border-quad primary, ORB+RANSAC fallback)
  diff_engine.py     # stage 4 (linework XOR + SSIM photo mode)
  regions.py         # stage 5
  visualize.py       # stage 6
  stats.py           # stage 7
  summarizer.py      # stage 8 (LLM, with offline fallback)
  pipeline.py        # orchestrator used by both CLI and API
routers/compare.py   # stage 9: POST /compare
main.py              # FastAPI app entry point
frontend/streamlit_app.py  # stage 10: upload UI
cli.py               # command-line runner
samples/generate_samples.py  # synthetic demo drawings
tests/               # per-stage unit tests
```
