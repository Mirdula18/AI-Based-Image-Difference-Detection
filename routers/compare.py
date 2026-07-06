"""Stage 9: API layer -- POST /compare.

Accepts two uploaded files (JPG/PNG/PDF), runs the full pipeline, and
returns a single JSON response with URLs to every generated artifact
plus the structured stats and AI summary text.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import uuid

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.ingest import IngestionError
from app.pipeline import run_pipeline

router = APIRouter()

OUTPUT_ROOT = os.path.join(os.getcwd(), "static", "results")
os.makedirs(OUTPUT_ROOT, exist_ok=True)


def _save_upload(upload: UploadFile, dest_dir: str, name: str) -> str:
    ext = os.path.splitext(upload.filename or "")[1] or ".png"
    dest_path = os.path.join(dest_dir, f"{name}{ext}")
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest_path


@router.post("/compare")
async def compare(
    file_a: UploadFile = File(..., description="Original/older drawing (JPG/PNG/PDF)"),
    file_b: UploadFile = File(..., description="Revised/newer drawing (JPG/PNG/PDF)"),
    mode: str = Query("linework", pattern="^(linework|photo)$"),
):
    job_id = uuid.uuid4().hex[:12]
    output_dir = os.path.join(OUTPUT_ROOT, job_id)
    os.makedirs(output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        path_a = _save_upload(file_a, tmp_dir, "input_a")
        path_b = _save_upload(file_b, tmp_dir, "input_b")

        try:
            result = run_pipeline(path_a, path_b, output_dir, diff_mode=mode)
        except IngestionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if result.stats.get("status") == "aborted":
            raise HTTPException(
                status_code=400,
                detail="Comparison aborted. Alignment confidence too low. Please review inputs."
            )
        elif result.stats.get("status") == "unreliable":
            raise HTTPException(
                status_code=400,
                detail="Comparison quality insufficient. Too many candidate changes detected. Likely registration failure."
            )


    image_urls = {
        name: f"/static/results/{job_id}/{os.path.basename(path)}"
        for name, path in result.image_paths.items()
    }

    return {
        "job_id": job_id,
        "images": image_urls,
        "stats": result.stats,
        "summary": result.summary,
        "pdf_report_url": image_urls.get("pdf_report", ""),
    }

