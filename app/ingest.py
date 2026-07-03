"""Stage 1: Ingestion.

Accepts JPG/PNG/PDF paths (or raw bytes) and returns a validated BGR
numpy array (OpenCV convention) for each input. PDFs are rasterized with
PyMuPDF at a fixed DPI since they are assumed to be single-page technical
drawing sheets.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import fitz  # PyMuPDF
import numpy as np
import cv2

from app.config import PDF_RASTER_DPI, SUPPORTED_RASTER_EXTS, SUPPORTED_PDF_EXTS


class IngestionError(Exception):
    """Raised when a file cannot be validated or loaded as an image."""


@dataclass
class IngestedImage:
    image: np.ndarray  # BGR, uint8
    source_path: str
    source_kind: str  # "raster" or "pdf"


def _rasterize_pdf(path: str, dpi: int = PDF_RASTER_DPI) -> np.ndarray:
    """Render the first page of a PDF to a BGR numpy array at `dpi`."""
    try:
        doc = fitz.open(path)
    except Exception as exc:  # malformed / non-PDF file
        raise IngestionError(f"Could not open PDF '{path}': {exc}") from exc

    if doc.page_count < 1:
        doc.close()
        raise IngestionError(f"PDF '{path}' has no pages")

    page = doc.load_page(0)
    zoom = dpi / 72.0  # PDF base unit is 72 DPI
    matrix = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    doc.close()

    img = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )
    if pixmap.n == 3:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif pixmap.n == 4:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img_bgr


def load_image(path: str) -> IngestedImage:
    """Load a single input file (JPG/PNG/PDF) into a validated BGR array.

    Raises IngestionError on unsupported extensions, missing files, or
    decode failures so callers can report a clear error instead of
    crashing deep inside the CV pipeline.
    """
    if not os.path.isfile(path):
        raise IngestionError(f"File not found: '{path}'")

    ext = os.path.splitext(path)[1].lower()

    if ext in SUPPORTED_PDF_EXTS:
        image = _rasterize_pdf(path)
        kind = "pdf"
    elif ext in SUPPORTED_RASTER_EXTS:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        kind = "raster"
    else:
        raise IngestionError(
            f"Unsupported file type '{ext}' for '{path}'. "
            f"Supported: {sorted(SUPPORTED_RASTER_EXTS | SUPPORTED_PDF_EXTS)}"
        )

    if image is None or image.size == 0:
        raise IngestionError(f"Failed to decode image data from '{path}'")

    return IngestedImage(image=image, source_path=path, source_kind=kind)


def load_pair(path_a: str, path_b: str) -> tuple[IngestedImage, IngestedImage]:
    """Load and validate both input images for a comparison run."""
    return load_image(path_a), load_image(path_b)
