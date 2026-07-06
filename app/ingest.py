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
    page_id: str = "Page 1"
    dpi: int = PDF_RASTER_DPI
    page_width: float = 0.0  # In PDF points (or pixels for rasters)
    page_height: float = 0.0 # In PDF points (or pixels for rasters)
    scale_factor: float = 1.0


def _rasterize_pdf(path: str, dpi: int = PDF_RASTER_DPI) -> np.ndarray:
    """Render the first page of a PDF to a BGR numpy array at `dpi`."""
    pages = _rasterize_pdf_pages(path, dpi)
    return pages[0].image


def _rasterize_pdf_pages(path: str, dpi: int = PDF_RASTER_DPI) -> list[IngestedImage]:
    """Render all pages of a PDF to a list of IngestedImage objects."""
    try:
        doc = fitz.open(path)
    except Exception as exc:  # malformed / non-PDF file
        raise IngestionError(f"Could not open PDF '{path}': {exc}") from exc

    if doc.page_count < 1:
        doc.close()
        raise IngestionError(f"PDF '{path}' has no pages")

    pages = []
    zoom = dpi / 72.0  # PDF base unit is 72 DPI
    matrix = fitz.Matrix(zoom, zoom)

    for idx in range(doc.page_count):
        page = doc.load_page(idx)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        img = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, pixmap.n
        )
        if pixmap.n == 3:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif pixmap.n == 4:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        w_pts = page.rect.width
        h_pts = page.rect.height

        pages.append(
            IngestedImage(
                image=img_bgr,
                source_path=path,
                source_kind="pdf",
                page_id=f"Page {idx + 1}",
                dpi=dpi,
                page_width=w_pts,
                page_height=h_pts,
                scale_factor=zoom,
            )
        )
    doc.close()
    return pages


def load_image_pages(path: str) -> list[IngestedImage]:
    """Load a file and return a list of IngestedImage objects for each page."""
    if not os.path.isfile(path):
        raise IngestionError(f"File not found: '{path}'")

    ext = os.path.splitext(path)[1].lower()

    if ext in SUPPORTED_PDF_EXTS:
        return _rasterize_pdf_pages(path)
    elif ext in SUPPORTED_RASTER_EXTS:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise IngestionError(f"Failed to decode image data from '{path}'")
        h, w = image.shape[:2]
        return [
            IngestedImage(
                image=image,
                source_path=path,
                source_kind="raster",
                page_id="Page 1",
                dpi=300,
                page_width=float(w),
                page_height=float(h),
                scale_factor=1.0,
            )
        ]
    else:
        raise IngestionError(
            f"Unsupported file type '{ext}' for '{path}'. "
            f"Supported: {sorted(SUPPORTED_RASTER_EXTS | SUPPORTED_PDF_EXTS)}"
        )


def load_image(path: str) -> IngestedImage:
    """Load a single input file (JPG/PNG/PDF) into a validated BGR array.

    Raises IngestionError on unsupported extensions, missing files, or
    decode failures so callers can report a clear error instead of
    crashing deep inside the CV pipeline.
    """
    pages = load_image_pages(path)
    return pages[0]


def load_pair(path_a: str, path_b: str) -> tuple[IngestedImage, IngestedImage]:
    """Load and validate both input images for a comparison run."""
    return load_image(path_a), load_image(path_b)


def load_pair_pages(path_a: str, path_b: str) -> tuple[list[IngestedImage], list[IngestedImage]]:
    """Load all pages for both files."""
    return load_image_pages(path_a), load_image_pages(path_b)

