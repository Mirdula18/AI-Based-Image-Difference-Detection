#!/usr/bin/env python
"""Generate two synthetic architectural-style drawing sheets for demo/testing.

Produces sample_a.png (baseline) and sample_b.png (revised), each with a
fixed drawing border/title block, some "wall" line-art, and a deliberate
difference (an added window + a moved door) between revisions. Sample B
is also rendered at a slightly different scale/offset than A to exercise
the registration stage, since the pipeline must not assume pixel-perfect
alignment between inputs.
"""
import os

import cv2
import numpy as np

CANVAS_W, CANVAS_H = 1600, 1200
BORDER_MARGIN = 60


def _draw_title_block(img: np.ndarray) -> None:
    cv2.rectangle(
        img,
        (BORDER_MARGIN, BORDER_MARGIN),
        (CANVAS_W - BORDER_MARGIN, CANVAS_H - BORDER_MARGIN),
        (0, 0, 0),
        4,
    )
    tb_h = 100
    cv2.rectangle(
        img,
        (CANVAS_W - BORDER_MARGIN - 400, CANVAS_H - BORDER_MARGIN - tb_h),
        (CANVAS_W - BORDER_MARGIN, CANVAS_H - BORDER_MARGIN),
        (0, 0, 0),
        2,
    )
    cv2.putText(
        img,
        "SHEET A-101 - ELEVATION",
        (CANVAS_W - BORDER_MARGIN - 380, CANVAS_H - BORDER_MARGIN - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def _draw_base_content(img: np.ndarray) -> None:
    # Outer wall footprint
    cv2.rectangle(img, (200, 200), (1350, 950), (0, 0, 0), 3)
    # Interior partition walls
    cv2.line(img, (700, 200), (700, 950), (0, 0, 0), 3)
    cv2.line(img, (200, 570), (700, 570), (0, 0, 0), 3)
    # A door opening (arc) on the partition wall - present in both versions
    cv2.ellipse(img, (700, 570), (80, 80), 0, 180, 270, (0, 0, 0), 2)
    # Windows on the outer wall (2 windows) - present in both versions
    for wx in (350, 950):
        cv2.rectangle(img, (wx, 195), (wx + 120, 205), (0, 0, 0), 2)
    # Room labels
    cv2.putText(img, "LIVING ROOM", (280, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "BEDROOM", (850, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)


def generate_sample_a() -> np.ndarray:
    img = np.full((CANVAS_H, CANVAS_W, 3), 255, dtype=np.uint8)
    _draw_title_block(img)
    _draw_base_content(img)
    return img


def generate_sample_b() -> np.ndarray:
    """Same drawing as A, but with an added window, a resized room, and a
    slight global scale + offset to simulate a non-pixel-aligned re-export.
    """
    img = np.full((CANVAS_H, CANVAS_W, 3), 255, dtype=np.uint8)
    _draw_title_block(img)
    _draw_base_content(img)

    # --- Deliberate differences vs sample A ---
    # 1. A new window added on the bottom wall (clearly new content)
    cv2.rectangle(img, (500, 945), (650, 955), (0, 0, 0), 2)
    # 2. An extra small closet partition added in the bedroom
    cv2.rectangle(img, (1150, 700), (1300, 850), (0, 0, 0), 3)
    cv2.putText(img, "CLOSET", (1160, 780), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

    # --- Simulate zoom/offset difference between the two exported files ---
    # Scale slightly DOWN so the drawing border stays fully in frame,
    # mimicking a re-export at a different zoom level.
    scale = 0.95
    tx, ty = 30, 22
    transform = np.array([[scale, 0, tx], [0, scale, ty]], dtype=np.float32)
    img = cv2.warpAffine(
        img, transform, (CANVAS_W, CANVAS_H), borderValue=(255, 255, 255)
    )
    return img


def main() -> None:
    out_dir = os.path.dirname(os.path.abspath(__file__))
    sample_a = generate_sample_a()
    sample_b = generate_sample_b()
    cv2.imwrite(os.path.join(out_dir, "sample_a.png"), sample_a)
    cv2.imwrite(os.path.join(out_dir, "sample_b.png"), sample_b)
    print(f"Wrote sample_a.png and sample_b.png to {out_dir}")


if __name__ == "__main__":
    main()
