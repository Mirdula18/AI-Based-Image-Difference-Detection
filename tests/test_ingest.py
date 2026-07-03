import os

import cv2
import numpy as np
import pytest

from app.ingest import IngestionError, load_image


def _write_png(tmp_path, name="img.png", size=(100, 80)):
    path = os.path.join(tmp_path, name)
    img = np.full((size[1], size[0], 3), 200, dtype=np.uint8)
    cv2.imwrite(path, img)
    return path


def test_load_valid_png(tmp_path):
    path = _write_png(tmp_path)
    result = load_image(path)
    assert result.source_kind == "raster"
    assert result.image.shape[:2] == (80, 100)


def test_missing_file_raises(tmp_path):
    with pytest.raises(IngestionError):
        load_image(os.path.join(tmp_path, "does_not_exist.png"))


def test_unsupported_extension_raises(tmp_path):
    path = os.path.join(tmp_path, "file.txt")
    with open(path, "w") as f:
        f.write("not an image")
    with pytest.raises(IngestionError):
        load_image(path)


def test_corrupt_image_raises(tmp_path):
    path = os.path.join(tmp_path, "bad.png")
    with open(path, "wb") as f:
        f.write(b"not really a png")
    with pytest.raises(IngestionError):
        load_image(path)
