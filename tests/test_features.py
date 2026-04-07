"""Tests for image feature extraction utilities."""

import io

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow not installed")


def _make_jpeg_bytes(width: int = 224, height: int = 224) -> bytes:
    from PIL import Image

    img = Image.new("RGB", (width, height), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_preprocess_image_output_shape():
    from fruit_classifier.features import preprocess_image

    arr = preprocess_image(_make_jpeg_bytes())
    assert arr.shape == (224, 224, 3)


def test_preprocess_image_value_range():
    """MobileNetV2 preprocess_input scales pixel values to [-1, 1]."""
    from fruit_classifier.features import preprocess_image

    arr = preprocess_image(_make_jpeg_bytes())
    assert arr.min() >= -1.0
    assert arr.max() <= 1.0


def test_preprocess_image_resizes_input():
    from fruit_classifier.features import preprocess_image

    arr = preprocess_image(_make_jpeg_bytes(width=100, height=150))
    assert arr.shape == (224, 224, 3)


def test_featurize_series_output_length():
    from unittest.mock import MagicMock

    import pandas as pd

    from fruit_classifier.features import featurize_series

    mock_model = MagicMock()
    mock_model.predict.return_value = np.random.rand(3, 1280)

    series = pd.Series([_make_jpeg_bytes() for _ in range(3)])
    result = featurize_series(mock_model, series)

    assert len(result) == 3
    assert all(isinstance(v, np.ndarray) for v in result)
