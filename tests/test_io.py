"""Tests for I/O utilities."""

import os
import tempfile

import pandas as pd
import pytest
from fruit_classifier.io import convert_parquet_to_csv


def test_convert_parquet_to_csv_creates_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        df = pd.DataFrame({"label": ["apple", "banana"], "value": [1.0, 2.0]})
        df.to_parquet(os.path.join(tmpdir, "part-0.parquet"))

        out_dir = os.path.join(tmpdir, "csv")
        os.makedirs(out_dir)
        convert_parquet_to_csv(tmpdir, out_dir)

        csv_files = [f for f in os.listdir(out_dir) if f.endswith(".csv")]
        assert len(csv_files) == 1


def test_convert_parquet_to_csv_preserves_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        df = pd.DataFrame({"label": ["apple", "banana"], "value": [1.0, 2.0]})
        df.to_parquet(os.path.join(tmpdir, "part-0.parquet"))

        out_dir = os.path.join(tmpdir, "csv")
        os.makedirs(out_dir)
        convert_parquet_to_csv(tmpdir, out_dir)

        csv_path = os.path.join(out_dir, os.listdir(out_dir)[0])
        result = pd.read_csv(csv_path)
        assert list(result.columns) == ["label", "value"]
        assert len(result) == 2
        assert list(result["label"]) == ["apple", "banana"]


def test_convert_parquet_to_csv_multiple_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            pd.DataFrame({"x": [i]}).to_parquet(os.path.join(tmpdir, f"part-{i}.parquet"))

        out_dir = os.path.join(tmpdir, "csv")
        os.makedirs(out_dir)
        convert_parquet_to_csv(tmpdir, out_dir)

        csv_files = [f for f in os.listdir(out_dir) if f.endswith(".csv")]
        assert len(csv_files) == 3


def test_convert_parquet_to_csv_empty_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = os.path.join(tmpdir, "csv")
        os.makedirs(out_dir)
        convert_parquet_to_csv(tmpdir, out_dir)
        assert os.listdir(out_dir) == []
