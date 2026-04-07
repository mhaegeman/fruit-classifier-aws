"""Tests for configuration loading."""

import pytest

from fruit_classifier.config import Config


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("S3_DATA_PATH", "s3://bucket/data")
    monkeypatch.setenv("S3_RESULTS_PATH", "s3://bucket/results")
    config = Config.from_env()
    assert config.s3_data_path == "s3://bucket/data"
    assert config.s3_results_path == "s3://bucket/results"
    assert config.pca_components == 8
    assert config.spark_master == "yarn"


def test_config_custom_pca_components(monkeypatch):
    monkeypatch.setenv("S3_DATA_PATH", "s3://bucket/data")
    monkeypatch.setenv("S3_RESULTS_PATH", "s3://bucket/results")
    monkeypatch.setenv("PCA_COMPONENTS", "16")
    config = Config.from_env()
    assert config.pca_components == 16


def test_config_missing_required_env(monkeypatch):
    monkeypatch.delenv("S3_DATA_PATH", raising=False)
    monkeypatch.delenv("S3_RESULTS_PATH", raising=False)
    with pytest.raises(KeyError):
        Config.from_env()


def test_config_custom_spark_master(monkeypatch):
    monkeypatch.setenv("S3_DATA_PATH", "s3://bucket/data")
    monkeypatch.setenv("S3_RESULTS_PATH", "s3://bucket/results")
    monkeypatch.setenv("SPARK_MASTER", "local[4]")
    config = Config.from_env()
    assert config.spark_master == "local[4]"
