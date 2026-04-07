#!/bin/bash
# Bootstrap script for AWS EMR cluster nodes.
# Installs all Python dependencies required by the fruit classifier pipeline.
set -euo pipefail

sudo python3 -m pip install -U setuptools pip wheel
sudo python3 -m pip install \
    "pillow" \
    "pandas>=1.3" \
    "tensorflow>=2.8" \
    "pyarrow>=6.0" \
    "boto3>=1.20" \
    "s3fs>=2022.1" \
    "fsspec>=2022.1" \
    "matplotlib>=3.4"
