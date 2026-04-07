# fruit-classifier-aws

Distributed fruit image classification pipeline running on AWS EMR.  
Images are read from S3, featurized with a pre-trained **MobileNetV2**, reduced with **PCA**, and written back to S3 as Parquet.

## Architecture

```
S3 (images)
    │
    ▼
Spark (binaryFile reader)
    │
    ▼
MobileNetV2 feature extraction  ← distributed via Scalar Iterator pandas UDF
    │
    ▼
PCA (k=8 components)            ← Spark MLlib
    │
    ▼
S3 (Parquet results)
```

## Project structure

```
fruit-classifier-aws/
├── src/fruit_classifier/
│   ├── config.py       # Environment-based configuration
│   ├── features.py     # MobileNetV2 feature extraction
│   ├── reduction.py    # PCA dimensionality reduction
│   ├── io.py           # S3 load / Parquet save / CSV conversion
│   └── pipeline.py     # Entry point
├── tests/
│   ├── test_config.py
│   ├── test_features.py
│   └── test_io.py
├── scripts/
│   └── bootstrap-emr.sh  # EMR node bootstrap
├── .github/workflows/
│   └── ci.yml
├── pyproject.toml
└── Makefile
```

## Configuration

All settings are passed via environment variables — no hardcoded paths.

| Variable | Required | Default | Description |
|---|---|---|---|
| `S3_DATA_PATH` | yes | — | S3 path to input images, e.g. `s3://fruit-data/Test` |
| `S3_RESULTS_PATH` | yes | — | S3 path for Parquet output |
| `S3_REPORT_PATH` | no | `""` | S3 path for PCA reports |
| `PCA_COMPONENTS` | no | `8` | Number of PCA components |
| `SPARK_MASTER` | no | `yarn` | Spark master URL |
| `IMAGE_BATCH_PARTITIONS` | no | `20` | Spark repartition count before featurization |

## Usage

### On AWS EMR

1. Upload the package to your cluster (or install via bootstrap).
2. Set the required environment variables.
3. Submit with `spark-submit`:

```bash
export S3_DATA_PATH=s3://fruit-data/Test
export S3_RESULTS_PATH=s3://fruit-data/Results

spark-submit \
  --master yarn \
  --py-files dist/fruit_classifier-0.1.0-py3-none-any.whl \
  -m fruit_classifier.pipeline
```

### Convert results to CSV

```python
from fruit_classifier.io import convert_parquet_to_csv
convert_parquet_to_csv("s3://fruit-data/Results", "s3://fruit-data/Results/csv")
```

## Development

```bash
# Install with dev extras
make install-dev

# Lint
make lint

# Run tests (TensorFlow / Spark tests skipped if deps absent)
make test
```

## AWS EMR Bootstrap

Upload `scripts/bootstrap-emr.sh` as the EMR bootstrap action to install dependencies on all nodes before the job starts.
