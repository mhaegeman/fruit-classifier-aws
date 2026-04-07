# Fruit Classifier — AWS EMR Pipeline

A scalable pipeline that turns a folder of fruit photos into compact, machine-learning-ready feature vectors — ready to feed into any classifier (k-NN, SVM, neural net, etc.).

It runs on **AWS EMR** (Elastic MapReduce), so it can process thousands of images in parallel without running out of memory on a single machine.

---

## What does it do?

Given a bucket of labelled fruit images stored in S3, the pipeline:

1. **Reads every image** from S3 in parallel across a Spark cluster.
2. **Extracts rich visual features** using MobileNetV2, a lightweight deep-learning model pre-trained on 1.4 million images (ImageNet). Each image becomes a 1 280-dimensional feature vector that captures shape, colour, and texture.
3. **Compresses the features** from 1 280 down to just 8 numbers using PCA (Principal Component Analysis), retaining ~92 % of the information. This makes downstream training fast and avoids the curse of dimensionality.
4. **Writes the result** as a Parquet file back to S3, one row per image, ready for model training or exploration.

### Output

Each output row contains:

| Column | Type | Description |
|---|---|---|
| `path` | string | Original S3 path of the image |
| `label` | string | Fruit category (inferred from folder name, e.g. `Apple Braeburn`) |
| `features` | array\<float\> | Raw MobileNetV2 feature vector (1 280 dims) |
| `features_vec` | vector | Spark MLlib vector form of `features` |
| `pca_features` | vector | Compressed 8-component PCA representation |

The `pca_features` column is what you feed into a classifier. The other columns are kept for traceability and debugging.

---

## Why 8 PCA components?

The chart below shows how much information is retained as we increase the number of PCA components. The curve flattens sharply after component 4–5, and by component 8 we capture **~92 % of the cumulative explained variance** — a good trade-off between compactness and information loss.

![Elbow Curve Analysis](images/Elbow_Curve_Analysis.png)

---

## Architecture

```
S3 (fruit images, organised by label folder)
    │
    ▼
Spark binaryFile reader
    │  reads JPEGs in parallel across EMR worker nodes
    ▼
MobileNetV2 feature extraction
    │  Scalar Iterator pandas UDF — model loaded once per executor
    │  output: 1 280-dim float vector per image
    ▼
PCA  (k = 8, via Spark MLlib)
    │  output: 8-dim compressed vector per image
    ▼
S3 (Parquet — overwrite mode)
```

### Technology choices

| Component | Why |
|---|---|
| **AWS EMR + Spark** | Distributes image loading and inference across a cluster — scales to millions of images |
| **MobileNetV2** | Lightweight CNN (3.4 M params) that punches above its weight; runs fast on CPU workers |
| **Scalar Iterator UDF** | Loads the model once per Spark executor rather than once per batch — much lower overhead |
| **PCA via Spark MLlib** | Reduces 1 280-dim vectors to 8 dims in a distributed, memory-efficient way |
| **Parquet output** | Columnar format — fast to read, compressed by default, integrates with any ML framework |

---

## Project structure

```
fruit-classifier-aws/
├── src/fruit_classifier/
│   ├── config.py       # All settings loaded from environment variables
│   ├── features.py     # MobileNetV2 model + pandas UDF
│   ├── reduction.py    # PCA dimensionality reduction
│   ├── io.py           # S3 image loading, Parquet save, CSV export
│   └── pipeline.py     # Entry point — wires everything together
├── tests/
│   ├── test_config.py
│   ├── test_features.py
│   └── test_io.py
├── scripts/
│   └── bootstrap-emr.sh  # Installs Python deps on EMR nodes at startup
├── .github/workflows/
│   └── ci.yml            # Lint + test on every push
├── pyproject.toml
└── Makefile
```

---

## Quick start on AWS EMR

### 1. Create an EMR cluster

Create an EMR cluster (EMR 6.x, Spark 3.x) and attach `scripts/bootstrap-emr.sh` as the **bootstrap action**. This installs all Python dependencies on every node before the job starts.

### 2. Upload your images to S3

Images must be organised in per-label sub-folders:

```
s3://your-bucket/images/
    Apple Braeburn/
        img_001.jpg
        img_002.jpg
    Banana/
        img_001.jpg
    ...
```

### 3. Run the pipeline

SSH into the master node (or use EMR Steps) and submit the job:

```bash
export S3_DATA_PATH=s3://your-bucket/images
export S3_RESULTS_PATH=s3://your-bucket/results

spark-submit \
  --master yarn \
  --py-files dist/fruit_classifier-0.1.0-py3-none-any.whl \
  -m fruit_classifier.pipeline
```

### 4. Export results to CSV (optional)

```python
from fruit_classifier.io import convert_parquet_to_csv

convert_parquet_to_csv("s3://your-bucket/results", "s3://your-bucket/results/csv")
```

---

## Configuration

All settings are passed via environment variables — no hardcoded paths anywhere in the code.

| Variable | Required | Default | Description |
|---|---|---|---|
| `S3_DATA_PATH` | yes | — | S3 path to the input image folder |
| `S3_RESULTS_PATH` | yes | — | S3 path where Parquet output is written |
| `S3_REPORT_PATH` | no | `""` | S3 path for optional PCA variance reports |
| `PCA_COMPONENTS` | no | `8` | Number of PCA components to retain |
| `SPARK_MASTER` | no | `yarn` | Spark master URL (`yarn` for EMR, `local[*]` for local testing) |
| `IMAGE_BATCH_PARTITIONS` | no | `20` | Number of Spark partitions for the image processing stage |

---

## Development

```bash
# Install package + dev tools (pytest, ruff)
make install-dev

# Lint
make lint

# Run tests
# Tests that need TensorFlow or Spark are skipped automatically if those
# packages are not installed locally.
make test
```
