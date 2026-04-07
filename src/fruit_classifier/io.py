"""I/O utilities: loading images from S3 and writing/converting results."""

import glob
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def load_images(spark, path: str):
    """Load binary JPEG files from S3 into a Spark DataFrame.

    Adds a 'label' column derived from the parent directory name.
    """
    from pyspark.sql.functions import element_at, split

    logger.info("Loading images from %s", path)
    images = (
        spark.read.format("binaryFile")
        .option("pathGlobFilter", "*.jpg")
        .option("recursiveFileLookup", "true")
        .load(path)
    )
    return images.withColumn("label", element_at(split(images["path"], "/"), -2))


def save_parquet(df, path: str) -> None:
    """Persist a Spark DataFrame as Parquet (overwrite mode)."""
    logger.info("Writing Parquet to %s", path)
    df.write.mode("overwrite").parquet(path)


def convert_parquet_to_csv(parquet_dir: str, output_dir: str) -> None:
    """Convert every Parquet file in `parquet_dir` to CSV in `output_dir`."""
    logger.info("Converting Parquet files in %s → %s", parquet_dir, output_dir)
    for i, filename in enumerate(glob.iglob(f"{parquet_dir}/*.parquet")):
        logger.info("Converting %s", filename)
        df = pd.read_parquet(filename)
        out_path = f"{output_dir}/preprocessed_images_{i}.csv"
        df.to_csv(out_path, index=False)
        logger.info("Saved %s", out_path)
