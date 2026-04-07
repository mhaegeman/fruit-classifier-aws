"""Entry point for the fruit classifier preprocessing pipeline."""

import logging

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from fruit_classifier import io as fruit_io
from fruit_classifier.config import Config
from fruit_classifier.features import build_feature_extractor, make_featurize_udf
from fruit_classifier.reduction import perform_pca

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def build_spark_session(app_name: str, master: str) -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.sql.parquet.writeLegacyFormat", "true")
        .getOrCreate()
    )


def run(config: Config) -> None:
    """Execute the full preprocessing pipeline."""
    logger.info("Starting fruit classifier pipeline")
    spark = build_spark_session(config.spark_app_name, config.spark_master)
    sc = spark.sparkContext

    images = fruit_io.load_images(spark, config.s3_data_path)
    logger.info("Schema: %s", images.schema)

    extractor = build_feature_extractor()
    broadcast_weights = sc.broadcast(extractor.get_weights())
    featurize_udf = make_featurize_udf(broadcast_weights)

    features_df = (
        images.repartition(config.image_batch_partitions)
        .select(col("path"), col("label"), featurize_udf("content").alias("features"))
    )

    result_df = perform_pca(features_df, n_components=config.pca_components)
    fruit_io.save_parquet(result_df, config.s3_results_path)
    logger.info("Pipeline completed successfully")


def main() -> None:
    config = Config.from_env()
    run(config)


if __name__ == "__main__":
    main()
