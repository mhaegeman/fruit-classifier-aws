"""Configuration loaded from environment variables."""

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    s3_data_path: str
    s3_results_path: str
    s3_report_path: str = ""
    pca_components: int = 8
    spark_master: str = "yarn"
    spark_app_name: str = "Fruit Images Preprocessing"
    image_batch_partitions: int = 20

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            s3_data_path=os.environ["S3_DATA_PATH"],
            s3_results_path=os.environ["S3_RESULTS_PATH"],
            s3_report_path=os.environ.get("S3_REPORT_PATH", ""),
            pca_components=int(os.environ.get("PCA_COMPONENTS", "8")),
            spark_master=os.environ.get("SPARK_MASTER", "yarn"),
            spark_app_name=os.environ.get("SPARK_APP_NAME", "Fruit Images Preprocessing"),
            image_batch_partitions=int(os.environ.get("IMAGE_BATCH_PARTITIONS", "20")),
        )
