"""PCA dimensionality reduction for extracted image features."""

import logging

from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql import DataFrame
from pyspark.sql.functions import udf

logger = logging.getLogger(__name__)


def perform_pca(features_df: DataFrame, n_components: int = 8) -> DataFrame:
    """Reduce the 'features' column to `n_components` principal components.

    Args:
        features_df: DataFrame with an array-typed 'features' column.
        n_components: Number of principal components to retain.

    Returns:
        DataFrame with an additional 'pca_features' column.
    """
    logger.info("Fitting PCA with %d components", n_components)
    to_vector = udf(lambda x: Vectors.dense(x), VectorUDT())
    vec_df = features_df.withColumn("features_vec", to_vector("features"))
    pca_model = PCA(k=n_components, inputCol="features_vec", outputCol="pca_features").fit(vec_df)
    return pca_model.transform(vec_df)
