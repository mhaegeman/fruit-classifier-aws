"""MobileNetV2-based feature extraction for fruit images."""

import io
import logging

import numpy as np
import pandas as pd
from PIL import Image
from pyspark.sql.functions import PandasUDFType, pandas_udf
from tensorflow.keras import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

logger = logging.getLogger(__name__)


def build_feature_extractor(weights: str = "imagenet") -> Model:
    """Build a MobileNetV2 feature extractor with the top classification layer removed."""
    base = MobileNetV2(weights=weights, include_top=True, input_shape=(224, 224, 3))
    for layer in base.layers:
        layer.trainable = False
    return Model(inputs=base.input, outputs=base.layers[-2].output)


def preprocess_image(content: bytes) -> np.ndarray:
    """Decode raw image bytes and prepare them for MobileNetV2 inference."""
    img = Image.open(io.BytesIO(content)).resize([224, 224])
    arr = img_to_array(img)
    return preprocess_input(arr)


def featurize_series(model: Model, content_series: pd.Series) -> pd.Series:
    """Run inference on a pandas Series of raw image bytes.

    Returns a Series of flattened feature vectors.
    """
    inputs = np.stack(content_series.map(preprocess_image))
    preds = model.predict(inputs)
    return pd.Series([p.flatten() for p in preds])


def make_featurize_udf(broadcast_weights):
    """Wrap feature extraction as a Spark Scalar Iterator pandas UDF.

    Using a Scalar Iterator UDF lets us load the model once per executor
    rather than once per batch, amortising the startup cost.
    """
    @pandas_udf("array<float>", PandasUDFType.SCALAR_ITER)
    def featurize_udf(content_series_iter):
        model = build_feature_extractor()
        model.set_weights(broadcast_weights.value)
        for content_series in content_series_iter:
            yield featurize_series(model, content_series)

    return featurize_udf
