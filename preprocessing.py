import pandas as pd
from PIL import Image
import numpy as np
import io

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import Model
from pyspark.sql.functions import col, pandas_udf, PandasUDFType, element_at, split, udf
from pyspark.sql import SparkSession
from pyspark.ml.feature import PCA, PCAModel
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import VectorAssembler
import matplotlib.pyplot as plt


# Set images path
PATH = 's3://fruit-data'
PATH_Data = PATH+'/Test'
PATH_Result = PATH+'/Results'
PATH_Report = PATH+'/PCA_Report'
print('PATH:        '+\
      PATH+'\nPATH_Data:   '+\
      PATH_Data+'\nPATH_Result: '+PATH_Result)

# Create spark session
spark = (SparkSession
             .builder
             .appName('Fruit Images Preprocessing')
             .master('local')
             .config("spark.sql.parquet.writeLegacyFormat", 'true')
             .getOrCreate()
)

sc = spark.sparkContext

# import images in spark object
images = spark.read.format("binaryFile") \
  .option("pathGlobFilter", "*.jpg") \
  .option("recursiveFileLookup", "true") \
  .load(PATH_Data)

images = images.withColumn('label', element_at(split(images['path'], '/'),-2))
print(images.printSchema())
print(images.select('path','label').show(5,False))

# Importing the model MobileNetV2
model = MobileNetV2(weights='imagenet',
                    include_top=True,
                    input_shape=(224, 224, 3))

new_model = Model(inputs=model.input,
                  outputs=model.layers[-2].output)

new_model.summary()

# saving model's weights in a variable for easy access
brodcast_weights = sc.broadcast(new_model.get_weights())

def model_fn():
    """
    Returns a MobileNetV2 model with top layer removed 
    and broadcasted pretrained weights.
    """
    model = MobileNetV2(weights='imagenet',
                        include_top=True,
                        input_shape=(224, 224, 3))
    for layer in model.layers:
        layer.trainable = False
    new_model = Model(inputs=model.input,
                  outputs=model.layers[-2].output)
    new_model.set_weights(brodcast_weights.value)
    return new_model


def preprocess(content):
    """
    Preprocesses raw image bytes for prediction.
    """
    img = Image.open(io.BytesIO(content)).resize([224, 224])
    arr = img_to_array(img)
    return preprocess_input(arr)

def featurize_series(model, content_series):
    """
    Featurize a pd.Series of raw images using the input model.
    :return: a pd.Series of image features
    """
    input = np.stack(content_series.map(preprocess))
    preds = model.predict(input)
    # For some layers, output features will be multi-dimensional tensors.
    # We flatten the feature tensors to vectors for easier storage in Spark DataFrames.
    output = [p.flatten() for p in preds]
    return pd.Series(output)

@pandas_udf('array<float>', PandasUDFType.SCALAR_ITER)
def featurize_udf(content_series_iter):
    '''
    This method is a Scalar Iterator pandas UDF wrapping our featurization function.
    The decorator specifies that this returns a Spark DataFrame column of type ArrayType(Float).

    :param content_series_iter: This argument is an iterator over batches of data, where each batch
                              is a pandas Series of image data.
    '''
    # With Scalar Iterator pandas UDFs, we can load the model once and then re-use it
    # for multiple data batches.  This amortizes the overhead of loading big models.
    model = model_fn()
    for content_series in content_series_iter:
        yield featurize_series(model, content_series)

# If images are too big avoid OOM with:
# spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1024")

#################################### PCA STUDY ########################################
# explained_variances=[]

# num_components = features_vec_df.select("features_vec").first()[0].size


# # Iterate through each component to calculate the explained variance
# for k in range(1, 11):
#     pca = PCA(k=k, inputCol="features_vec", outputCol="pca_features")
#     pca_model = pca.fit(features_vec_df)
#     explained_variances.append(sum(pca_model.explainedVariance))
    

# # Plot the elbow curve analysis
# plt.plot(range(1, k + 1), explained_variances, marker='o')
# plt.xlabel('Number of Principal Components')
# plt.ylabel('Cumulative Explained Variance')
# plt.title('Elbow Curve Analysis')
# plt.savefig('Elbow_Curve_Analysis.png')
# plt.show()

def perform_pca(features_df):

  to_vector = udf(lambda x: Vectors.dense(x), VectorUDT())
  features_vec_df = features_df.withColumn("features_vec", to_vector("features"))
  
  pca = PCA(k=8,
          inputCol="features_vec",
          outputCol="pca_features")

  pca_model = pca.fit(features_vec_df)

  res = pca_model.transform(features_vec_df)
  return res

# Apply functions
features_pca_df = perform_pca(images.repartition(20).select(col("path"),
                                            col("label"),
                                            featurize_udf("content").alias("features")
                                           ))

# Saving pca_df in Parquet format
features_pca_df.write.mode("overwrite").parquet(PATH_Result)
