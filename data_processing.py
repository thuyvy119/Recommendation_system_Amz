import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, DoubleType, ArrayType, StringType
from pyspark.sql.functions import col, dayofweek, month, hour, from_unixtime, udf
from pyspark.sql.types import ArrayType, DoubleType, FloatType
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import expr
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.linalg import Vectors, VectorUDT

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel
import torch

spark = (SparkSession.builder.master("local[*]").appName("AmazonReviews").getOrCreate())

# check spark session is correctly initialized
if spark is None:
    raise RuntimeError("SparkSession not initialized")

review_filepath = './data/Gift_Cards.jsonl'
metadata_filepath = './data/meta_Gift_Cards.jsonl'

reviews_df = spark.read.json(review_filepath)
metadata_df = spark.read.json(metadata_filepath)

reviews_df = reviews_df.repartition(10)
metadata_df = metadata_df.repartition(10)

print(reviews_df.select("parent_asin").distinct().count())
print(metadata_df.select("parent_asin").distinct().count())

# fill NaN values with 0
reviews_df = reviews_df.fillna(0)
metadata_df = metadata_df.fillna(0)

# reviews_df.printSchema()
# metadata_df.printSchema()

# cache
reviews_df.cache()
metadata_df.cache()

# convert timestamp from ms to s and then to timestamp type
reviews_df = reviews_df.withColumn("timestamp", from_unixtime(col("timestamp") / 1000).cast("timestamp"))

# extract temporal features
reviews_df = reviews_df.withColumn("day_of_week", dayofweek(col("timestamp")))
reviews_df = reviews_df.withColumn("month", month(col("timestamp")))
reviews_df = reviews_df.withColumn("hour", hour(col("timestamp")))

# drop video, img columns
metadata_df = metadata_df.drop('video', 'image')

# initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen1.5-0.5B')
model = AutoModel.from_pretrained('Qwen/Qwen1.5-0.5B')

def generate_embeddings_udf(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy().tolist()
    return embeddings

generate_embeddings_udf = udf(lambda x: generate_embeddings_udf(x), ArrayType(DoubleType()))

# generate embeddings for textual features
reviews_df = reviews_df.withColumn("text_embedding", generate_embeddings_udf(col("text")))
reviews_df = reviews_df.withColumn("title_embedding", generate_embeddings_udf(col("title")))

metadata_df = metadata_df.withColumn("features_embedding", generate_embeddings_udf(col("features")))
metadata_df = metadata_df.withColumn("description_embedding", generate_embeddings_udf(col("description")))
metadata_df = metadata_df.withColumn("details_embedding", generate_embeddings_udf(col("details")))

# encode user_id and asin
indexer_user = StringIndexer(inputCol="user_id", outputCol="user_id_encoded")
indexer_asin = StringIndexer(inputCol="asin", outputCol="asin_encoded")

reviews_df = indexer_user.fit(reviews_df).transform(reviews_df)
reviews_df = indexer_asin.fit(reviews_df).transform(reviews_df)

# merging metadata with reviews based on parent_asin
df = reviews_df.join(metadata_df, on='parent_asin', how='left')

# sentiment analysis
def sentiment_analysis(text):
    # Placeholder function for sentiment analysis
    # Implement actual sentiment analysis using a model or library
    return {'label': 'POSITIVE', 'score': 0.9}  # Example output

sentiment_analysis_udf = udf(lambda text: sentiment_analysis(text)['score'] if sentiment_analysis(text)['label'] == 'POSITIVE' else -sentiment_analysis(text)['score'], FloatType())

df = df.withColumn("sentiment_score", sentiment_analysis_udf(col("text")))

# compute similarity scores
def compute_similarity(embeddings):
    similarity_matrix = cosine_similarity(np.vstack(embeddings))
    return similarity_matrix.mean(axis=1).tolist()

compute_similarity_udf = udf(compute_similarity, ArrayType(FloatType()))

df = df.withColumn("similarity_score", compute_similarity_udf(col("text_embedding")))

# define UDF to convert array to vector
def array_to_vector(array):
    return Vectors.dense(array)

array_to_vector_udf = udf(array_to_vector, VectorUDT())

# convert array columns to vector columns
df = df.withColumn("text_embedding", array_to_vector_udf(col("text_embedding")))
df = df.withColumn("title_embedding", array_to_vector_udf(col("title_embedding")))
df = df.withColumn("features_embedding", array_to_vector_udf(col("features_embedding")))
df = df.withColumn("description_embedding", array_to_vector_udf(col("description_embedding")))
df = df.withColumn("details_embedding", array_to_vector_udf(col("details_embedding")))
df = df.withColumn("similarity_score", array_to_vector_udf(col("similarity_score")))

# combining features into a final feature set
features = ['user_id_encoded', 'asin_encoded', 'rating', 'verified_purchase', 
            'helpful_vote', 'day_of_week', 'month', 'hour', 'sentiment_score', 'similarity_score']
embeddings = ['text_embedding', 'title_embedding', 'features_embedding', 'description_embedding', 'details_embedding']

# split into features and targets
feature_columns = features + embeddings
target_column = 'rating'

# create features, target df
features_df = df.select(*feature_columns)
target_df = df.select(target_column)

data = features_df.withColumn("rating", target_df[target_column])

# assemble all feature columns into a single vector column
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data = assembler.transform(data)

# train a Random Forest model
rf = RandomForestClassifier(labelCol="rating", featuresCol="features", numTrees=100)
rf_model = rf.fit(data)

# get feature importances
importances = rf_model.featureImportances

# convert importances to a list of tuples (feature, importance)
feature_importances = [(feature, importance) for feature, importance in zip(feature_columns, importances.toArray())]

# sort the features by importance
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

# print the feature importances
for feature, importance in feature_importances:
    print(f"Feature: {feature}, Importance: {importance}")

# select the top N features
N = 10
selected_features = [feature for feature, importance in feature_importances[:N]]

# create a new DataFrame with only the selected features
selected_features_df = df.select(*selected_features)
selected_features_df.show()

# keep the target column
target_df = df.select(target_column)