import os
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

java_opts = " ".join([
    "--add-opens=java.base/javax.security.auth=ALL-UNNAMED",
    "--add-opens=java.base/java.lang=ALL-UNNAMED",
    "--add-opens=java.base/java.net=ALL-UNNAMED",
    "--add-opens=java.base/java.util=ALL-UNNAMED",
    "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
    "--add-opens=java.base/java.text=ALL-UNNAMED",
    "--add-opens=java.sql/java.sql=ALL-UNNAMED",
])

os.environ['JAVA_TOOL_OPTIONS'] = java_opts

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    lower, regexp_replace, concat_ws,
    coalesce, lit, from_json, col, current_timestamp, when, udf
)
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType, FloatType
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel
import pyspark.sql.functions as F

def apply_cleaning(df):
    """
    Nettoyage IDENTIQUE à Notebook 02 — NE PAS MODIFIER.

    Le pipeline TF-IDF (preprocessing_pipeline) a été fitté
    sur ce format exact. Toute divergence dégrade les prédictions
    car les tokens ne correspondent plus au vocabulaire appris.

    Étapes (dans l'ordre obligatoire) :
    1. Combiner Summary + Text  ← crucial, le pipeline a été fitté sur les deux
    2. Lowercase
    3. Supprimer balises HTML
    4. Supprimer URLs
    5. Supprimer caractères non-alpha
    6. Normaliser les espaces
    """
    # Étape 1 : Combiner Summary + Text
    # Ton code utilisait Text seul — le pipeline attend les deux colonnes
    df = df.withColumn("full_text",
        concat_ws(" ",
                  coalesce(col("Summary"), lit("")),
                  coalesce(col("Text"),    lit(""))))

    # Étapes 2-6 : Nettoyage identique à Notebook 02
    df = df.withColumn("cleaned_text", lower(col("full_text")))
    df = df.withColumn("cleaned_text",
        regexp_replace(col("cleaned_text"), r"<[^>]+>", " "))
    df = df.withColumn("cleaned_text",
        regexp_replace(col("cleaned_text"), r"https?://\S+|www\.\S+", " "))
    df = df.withColumn("cleaned_text",
        regexp_replace(col("cleaned_text"), r"[^a-z\s]", " "))
    df = df.withColumn("cleaned_text",
        regexp_replace(col("cleaned_text"), r"\s+", " "))

    return df

spark = SparkSession.builder \
    .appName("AmazonReviewsFinal") \
    .config("spark.driver.extraJavaOptions", java_opts) \
    .config("spark.executor.extraJavaOptions", java_opts) \
    .config("spark.jars.packages",
        "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,"
        "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0") \
    .config("spark.mongodb.output.uri", "mongodb://127.0.0.1:27017/amazon_db.reviews") \
    .getOrCreate()

# Désactiver checksums
spark.sparkContext._jsc.hadoopConfiguration().set("fs.file.impl.disable.cache", "true")
spark.sparkContext._jsc.hadoopConfiguration().set("dfs.client.read.shortcircuit.skip.checksum", "true")

# Schéma Kafka
schema = StructType([
    StructField("Id", IntegerType()),
    StructField("ProductId", StringType()),
    StructField("UserId", StringType()),
    StructField("ProfileName", StringType()),
    StructField("HelpfulnessNumerator", IntegerType()),
    StructField("HelpfulnessDenominator", IntegerType()),
    StructField("Score", IntegerType()),
    StructField("Time", LongType()),
    StructField("Summary", StringType()),
    StructField("Text", StringType())
])

# Chargement modèles
preprocessing_pipeline = PipelineModel.load("models/preprocessing_pipeline")
model = LogisticRegressionModel.load("models/final_best_model")

# Lecture Kafka
df_raw = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "amazon_reviews") \
    .load()

# Parsing + Appel fonction nettoyage texte
df_parsed = df_raw.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*")

df_parsed = apply_cleaning(df_parsed) 
# Preprocessing + prédiction
df_preprocessed = preprocessing_pipeline.transform(df_parsed)
df_predictions = model.transform(df_preprocessed)

# Extraction confiance
extract_prob = udf(lambda v: float(v.toArray().max()), FloatType())

final_df = df_predictions.withColumn("predicted_sentiment",
    when(col("prediction") == 0, "negative")
    .when(col("prediction") == 1, "neutral")
    .otherwise("positive")
).withColumn("confidence", extract_prob(col("probability"))) \
 .withColumn("processed_at", current_timestamp()) \
 .select(
    "Id", "ProductId", "UserId", "ProfileName",
    "HelpfulnessNumerator", "HelpfulnessDenominator",
    "Score", "Time", "Summary", "Text",
    "predicted_sentiment", "confidence", "processed_at"
)

# ============================================
# 8. Un seul foreachBatch qui gère tout
# ============================================
from pyspark.sql.functions import avg, count, sum as spark_sum, when, round as spark_round

def process_batch(batch_df, batch_id):
    if batch_df.count() == 0:
        print(f"[BATCH {batch_id}] Batch vide, skip")
        return

    # --- Collection 1 : reviews ---
    batch_df.write \
        .format("mongodb") \
        .option("database", "amazon_db") \
        .option("collection", "reviews") \
        .mode("append") \
        .save()
    print(f"[BATCH {batch_id}] ✅ reviews stockés ({batch_df.count()} docs)")

    # --- Collection 2 : product_metrics ---
    product_metrics = batch_df.groupBy("ProductId").agg(
        count("*").alias("review_count"),
        spark_round(avg(
            when(col("predicted_sentiment") == "positive", 2)
            .when(col("predicted_sentiment") == "neutral", 1)
            .otherwise(0)
        ), 2).alias("avg_sentiment"),
        spark_round(
            spark_sum(when(col("predicted_sentiment") == "positive", 1).otherwise(0)) / count("*"), 2
        ).alias("positive_ratio"),
        spark_round(
            spark_sum(when(col("predicted_sentiment") == "negative", 1).otherwise(0)) / count("*"), 2
        ).alias("negative_ratio"),
        spark_round(
            spark_sum(when(col("predicted_sentiment") == "neutral", 1).otherwise(0)) / count("*"), 2
        ).alias("neutral_ratio"),
        current_timestamp().alias("last_updated")
    ).withColumnRenamed("ProductId", "_id")

    product_metrics.write \
        .format("mongodb") \
        .option("database", "amazon_db") \
        .option("collection", "product_metrics") \
        .mode("append") \
        .save()
    print(f"[BATCH {batch_id}] ✅ product_metrics mis à jour")

    # --- Collection 3 : model_metrics ---
    total = batch_df.count()
    correct = batch_df.filter(
        ((col("predicted_sentiment") == "positive") & (col("Score") >= 4)) |
        ((col("predicted_sentiment") == "negative") & (col("Score") <= 2)) |
        ((col("predicted_sentiment") == "neutral")  & (col("Score") == 3))
    ).count()

    accuracy = round(correct / total, 4) if total > 0 else 0.0

    model_metrics_data = [{"accuracy": accuracy, "total_processed": total,
                           "correct_predictions": correct,
                           "model_path": "models/final_best_model"}]
    model_metrics_df = batch_df.sparkSession.createDataFrame(model_metrics_data)

    model_metrics_df.write \
        .format("mongodb") \
        .option("database", "amazon_db") \
        .option("collection", "model_metrics") \
        .mode("append") \
        .save()
    print(f"[BATCH {batch_id}] ✅ model_metrics — accuracy: {accuracy}")

# ============================================
# 9. Lancer UN SEUL stream
# ============================================
query = final_df.writeStream \
    .foreachBatch(process_batch) \
    .option("checkpointLocation", "./checkpoint_final") \
    .outputMode("append") \
    .trigger(processingTime="5 seconds") \
    .start()

query.awaitTermination()
