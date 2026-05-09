import os
import sys

# ─────────────────────────────────────────────────────────────────
# Configuration Java
# (PYSPARK_PYTHON défini dans le Dockerfile — ne pas écraser ici)
# ─────────────────────────────────────────────────────────────────

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
    coalesce, lit, from_json, col,
    current_timestamp, when, udf,
    avg, count, sum as spark_sum, round as spark_round
)
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, LongType, FloatType
)
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel

# ─────────────────────────────────────────────────────────────────
# CORRECTION 1 : MongoDB URI → mongodb (nom service Docker)
#                au lieu de 127.0.0.1 / localhost qui ne fonctionne
#                pas depuis l'intérieur d'un conteneur Docker
# ─────────────────────────────────────────────────────────────────
MONGO_URI   = "mongodb://mongodb:27017"   # ← nom du service Docker
MONGO_DB    = "amazon_db"

# ─────────────────────────────────────────────────────────────────
# CORRECTION 2 : Kafka bootstrap → kafka:29092 (réseau interne Docker)
#                localhost:9092 n'est accessible que depuis l'hôte Windows
# ─────────────────────────────────────────────────────────────────
KAFKA_BOOTSTRAP = "kafka:29092"           # ← listener interne Docker

# ─────────────────────────────────────────────────────────────────
# CORRECTION 3 : Spark UI bindé sur 0.0.0.0 pour être visible
#                depuis l'hôte via http://localhost:4040
# ─────────────────────────────────────────────────────────────────
spark = SparkSession.builder \
    .appName("AmazonReviewsFinal") \
    .config("spark.sql.shuffle.partitions", "1") \
    .config("spark.driver.extraJavaOptions",   java_opts) \
    .config("spark.executor.extraJavaOptions", java_opts) \
    .config("spark.ui.host", "0.0.0.0") \
    .config("spark.ui.port", "4040") \
    .config("spark.jars.packages",
        "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,"
        "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0") \
    .config("spark.mongodb.write.connection.uri", f"{MONGO_URI}") \
    .config("spark.mongodb.read.connection.uri",  f"{MONGO_URI}") \
    .getOrCreate()

# Réduire les logs Spark (optionnel mais utile en prod)
spark.sparkContext.setLogLevel("WARN")

# Désactiver checksums (évite les ChecksumException sur Windows/Docker)
spark.sparkContext._jsc.hadoopConfiguration().set(
    "fs.file.impl.disable.cache", "true")
spark.sparkContext._jsc.hadoopConfiguration().set(
    "dfs.client.read.shortcircuit.skip.checksum", "true")

# ─────────────────────────────────────────────────────────────────
# Schéma Kafka
# ─────────────────────────────────────────────────────────────────
schema = StructType([
    StructField("Id",                     IntegerType()),
    StructField("ProductId",              StringType()),
    StructField("UserId",                 StringType()),
    StructField("ProfileName",            StringType()),
    StructField("HelpfulnessNumerator",   IntegerType()),
    StructField("HelpfulnessDenominator", IntegerType()),
    StructField("Score",                  IntegerType()),
    StructField("Time",                   LongType()),
    StructField("Summary",                StringType()),
    StructField("Text",                   StringType()),
])

# ─────────────────────────────────────────────────────────────────
# Chargement modèles ML
# ─────────────────────────────────────────────────────────────────
preprocessing_pipeline = PipelineModel.load("models/preprocessing_pipeline")
model                  = LogisticRegressionModel.load("models/final_best_model")

# ─────────────────────────────────────────────────────────────────
# Nettoyage texte — identique au Notebook 02 (NE PAS MODIFIER)
# ─────────────────────────────────────────────────────────────────
def apply_cleaning(df):
    df = df.withColumn("full_text",
        concat_ws(" ",
                  coalesce(col("Summary"), lit("")),
                  coalesce(col("Text"),    lit(""))))
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

# ─────────────────────────────────────────────────────────────────
# Lecture Kafka — CORRECTION : kafka:29092 au lieu de localhost:9092
# ─────────────────────────────────────────────────────────────────
df_raw = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP) \
    .option("subscribe", "amazon_reviews") \
    .option("startingOffsets", "latest") \
    .option("failOnDataLoss", "false") \
    .load()

# ─────────────────────────────────────────────────────────────────
# Parsing + nettoyage + prédiction ML
# ─────────────────────────────────────────────────────────────────
df_parsed = df_raw.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*")

df_parsed       = apply_cleaning(df_parsed)
df_preprocessed = preprocessing_pipeline.transform(df_parsed)
df_predictions  = model.transform(df_preprocessed)

extract_prob = udf(lambda v: float(v.toArray().max()), FloatType())

final_df = df_predictions \
    .withColumn("predicted_sentiment",
        when(col("prediction") == 0, "negative")
        .when(col("prediction") == 1, "neutral")
        .otherwise("positive")) \
    .withColumn("confidence",   extract_prob(col("probability"))) \
    .withColumn("processed_at", current_timestamp()) \
    .select(
        "Id", "ProductId", "UserId", "ProfileName",
        "HelpfulnessNumerator", "HelpfulnessDenominator",
        "Score", "Time", "Summary", "Text",
        "predicted_sentiment", "confidence", "processed_at"
    )

# Réduire à 1 partition — moins de surcharge Spark pour petits batches
final_df = final_df.coalesce(1)

# ─────────────────────────────────────────────────────────────────
# foreachBatch — écriture MongoDB
# CORRECTION : utiliser MONGO_URI (mongodb://mongodb:27017)
#              partout au lieu de l'URI hardcodée localhost
# ─────────────────────────────────────────────────────────────────
def process_batch(batch_df, batch_id):
    total = batch_df.count()
    if total == 0:
        print(f"[BATCH {batch_id}] Batch vide, skip")
        return

    # ── Collection 1 : reviews ──────────────────────────────────
    batch_df.write \
        .format("mongodb") \
        .option("uri",        MONGO_URI) \
        .option("database",   MONGO_DB) \
        .option("collection", "reviews") \
        .mode("append") \
        .save()
    print(f"[BATCH {batch_id}] ✅ reviews stockés ({total} docs)")  # total déjà calculé

    # ── Collection 2 : product_metrics ──────────────────────────
    product_metrics = batch_df.groupBy("ProductId").agg(
        count("*").alias("review_count"),
        spark_round(avg(
            when(col("predicted_sentiment") == "positive", 2)
            .when(col("predicted_sentiment") == "neutral",  1)
            .otherwise(0)
        ), 2).alias("avg_sentiment"),
        spark_round(
            spark_sum(when(col("predicted_sentiment") == "positive", 1).otherwise(0)) / count("*"), 2
        ).alias("positive_ratio"),
        spark_round(
            spark_sum(when(col("predicted_sentiment") == "negative", 1).otherwise(0)) / count("*"), 2
        ).alias("negative_ratio"),
        spark_round(
            spark_sum(when(col("predicted_sentiment") == "neutral",  1).otherwise(0)) / count("*"), 2
        ).alias("neutral_ratio"),
        current_timestamp().alias("last_updated")
    ).withColumnRenamed("ProductId", "_id")

    product_metrics.write \
        .format("mongodb") \
        .option("uri",        MONGO_URI) \
        .option("database",   MONGO_DB) \
        .option("collection", "product_metrics") \
        .mode("append") \
        .save()
    print(f"[BATCH {batch_id}] ✅ product_metrics mis à jour")

    # ── Collection 3 : model_metrics ────────────────────────────
    # total déjà calculé en haut du batch — pas de double count()
    correct = batch_df.filter(
        ((col("predicted_sentiment") == "positive") & (col("Score") >= 4)) |
        ((col("predicted_sentiment") == "negative") & (col("Score") <= 2)) |
        ((col("predicted_sentiment") == "neutral")  & (col("Score") == 3))
    ).count()

    accuracy = round(correct / total, 4) if total > 0 else 0.0

    model_metrics_df = batch_df.sparkSession.createDataFrame([{
        "accuracy":           accuracy,
        "total_processed":    total,
        "correct_predictions": correct,
        "model_path":         "models/final_best_model"
    }])

    model_metrics_df.write \
        .format("mongodb") \
        .option("uri",        MONGO_URI) \
        .option("database",   MONGO_DB) \
        .option("collection", "model_metrics") \
        .mode("append") \
        .save()
    print(f"[BATCH {batch_id}] ✅ model_metrics — accuracy: {accuracy} ({correct}/{total})")

# ─────────────────────────────────────────────────────────────────
# Lancement du stream
# ─────────────────────────────────────────────────────────────────
query = final_df.writeStream \
    .foreachBatch(process_batch) \
    .option("checkpointLocation", "./checkpoint_final") \
    .outputMode("append") \
    .trigger(processingTime="1 seconds") \
    .start()

print("✅ Spark Streaming démarré — en attente de messages Kafka...")
print(f"   Kafka  : {KAFKA_BOOTSTRAP}")
print(f"   MongoDB: {MONGO_URI}/{MONGO_DB}")
print(f"   Spark UI: http://localhost:4040")

query.awaitTermination()