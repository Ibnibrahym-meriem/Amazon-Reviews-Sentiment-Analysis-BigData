"""
================================================
dags/dag_model_retraining.py
VERSION FINALE — avec MLflow intégré
================================================
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────
MONGO_URI  = "mongodb://mongodb:27017/"
MONGO_DB   = "amazon_db"

COL_REVIEWS         = "reviews"
COL_PRODUCT_METRICS = "product_metrics"
COL_MODEL_METRICS   = "model_metrics"

PREPROCESSING_PATH = "/opt/models/preprocessing_pipeline"
MODEL_PATH         = "/opt/models/final_best_model"
CANDIDATE_PATH     = "/opt/models/candidate_model"

MLFLOW_URI         = "http://mlflow:5000"
MLFLOW_EXPERIMENT  = "Amazon_Sentiment_Retraining"

MIN_SAMPLES         = 300
PROMOTION_THRESHOLD = 0.005
NEG_THRESHOLD       = 0.35

PARQUET_RAW = "/tmp/airflow_raw_data"

default_args = {
    "owner":            "ml_engineer",
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
    "depends_on_past":  False,
    "email_on_failure": False,
}


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Extraction et préparation
# ════════════════════════════════════════════════════════════════════════
def task_extract_and_prepare(**context):
    from pymongo import MongoClient
    import pandas as pd
    import re, os

    client = MongoClient(MONGO_URI)
    db     = client[MONGO_DB]
    cursor = db[COL_REVIEWS].find(
        {"Score": {"$exists": True, "$ne": None}},
        {"Score": 1, "Summary": 1, "Text": 1, "_id": 0}
    ).limit(50000)

    docs = list(cursor)
    client.close()

    n = len(docs)
    logger.info(f"Task 1 : {n:,} reviews extraites")

    if n < MIN_SAMPLES:
        logger.warning(f"Task 1 : {n} samples < minimum {MIN_SAMPLES}. Annulé.")
        context["ti"].xcom_push(key="n_samples", value=0)
        return

    df = pd.DataFrame(docs).dropna(subset=["Score", "Text"])

    def score_to_label(score):
        try:
            s = int(score)
            return 0 if s < 3 else (1 if s == 3 else 2)
        except:
            return None

    df["label"] = df["Score"].apply(score_to_label)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    def clean_text(row):
        summary = str(row.get("Summary", "") or "")
        text    = str(row.get("Text",    "") or "")
        full    = (summary + " " + text).lower()
        full    = re.sub(r"<[^>]+>",            " ", full)
        full    = re.sub(r"https?://\S+|www\.\S+", " ", full)
        full    = re.sub(r"[^a-z\s]",           " ", full)
        full    = re.sub(r"\s+",                " ", full).strip()
        return full if len(full) > 10 else None

    df["cleaned_text"] = df.apply(clean_text, axis=1)
    df = df.dropna(subset=["cleaned_text"])

    os.makedirs(PARQUET_RAW, exist_ok=True)
    df[["cleaned_text", "label"]].to_parquet(f"{PARQUET_RAW}/data.parquet", index=False)

    context["ti"].xcom_push(key="n_samples", value=len(df))
    logger.info(f"Task 1 ✅ {len(df):,} reviews sauvegardées")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Réentraînement + MLflow
# ════════════════════════════════════════════════════════════════════════
def task_retrain_model(**context):
    import os, sys, time
    os.environ["PYSPARK_PYTHON"]        = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    n_samples = context["ti"].xcom_pull(key="n_samples", task_ids="extract_and_prepare_data")

    if not n_samples or n_samples < MIN_SAMPLES:
        logger.info("Task 2 : Pas assez de samples — ignorée.")
        for key in ["new_f1", "new_acc", "recall_neg", "recall_neu",
                    "f1_neg", "mlflow_run_id", "run_name", "n_train"]:
            context["ti"].xcom_push(key=key, value=None)
        return

    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, when
    from pyspark.ml import PipelineModel
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    import pandas as pd
    import mlflow

    spark = SparkSession.builder \
        .appName("Airflow_Retrain") \
        .master("local[2]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    try:
        pdf = pd.read_parquet(f"{PARQUET_RAW}/data.parquet")
        df  = spark.createDataFrame(pdf)

        preprocessing = PipelineModel.load(PREPROCESSING_PATH)
        transformed   = preprocessing.transform(df)

        N = float(transformed.count())
        K = 3.0
        counts = {
            float(r["label_idx"]): float(r["count"])
            for r in transformed.groupBy("label_idx").count().collect()
        }

        transformed = transformed.withColumn("classWeight",
            when(col("label_idx") == 0.0, N / (K * counts.get(0.0, 1)))
            .when(col("label_idx") == 1.0, N / (K * counts.get(1.0, 1)))
            .otherwise(                    N / (K * counts.get(2.0, 1)))
        )

        train_df, eval_df = transformed.randomSplit([0.85, 0.15], seed=42)
        train_df.cache()
        eval_df.cache()

        n_train = train_df.count()
        n_eval  = eval_df.count()

        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        run_name = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M')}"

        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id

            mlflow.log_param("algorithm",      "LogisticRegression")
            mlflow.log_param("maxIter",         60)
            mlflow.log_param("regParam",        0.05)
            mlflow.log_param("elasticNetParam", 0.0)
            mlflow.log_param("family",          "multinomial")
            mlflow.log_param("neg_threshold",   NEG_THRESHOLD)
            mlflow.log_param("n_training",      n_train)
            mlflow.log_param("n_eval",          n_eval)
            mlflow.log_param("class_weight",    "N/(K*count)")
            mlflow.log_param("features",        "TF-IDF unigrams+bigrams ~20K")

            lr = LogisticRegression(
                featuresCol     = "features",
                labelCol        = "label_idx",
                weightCol       = "classWeight",
                maxIter         = 60,
                regParam        = 0.05,
                elasticNetParam = 0.0,
                family          = "multinomial"
            )

            t0        = time.time()
            new_model = lr.fit(train_df)
            train_time = round(time.time() - t0, 1)
            mlflow.log_metric("train_time_seconds", train_time)

            eval_preds = new_model.transform(eval_df)

            new_f1  = MulticlassClassificationEvaluator(labelCol="label_idx", metricName="f1").evaluate(eval_preds)
            new_acc = MulticlassClassificationEvaluator(labelCol="label_idx", metricName="accuracy").evaluate(eval_preds)

            per_class = {}
            for idx in [0.0, 1.0, 2.0]:
                tp = eval_preds.filter((col("label_idx")==idx) & (col("prediction")==idx)).count()
                fp = eval_preds.filter((col("label_idx")!=idx) & (col("prediction")==idx)).count()
                fn = eval_preds.filter((col("label_idx")==idx) & (col("prediction")!=idx)).count()
                pr  = tp/(tp+fp) if (tp+fp) > 0 else 0
                rc  = tp/(tp+fn) if (tp+fn) > 0 else 0
                f1c = 2*pr*rc/(pr+rc) if (pr+rc) > 0 else 0
                per_class[idx] = {"precision": pr, "recall": rc, "f1": f1c}

            mlflow.log_metric("f1_macro",           round(new_f1, 4))
            mlflow.log_metric("accuracy",           round(new_acc, 4))
            mlflow.log_metric("f1_negative",        round(per_class[0.0]["f1"], 4))
            mlflow.log_metric("f1_neutral",         round(per_class[1.0]["f1"], 4))
            mlflow.log_metric("f1_positive",        round(per_class[2.0]["f1"], 4))
            mlflow.log_metric("recall_negative",    round(per_class[0.0]["recall"], 4))
            mlflow.log_metric("recall_neutral",     round(per_class[1.0]["recall"], 4))
            mlflow.log_metric("recall_positive",    round(per_class[2.0]["recall"], 4))
            mlflow.log_metric("precision_negative", round(per_class[0.0]["precision"], 4))

            new_model.write().overwrite().save(CANDIDATE_PATH)

            context["ti"].xcom_push(key="new_f1",       value=new_f1)
            context["ti"].xcom_push(key="new_acc",      value=new_acc)
            context["ti"].xcom_push(key="recall_neg",   value=per_class[0.0]["recall"])
            context["ti"].xcom_push(key="recall_neu",   value=per_class[1.0]["recall"])
            context["ti"].xcom_push(key="f1_neg",       value=per_class[0.0]["f1"])
            context["ti"].xcom_push(key="mlflow_run_id", value=run_id)
            context["ti"].xcom_push(key="run_name",     value=run_name)
            context["ti"].xcom_push(key="n_train",      value=int(N))

            logger.info(f"Task 2 ✅ F1={new_f1:.4f}  Acc={new_acc:.4f}  RunID={run_id}")

    finally:
        spark.stop()


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Évaluation et Promotion
# ════════════════════════════════════════════════════════════════════════
def task_evaluate_and_promote(**context):
    import os, sys
    from mlflow.tracking import MlflowClient
    os.environ["PYSPARK_PYTHON"]        = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    new_f1 = context["ti"].xcom_pull(key="new_f1",        task_ids="retrain_model")
    run_id = context["ti"].xcom_pull(key="mlflow_run_id", task_ids="retrain_model")

    if new_f1 is None:
        logger.info("Task 3 : Pas de modèle candidat — ignorée.")
        context["ti"].xcom_push(key="promoted", value=False)
        context["ti"].xcom_push(key="old_f1",   value=None)
        return

    from pyspark.ml.classification import LogisticRegressionModel
    import mlflow

    promoted = False
    old_f1   = 0.0

    try:
        # Récupérer le meilleur F1 parmi les runs promus
        mlflow.set_tracking_uri(MLFLOW_URI)
        client_mlflow = MlflowClient(tracking_uri=MLFLOW_URI)
        experiment    = client_mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)

        best_runs = client_mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.promoted = 'True'",
            order_by=["metrics.f1_macro DESC"],
            max_results=1
        )

        if best_runs:
            old_f1 = best_runs[0].data.metrics.get("f1_macro", 0.0)

        improvement = (new_f1 - old_f1) / old_f1 * 100 if old_f1 > 0 else 100

        logger.info(
            f"Task 3 : Production F1={old_f1:.4f}  "
            f"Candidat F1={new_f1:.4f}  "
            f"Amélioration={improvement:+.2f}%"
        )

        if new_f1 > old_f1 + PROMOTION_THRESHOLD:
            candidate = LogisticRegressionModel.load(CANDIDATE_PATH)
            candidate.write().overwrite().save(MODEL_PATH)
            promoted = True
            logger.info(f"Task 3 🚀 PROMU — F1: {old_f1:.4f} → {new_f1:.4f} ({improvement:+.2f}%)")
        else:
            logger.info(f"Task 3 ⏭️  REJETÉ — {improvement:+.2f}%")

    except Exception as e:
        logger.warning(f"Task 3 : Premier run ou erreur ({type(e).__name__}). Promotion automatique.")
        from pyspark.ml.classification import LogisticRegressionModel
        candidate = LogisticRegressionModel.load(CANDIDATE_PATH)
        candidate.write().overwrite().save(MODEL_PATH)
        promoted = True
        old_f1   = 0.0

    # Log MLflow — previous_f1 pour calcul d'amélioration côté dashboard
    with mlflow.start_run(run_id=run_id):
        mlflow.set_tag("promoted", str(promoted))
        mlflow.set_tag("decision", "PROMOTED" if promoted else "REJECTED")
        mlflow.log_metric("previous_f1", round(old_f1, 4))  # 0.0 = first run

    context["ti"].xcom_push(key="promoted", value=promoted)
    context["ti"].xcom_push(key="old_f1",   value=old_f1)


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Mise à jour MongoDB
# ════════════════════════════════════════════════════════════════════════
def task_update_model_metrics(**context):
    from pymongo import MongoClient

    ti         = context["ti"]
    new_f1     = ti.xcom_pull(key="new_f1",       task_ids="retrain_model")
    new_acc    = ti.xcom_pull(key="new_acc",       task_ids="retrain_model")
    recall_neg = ti.xcom_pull(key="recall_neg",    task_ids="retrain_model")
    recall_neu = ti.xcom_pull(key="recall_neu",    task_ids="retrain_model")
    f1_neg     = ti.xcom_pull(key="f1_neg",        task_ids="retrain_model")
    run_id     = ti.xcom_pull(key="mlflow_run_id", task_ids="retrain_model")
    run_name   = ti.xcom_pull(key="run_name",      task_ids="retrain_model")
    n_train    = ti.xcom_pull(key="n_train",       task_ids="retrain_model")
    n_samples  = ti.xcom_pull(key="n_samples",     task_ids="extract_and_prepare_data")
    promoted   = ti.xcom_pull(key="promoted",      task_ids="evaluate_and_promote")
    old_f1     = ti.xcom_pull(key="old_f1",        task_ids="evaluate_and_promote")

    if new_f1 is None:
        logger.info("Task 4 : Pas de métriques — ignorée.")
        return

    client = MongoClient(MONGO_URI)
    db     = client[MONGO_DB]

    try:
        if old_f1 and old_f1 > 0:
            improvement = round((new_f1 - old_f1) / old_f1 * 100, 2)
        else:
            improvement = None  # first run

        db[COL_MODEL_METRICS].insert_one({
            "type":               "retraining",
            "accuracy":           round(new_acc, 4)    if new_acc    else None,
            "f1_macro":           round(new_f1, 4),
            "total_processed":    n_samples,
            "recall_negative":    round(recall_neg, 4) if recall_neg else None,
            "recall_neutral":     round(recall_neu, 4) if recall_neu else None,
            "f1_negative":        round(f1_neg, 4)     if f1_neg     else None,
            "previous_f1":        round(old_f1, 4)     if old_f1     else None,
            "improvement_pct":    improvement,
            "n_training_samples": n_train,
            "promoted":           promoted,
            "mlflow_run_id":      run_id,
            "mlflow_url":         f"{MLFLOW_URI}/#/experiments/1/runs/{run_id}",
            "run_name":           run_name,
            "retrained_at":       datetime.now(),
        })

        logger.info("=" * 52)
        logger.info("  RÉSUMÉ RÉENTRAÎNEMENT")
        logger.info("=" * 52)
        logger.info(f"  Run name         : {run_name}")
        logger.info(f"  Samples          : {n_samples:,}")
        logger.info(f"  F1 nouveau       : {new_f1:.4f}")
        logger.info(f"  F1 ancien        : {old_f1:.4f}" if old_f1 else "  F1 ancien        : N/A (premier run)")
        if improvement is not None:
            logger.info(f"  Amélioration     : {improvement:+.2f}%")
        else:
            logger.info("  Amélioration     : Premier run")
        logger.info(f"  Promu            : {'✅ OUI' if promoted else '⏭️  NON'}")
        logger.info(f"  MLflow           : {MLFLOW_URI}/#/experiments/1/runs/{run_id}")
        logger.info("=" * 52)

    finally:
        client.close()


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Cleanup
# ════════════════════════════════════════════════════════════════════════
def task_cleanup(**context):
    import shutil
    try:
        shutil.rmtree(PARQUET_RAW, ignore_errors=True)
        logger.info(f"Task 5 ✅ Supprimé : {PARQUET_RAW}")
    except Exception as e:
        logger.warning(f"Task 5 : Cleanup échoué ({e})")


# ════════════════════════════════════════════════════════════════════════
# DAG
# ════════════════════════════════════════════════════════════════════════
with DAG(
    dag_id            = "daily_sentiment_retraining",
    default_args      = default_args,
    description       = "Réentraînement quotidien + MLflow tracking",
    schedule_interval = "0 3 * * *",
    start_date        = datetime(2026, 5, 1),
    catchup           = False,
    tags              = ["ml", "retraining", "mlflow", "sentiment"],
) as dag:

    t1 = PythonOperator(task_id="extract_and_prepare_data", python_callable=task_extract_and_prepare, provide_context=True)
    t2 = PythonOperator(task_id="retrain_model",            python_callable=task_retrain_model,        provide_context=True)
    t3 = PythonOperator(task_id="evaluate_and_promote",     python_callable=task_evaluate_and_promote, provide_context=True)
    t4 = PythonOperator(task_id="update_model_metrics",     python_callable=task_update_model_metrics, provide_context=True)
    t5 = PythonOperator(task_id="cleanup",                  python_callable=task_cleanup,              provide_context=True, trigger_rule="all_done")

    t1 >> t2 >> t3 >> t4 >> t5