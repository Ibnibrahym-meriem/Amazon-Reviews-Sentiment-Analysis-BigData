"""
================================================
dags/dag_model_retraining.py
VERSION FINALE — avec MLflow intégré
================================================

MLflow :
  - Tracking URI  : http://mlflow:5000  (service Docker)
  - Expérience    : Amazon_Sentiment_Retraining
  - Chaque run logge : params, metrics (F1/acc/recall), tag promoted
  - UI accessible : http://localhost:5000

DAG Flow :
  extract_and_prepare_data
          │
          ▼
  retrain_model  ──────────────────► MLflow run créé + métriques loggées
          │
          ▼
  evaluate_and_promote  ───────────► MLflow tag promoted=True/False
          │
          ▼
  update_model_metrics  ───────────► MongoDB model_metrics + lien MLflow
          │
          ▼
  cleanup  (trigger_rule=all_done)
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────

MONGO_URI  = "mongodb://mongodb:27017/"
MONGO_DB   = "amazon_db"

# Collections (celles que Spark écrit)
COL_REVIEWS         = "reviews"
COL_PRODUCT_METRICS = "product_metrics"
COL_MODEL_METRICS   = "model_metrics"

# Chemins modèles — volume ./models:/opt/models dans docker-compose
PREPROCESSING_PATH = "/opt/models/preprocessing_pipeline"
MODEL_PATH         = "/opt/models/final_best_model"
CANDIDATE_PATH     = "/opt/models/candidate_model"

# MLflow — service Docker
MLFLOW_URI         = "http://mlflow:5000"
MLFLOW_EXPERIMENT  = "Amazon_Sentiment_Retraining"

# Seuils
MIN_SAMPLES         = 300
PROMOTION_THRESHOLD = 0.015   # +1.5% F1 pour promouvoir
NEG_THRESHOLD       = 0.35    # seuil Négatif (décision business NB03)

# Fichiers temporaires dans le conteneur Airflow
PARQUET_RAW = "/tmp/airflow_raw_data"

default_args = {
    "owner":            "ml_engineer",
    "retries":          1,
    "retry_delay":      timedelta(minutes=5),
    "depends_on_past":  False,
    "email_on_failure": False,
}


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Extraction et préparation depuis MongoDB
# ════════════════════════════════════════════════════════════════════════
def task_extract_and_prepare(**context):
    """
    Lit les reviews depuis MongoDB (collection reviews).
    Nettoie le texte — IDENTIQUE à Notebook 02.
    Sauvegarde en Parquet pour Task 2.
    """
    from pymongo import MongoClient
    import pandas as pd
    import re, os

    logger.info("Task 1 : Connexion MongoDB...")
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
        logger.warning(
            f"Task 1 : {n} samples < minimum {MIN_SAMPLES}. "
            f"Réentraînement annulé."
        )
        context["ti"].xcom_push(key="n_samples", value=0)
        return

    df = pd.DataFrame(docs)
    df = df.dropna(subset=["Score", "Text"])

    # Label — même règle que Notebook 02
    def score_to_label(score):
        try:
            s = int(score)
            return 0 if s < 3 else (1 if s == 3 else 2)
        except:
            return None

    df["label"] = df["Score"].apply(score_to_label)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # Nettoyage texte — IDENTIQUE à Notebook 02 (5 étapes)
    def clean_text(row):
        summary = str(row.get("Summary", "") or "")
        text    = str(row.get("Text",    "") or "")
        full    = (summary + " " + text).lower()
        full    = re.sub(r"<[^>]+>",           " ", full)  # HTML
        full    = re.sub(r"https?://\S+|www\.\S+", " ", full)  # URLs
        full    = re.sub(r"[^a-z\s]",          " ", full)  # non-alpha
        full    = re.sub(r"\s+",               " ", full).strip()
        return full if len(full) > 10 else None

    df["cleaned_text"] = df.apply(clean_text, axis=1)
    df = df.dropna(subset=["cleaned_text"])

    logger.info(f"Task 1 : {len(df):,} reviews après nettoyage")

    os.makedirs(PARQUET_RAW, exist_ok=True)
    df[["cleaned_text", "label"]].to_parquet(
        f"{PARQUET_RAW}/data.parquet", index=False
    )

    context["ti"].xcom_push(key="n_samples", value=len(df))
    logger.info(f"Task 1 ✅ Parquet sauvegardé : {PARQUET_RAW}/data.parquet")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Réentraînement + MLflow
# ════════════════════════════════════════════════════════════════════════
def task_retrain_model(**context):
    """
    Applique le preprocessing_pipeline (TF-IDF).
    Entraîne un nouveau modèle LR.
    Logue TOUT dans MLflow : params, metrics, modèle.
    Sauvegarde le candidat dans /opt/models/candidate_model.
    """
    import os, sys, time
    os.environ["PYSPARK_PYTHON"]        = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    n_samples = context["ti"].xcom_pull(
        key="n_samples", task_ids="extract_and_prepare_data"
    )

    if not n_samples or n_samples < MIN_SAMPLES:
        logger.info("Task 2 : Pas assez de samples — ignorée.")
        for key in ["new_f1", "new_acc", "recall_neg",
                    "mlflow_run_id", "run_name", "n_train"]:
            context["ti"].xcom_push(key=key, value=None)
        return

    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, when
    from pyspark.ml import PipelineModel
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    import pandas as pd
    import mlflow
    import mlflow.spark

    spark = SparkSession.builder \
        .appName("Airflow_Retrain") \
        .master("local[2]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    try:
        # Charger données
        pdf = pd.read_parquet(f"{PARQUET_RAW}/data.parquet")
        df  = spark.createDataFrame(pdf)

        # Vérifier que le pipeline est accessible via le volume monté
        if not os.path.exists(PREPROCESSING_PATH):
            raise FileNotFoundError(
                f"Pipeline introuvable : {PREPROCESSING_PATH}\n"
                f"Vérifier que './models:/opt/models' est dans docker-compose.yml"
            )

        preprocessing = PipelineModel.load(PREPROCESSING_PATH)
        transformed   = preprocessing.transform(df)

        # Class weights — formule N/(K×count)
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

        # ── MLflow ────────────────────────────────────────────────
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        run_name = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M')}"

        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id

            # Log paramètres
            mlflow.log_param("algorithm",         "LogisticRegression")
            mlflow.log_param("maxIter",            60)
            mlflow.log_param("regParam",           0.05)
            mlflow.log_param("elasticNetParam",    0.0)
            mlflow.log_param("family",             "multinomial")
            mlflow.log_param("neg_threshold",      NEG_THRESHOLD)
            mlflow.log_param("n_training",         n_train)
            mlflow.log_param("n_eval",             n_eval)
            mlflow.log_param("class_weight",       "N/(K*count)")
            mlflow.log_param("features",           "TF-IDF unigrams+bigrams ~20K")

            # Entraînement
            lr = LogisticRegression(
                featuresCol     = "features",
                labelCol        = "label_idx",
                weightCol       = "classWeight",
                maxIter         = 60,
                regParam        = 0.05,
                elasticNetParam = 0.0,
                family          = "multinomial"
            )

            t0         = time.time()
            new_model  = lr.fit(train_df)
            train_time = round(time.time() - t0, 1)

            mlflow.log_metric("train_time_seconds", train_time)
            logger.info(f"Task 2 : Modèle entraîné en {train_time}s")

            # Évaluation
            eval_preds = new_model.transform(eval_df)

            new_f1  = MulticlassClassificationEvaluator(
                labelCol="label_idx", metricName="f1"
            ).evaluate(eval_preds)

            new_acc = MulticlassClassificationEvaluator(
                labelCol="label_idx", metricName="accuracy"
            ).evaluate(eval_preds)

            # Métriques par classe
            per_class = {}
            for idx in [0.0, 1.0, 2.0]:
                tp = eval_preds.filter(
                    (col("label_idx")==idx) & (col("prediction")==idx)
                ).count()
                fp = eval_preds.filter(
                    (col("label_idx")!=idx) & (col("prediction")==idx)
                ).count()
                fn = eval_preds.filter(
                    (col("label_idx")==idx) & (col("prediction")!=idx)
                ).count()
                pr  = tp/(tp+fp) if (tp+fp) > 0 else 0
                rc  = tp/(tp+fn) if (tp+fn) > 0 else 0
                f1c = 2*pr*rc/(pr+rc) if (pr+rc) > 0 else 0
                per_class[idx] = {"precision": pr, "recall": rc, "f1": f1c}

            # Log métriques MLflow
            mlflow.log_metric("f1_macro",           round(new_f1, 4))
            mlflow.log_metric("accuracy",           round(new_acc, 4))
            mlflow.log_metric("f1_negative",        round(per_class[0.0]["f1"], 4))
            mlflow.log_metric("f1_neutral",         round(per_class[1.0]["f1"], 4))
            mlflow.log_metric("f1_positive",        round(per_class[2.0]["f1"], 4))
            mlflow.log_metric("recall_negative",    round(per_class[0.0]["recall"], 4))
            mlflow.log_metric("recall_neutral",     round(per_class[1.0]["recall"], 4))
            mlflow.log_metric("recall_positive",    round(per_class[2.0]["recall"], 4))
            mlflow.log_metric("precision_negative", round(per_class[0.0]["precision"], 4))

            # Log du modèle dans MLflow
            mlflow.spark.log_model(
                spark_model   = new_model,
                artifact_path = "sentiment_model"
            )

            logger.info(
                f"Task 2 : F1={new_f1:.4f}  Acc={new_acc:.4f}  "
                f"RecallNeg={per_class[0.0]['recall']:.4f}  "
                f"RunID={run_id}"
            )

            # Sauvegarder le candidat localement
            new_model.write().overwrite().save(CANDIDATE_PATH)

            # XCom vers tasks suivantes
            context["ti"].xcom_push(key="new_f1",      value=new_f1)
            context["ti"].xcom_push(key="new_acc",     value=new_acc)
            context["ti"].xcom_push(key="recall_neg",  value=per_class[0.0]["recall"])
            context["ti"].xcom_push(key="recall_neu",  value=per_class[1.0]["recall"])
            context["ti"].xcom_push(key="f1_neg",      value=per_class[0.0]["f1"])
            context["ti"].xcom_push(key="mlflow_run_id", value=run_id)
            context["ti"].xcom_push(key="run_name",    value=run_name)
            context["ti"].xcom_push(key="n_train",     value=int(N))

    finally:
        spark.stop()


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Évaluation et Promotion
# ════════════════════════════════════════════════════════════════════════
def task_evaluate_and_promote(**context):
    """
    Compare candidat vs modèle production.
    Promeut si F1 > +1.5%.
    Met à jour le tag MLflow avec la décision finale.
    """
    import os, sys
    os.environ["PYSPARK_PYTHON"]        = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    new_f1  = context["ti"].xcom_pull(key="new_f1",        task_ids="retrain_model")
    run_id  = context["ti"].xcom_pull(key="mlflow_run_id", task_ids="retrain_model")

    if new_f1 is None:
        logger.info("Task 3 : Pas de modèle candidat — ignorée.")
        context["ti"].xcom_push(key="promoted", value=False)
        context["ti"].xcom_push(key="old_f1",   value=None)
        return

    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, when
    from pyspark.ml import PipelineModel
    from pyspark.ml.classification import LogisticRegressionModel
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    import pandas as pd
    import mlflow

    spark = SparkSession.builder \
        .appName("Airflow_Evaluate") \
        .master("local[2]") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    promoted = False
    old_f1   = None

    try:
        # Reconstruire le même eval_df (seed=42 → mêmes lignes)
        pdf = pd.read_parquet(f"{PARQUET_RAW}/data.parquet")
        df  = spark.createDataFrame(pdf)
        preprocessing = PipelineModel.load(PREPROCESSING_PATH)
        transformed   = preprocessing.transform(df)
        _, eval_df    = transformed.randomSplit([0.85, 0.15], seed=42)

        evaluator = MulticlassClassificationEvaluator(
            labelCol="label_idx", metricName="f1"
        )

        try:
            old_model = LogisticRegressionModel.load(MODEL_PATH)
            old_preds = old_model.transform(eval_df)
            old_f1    = evaluator.evaluate(old_preds)

            improvement = (new_f1 - old_f1) / old_f1 * 100 if old_f1 > 0 else 100

            logger.info(
                f"Task 3 : Production F1={old_f1:.4f}  "
                f"Candidat F1={new_f1:.4f}  "
                f"Amélioration={improvement:+.2f}%"
            )

            if new_f1 > old_f1 * (1 + PROMOTION_THRESHOLD):
                candidate = LogisticRegressionModel.load(CANDIDATE_PATH)
                candidate.write().overwrite().save(MODEL_PATH)
                promoted = True
                logger.info(
                    f"Task 3 🚀 PROMU — "
                    f"F1: {old_f1:.4f} → {new_f1:.4f} ({improvement:+.2f}%)"
                )
            else:
                logger.info(
                    f"Task 3 ⏭️  REJETÉ — "
                    f"{improvement:+.2f}% < {PROMOTION_THRESHOLD*100:.1f}%"
                )

        except Exception as e:
            logger.warning(
                f"Task 3 : Modèle production introuvable "
                f"({type(e).__name__}). Promotion automatique."
            )
            candidate = LogisticRegressionModel.load(CANDIDATE_PATH)
            candidate.write().overwrite().save(MODEL_PATH)
            promoted = True
            old_f1   = 0.0

        # ── Mettre à jour le run MLflow avec la décision ──────────
        mlflow.set_tracking_uri(MLFLOW_URI)
        with mlflow.start_run(run_id=run_id):
            mlflow.set_tag("promoted", str(promoted))
            mlflow.set_tag(
                "decision", "PROMOTED" if promoted else "REJECTED"
            )
            if old_f1 is not None and old_f1 > 0:
                mlflow.log_metric("previous_f1", round(old_f1, 4))
                mlflow.log_metric(
                    "f1_improvement",
                    round((new_f1 - old_f1) / old_f1 * 100, 2)
                )

    finally:
        context["ti"].xcom_push(key="promoted", value=promoted)
        context["ti"].xcom_push(key="old_f1",   value=old_f1)
        spark.stop()


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Mise à jour MongoDB + résumé dans les logs
# ════════════════════════════════════════════════════════════════════════
def task_update_model_metrics(**context):
    """
    Écrit les résultats dans MongoDB collection model_metrics.
    Inclut le lien MLflow pour accéder au run depuis le dashboard.
    Pas de SparkSession — pymongo direct.
    """
    from pymongo import MongoClient

    new_f1     = context["ti"].xcom_pull(key="new_f1",        task_ids="retrain_model")
    new_acc    = context["ti"].xcom_pull(key="new_acc",        task_ids="retrain_model")
    recall_neg = context["ti"].xcom_pull(key="recall_neg",     task_ids="retrain_model")
    recall_neu = context["ti"].xcom_pull(key="recall_neu",     task_ids="retrain_model")
    f1_neg     = context["ti"].xcom_pull(key="f1_neg",         task_ids="retrain_model")
    run_id     = context["ti"].xcom_pull(key="mlflow_run_id",  task_ids="retrain_model")
    run_name   = context["ti"].xcom_pull(key="run_name",       task_ids="retrain_model")
    n_train    = context["ti"].xcom_pull(key="n_train",        task_ids="retrain_model")
    n_samples  = context["ti"].xcom_pull(key="n_samples",      task_ids="extract_and_prepare_data")
    promoted   = context["ti"].xcom_pull(key="promoted",       task_ids="evaluate_and_promote")
    old_f1     = context["ti"].xcom_pull(key="old_f1",         task_ids="evaluate_and_promote")

    if new_f1 is None:
        logger.info("Task 4 : Pas de métriques — ignorée.")
        return

    client = MongoClient(MONGO_URI)
    db     = client[MONGO_DB]

    try:
        improvement = None
        if old_f1 and old_f1 > 0:
            improvement = round((new_f1 - old_f1) / old_f1 * 100, 2)

        doc = {
            # Champs compatibles avec collection model_metrics existante
            "type":               "retraining",
            "accuracy":           round(new_acc, 4)    if new_acc    else None,
            "f1_macro":           round(new_f1, 4),
            "total_processed":    n_samples,

            # Champs enrichis pour le dashboard Model Health
            "recall_negative":    round(recall_neg, 4) if recall_neg else None,
            "recall_neutral":     round(recall_neu, 4) if recall_neu else None,
            "f1_negative":        round(f1_neg, 4)     if f1_neg     else None,
            "previous_f1":        round(old_f1, 4)     if old_f1     else None,
            "improvement_pct":    improvement,
            "n_training_samples": n_train,
            "neg_threshold":      NEG_THRESHOLD,
            "model_path":         MODEL_PATH,
            "promoted":           promoted,

            # Lien MLflow — cliquable depuis le dashboard Flask
            "mlflow_run_id":      run_id,
            "mlflow_url":         f"{MLFLOW_URI}/#/experiments/1/runs/{run_id}",
            "run_name":           run_name,

            "retrained_at":       datetime.now(),
        }

        db[COL_MODEL_METRICS].insert_one(doc)

        logger.info(
            f"Task 4 ✅ model_metrics mis à jour — "
            f"F1={new_f1:.4f}  promoted={promoted}  RunID={run_id}"
        )

        # ── Résumé visible dans les logs Airflow ─────────────────
        logger.info("=" * 52)
        logger.info("  RÉSUMÉ RÉENTRAÎNEMENT QUOTIDIEN")
        logger.info("=" * 52)
        logger.info(f"  Run name           : {run_name}")
        logger.info(f"  Samples utilisés   : {n_samples:,}")
        logger.info(f"  F1 Macro nouveau   : {new_f1:.4f}")
        logger.info(f"  F1 Macro ancien    : {old_f1:.4f if old_f1 else 'N/A'}")
        logger.info(f"  Amélioration       : {improvement:+.2f}%" if improvement else "  Amélioration       : Premier run")
        logger.info(f"  Accuracy           : {new_acc:.4f if new_acc else 'N/A'}")
        logger.info(f"  Recall Négatif     : {recall_neg:.4f if recall_neg else 'N/A'}")
        logger.info(f"  Modèle promu       : {'✅ OUI' if promoted else '⏭️  NON'}")
        logger.info(f"  MLflow UI          : {MLFLOW_URI}/#/experiments/1/runs/{run_id}")
        logger.info("=" * 52)

    finally:
        client.close()


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Cleanup
# ════════════════════════════════════════════════════════════════════════
def task_cleanup(**context):
    """Supprime les fichiers Parquet temporaires."""
    import shutil
    try:
        shutil.rmtree(PARQUET_RAW, ignore_errors=True)
        logger.info(f"Task 5 ✅ Supprimé : {PARQUET_RAW}")
    except Exception as e:
        logger.warning(f"Task 5 : Cleanup échoué ({e}) — non critique")


# ════════════════════════════════════════════════════════════════════════
# DÉFINITION DU DAG
# ════════════════════════════════════════════════════════════════════════
with DAG(
    dag_id            = "daily_sentiment_retraining",
    default_args      = default_args,
    description       = "Réentraînement quotidien + MLflow tracking",
    schedule_interval = "0 3 * * *",
    start_date        = datetime(2026, 5, 1),
    catchup           = False,
    tags              = ["ml", "retraining", "mlflow", "sentiment"],
    doc_md            = __doc__,
) as dag:

    t1 = PythonOperator(
        task_id         = "extract_and_prepare_data",
        python_callable = task_extract_and_prepare,
        provide_context = True,
        doc_md          = "MongoDB reviews → nettoyage → Parquet",
    )

    t2 = PythonOperator(
        task_id         = "retrain_model",
        python_callable = task_retrain_model,
        provide_context = True,
        doc_md          = "TF-IDF → LR.fit() → MLflow log → candidat sauvegardé",
    )

    t3 = PythonOperator(
        task_id         = "evaluate_and_promote",
        python_callable = task_evaluate_and_promote,
        provide_context = True,
        doc_md          = "Compare candidat vs production → promeut si F1 > +1.5%",
    )

    t4 = PythonOperator(
        task_id         = "update_model_metrics",
        python_callable = task_update_model_metrics,
        provide_context = True,
        doc_md          = "Écrit résultats dans MongoDB + lien MLflow",
    )

    t5 = PythonOperator(
        task_id         = "cleanup",
        python_callable = task_cleanup,
        provide_context = True,
        trigger_rule    = "all_done",
        doc_md          = "Supprime les fichiers temporaires",
    )

    t1 >> t2 >> t3 >> t4 >> t5
