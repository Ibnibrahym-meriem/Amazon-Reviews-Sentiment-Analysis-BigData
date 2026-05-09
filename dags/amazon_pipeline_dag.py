from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import socket
import pymongo

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

KAFKA_HOST  = "kafka"
KAFKA_PORT  = 29092
MONGO_HOST  = "mongodb"
MONGO_PORT  = 27017
SPARK_HOST  = "spark-streaming"  # nom du service Docker Spark (voir docker-compose.yml)
SPARK_PORT  = 4040           # Spark UI / REST API port


# ─────────────────────────────────────────────
# TÂCHE 1 — Vérifier Kafka
# ─────────────────────────────────────────────
def check_kafka(**context):
    try:
        sock = socket.create_connection((KAFKA_HOST, KAFKA_PORT), timeout=10)
        sock.close()
        print(f"OK Kafka accessible sur {KAFKA_HOST}:{KAFKA_PORT}")
        return True
    except Exception as e:
        raise Exception(f"Kafka inaccessible ({KAFKA_HOST}:{KAFKA_PORT}) : {e}")


# ─────────────────────────────────────────────
# TÂCHE 2 — Vérifier Spark Streaming (NOUVEAU)
# ─────────────────────────────────────────────
def check_spark(**context):
    """
    Vérifie que Spark Streaming est actif via deux méthodes complémentaires :
    1. Connexion TCP sur le port Spark UI (4040) → confirme que le process tourne
    2. Lecture de model_metrics dans MongoDB    → confirme que Spark écrit bien des données
    """
    errors = []

    # — Méthode 1 : Port Spark UI (4040) ————————————————————————
    try:
        sock = socket.create_connection((SPARK_HOST, SPARK_PORT), timeout=10)
        sock.close()
        print(f"OK Spark UI accessible sur {SPARK_HOST}:{SPARK_PORT}")
    except Exception as e:
        errors.append(f"Spark UI inaccessible ({SPARK_HOST}:{SPARK_PORT}) : {e}")

    # — Méthode 2 : MongoDB model_metrics ————————————————————————
    try:
        client = pymongo.MongoClient(
            f"mongodb://{MONGO_HOST}:{MONGO_PORT}",
            serverSelectionTimeoutMS=5000
        )
        db = client["amazon_db"]
        metrics = db["model_metrics"].find_one()
        client.close()

        if metrics is None:
            errors.append("model_metrics vide — Spark n'a encore rien écrit dans MongoDB")
        else:
            total_processed = metrics.get("total_processed", 0)
            accuracy        = metrics.get("accuracy", "N/A")
            print(f"OK Spark actif — {total_processed} reviews traités | Accuracy : {accuracy}")

    except Exception as e:
        errors.append(f"Impossible de lire model_metrics : {e}")

    # — Résultat final ————————————————————————————————————————————
    if errors:
        # Si le port est KO mais MongoDB est OK (ou inversement), on warn sans bloquer
        # Si les deux sont KO, on lève une exception
        if len(errors) == 2:
            raise Exception("Spark semble inactif :\n  - " + "\n  - ".join(errors))
        else:
            # Un seul problème → avertissement dans les logs mais pas de blocage
            print(f"AVERTISSEMENT Spark : {errors[0]}")
    return True


# ─────────────────────────────────────────────
# TÂCHE 3 — Vérifier MongoDB
# ─────────────────────────────────────────────
def check_mongodb(**context):
    try:
        client = pymongo.MongoClient(
            f"mongodb://{MONGO_HOST}:{MONGO_PORT}",
            serverSelectionTimeoutMS=5000
        )
        client.server_info()
        db    = client["amazon_db"]
        count = db["reviews"].count_documents({})
        print(f"OK MongoDB - {count} reviews dans amazon_db")
        client.close()
        return count
    except Exception as e:
        raise Exception(f"MongoDB inaccessible : {e}")


# ─────────────────────────────────────────────
# TÂCHE 4 — Rapport santé pipeline
# ─────────────────────────────────────────────
def check_pipeline_health(**context):
    try:
        client   = pymongo.MongoClient(f"mongodb://{MONGO_HOST}:{MONGO_PORT}", serverSelectionTimeoutMS=5000)
        db       = client["amazon_db"]
        total    = db["reviews"].count_documents({})
        positive = db["reviews"].count_documents({"predicted_sentiment": "positive"})
        negative = db["reviews"].count_documents({"predicted_sentiment": "negative"})
        neutral  = db["reviews"].count_documents({"predicted_sentiment": "neutral"})
        metrics  = db["model_metrics"].find_one()
        accuracy = metrics.get("accuracy", "N/A") if metrics else "N/A"
        total_processed = metrics.get("total_processed", "N/A") if metrics else "N/A"

        print("=================================================")
        print("RAPPORT PIPELINE AMAZON REVIEWS SENTIMENT")
        print("=================================================")
        print(f"Total reviews    : {total}")
        if total > 0:
            print(f"Positifs         : {positive} ({round(positive / total * 100, 1)}%)")
            print(f"Negatifs         : {negative} ({round(negative / total * 100, 1)}%)")
            print(f"Neutres          : {neutral}  ({round(neutral  / total * 100, 1)}%)")
        print(f"Accuracy modele  : {accuracy}")
        print(f"Total processed  : {total_processed}")
        print("=================================================")
        client.close()

        if total == 0:
            raise Exception("Aucun review dans MongoDB !")
        return total

    except pymongo.errors.ServerSelectionTimeoutError as e:
        raise Exception(f"MongoDB inaccessible : {e}")


# ─────────────────────────────────────────────
# TÂCHE 5 — Top produits
# ─────────────────────────────────────────────
def check_product_metrics(**context):
    try:
        client      = pymongo.MongoClient(f"mongodb://{MONGO_HOST}:{MONGO_PORT}", serverSelectionTimeoutMS=5000)
        db          = client["amazon_db"]
        nb_products = db["product_metrics"].count_documents({})
        top         = list(db["product_metrics"].find().sort("review_count", -1).limit(3))
        print(f"OK {nb_products} produits dans product_metrics")
        for p in top:
            print(
                f"  - {p.get('_id')} : {p.get('review_count')} reviews "
                f"| {round(p.get('positive_ratio', 0) * 100)}% positifs"
            )
        client.close()
        return nb_products
    except Exception as e:
        raise Exception(f"Erreur product_metrics : {e}")


# ─────────────────────────────────────────────
# DAG — Définition et chaîne des tâches
# ─────────────────────────────────────────────
with DAG(
    dag_id='amazon_reviews_pipeline',
    default_args=default_args,
    description='Orchestration du pipeline Amazon Reviews Sentiment Analysis',
    schedule_interval='@hourly',
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['bigdata', 'kafka', 'spark', 'mongodb'],
) as dag:

    t1 = PythonOperator(task_id='check_kafka',            python_callable=check_kafka)
    t2 = PythonOperator(task_id='check_spark',            python_callable=check_spark)
    t3 = PythonOperator(task_id='check_mongodb',          python_callable=check_mongodb)
    t4 = PythonOperator(task_id='pipeline_health_report', python_callable=check_pipeline_health)
    t5 = PythonOperator(task_id='check_product_metrics',  python_callable=check_product_metrics)

    t1 >> t2 >> t3 >> t4 >> t5