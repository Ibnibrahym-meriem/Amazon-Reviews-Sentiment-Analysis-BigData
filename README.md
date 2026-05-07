# Amazon Reviews Sentiment Analysis — Big Data

Analyse des avis clients Amazon en temps réel avec Kafka, Spark Streaming, MongoDB, ML et Apache Airflow.

---

## Architecture

```
Reviews.csv (500 000 avis)
        │
        ├── 80% → Entraînement ML (LogisticRegression + TF-IDF)
        ├── 10% → Validation
        └── 10% → Flux temps réel via Kafka
                        │
                producer.py (1 avis/sec)
                        │
                Apache Kafka (Topic: amazon_reviews)
                        │
                Spark Streaming (prédiction sentiment)
                        │
                MongoDB (3 collections)
                ├── reviews          → avis + prédictions
                ├── product_metrics  → stats par produit
                └── model_metrics    → performance du modèle
                        │
                Apache Airflow (orchestration @hourly)
                ├── check_kafka          → vérifie Kafka
                ├── check_mongodb        → vérifie MongoDB
                ├── pipeline_health_report → rapport sentiment
                └── check_product_metrics  → top produits
```

---

## Technologies

| Technologie | Rôle |
|---|---|
| Apache Kafka | Collecte flux temps réel |
| Zookeeper | Gestion cluster Kafka |
| Apache Spark (PySpark) | Traitement distribué + ML |
| MongoDB | Stockage NoSQL des résultats |
| **Apache Airflow** | **Orchestration et monitoring du pipeline** |
| **PostgreSQL** | **Base de données interne Airflow** |
| Docker | Conteneurisation des services |
| Python 3.10 | Développement (via conda spark_env) |
| Java 17 | Requis pour Spark |

---

## Prérequis

Installer dans cet ordre :

1. **Docker Desktop** → https://www.docker.com/products/docker-desktop
2. **Anaconda** → https://www.anaconda.com/download (pour gérer les environnements Python)
3. **Java 17 (Temurin)** → https://adoptium.net/temurin/releases/?version=17&os=windows&arch=x64&package=jdk
4. **winutils.exe** (requis pour Spark sur Windows) :
   - Télécharger `winutils.exe` et `hadoop.dll` depuis https://github.com/cdarlint/winutils/tree/master/hadoop-3.3.6/bin
   - Créer le dossier `C:\hadoop\bin` et y placer les deux fichiers

### Environnement Python (conda)

```bash
conda create -n spark_env python=3.10 -y
conda activate spark_env
pip install pyspark==3.5.1
pip install kafka-python
pip install pandas
pip install pymongo
```

> ⚠️ Utiliser Python 3.10 — PySpark 3.5.1 est incompatible avec Python 3.11+

---

## Installation et lancement

### Étape 1 — Cloner le repo

```bash
git clone https://github.com/Ibnibrahym-meriem/Amazon-Reviews-Sentiment-Analysis-BigData.git
cd Amazon-Reviews-Sentiment-Analysis-BigData
```

### Étape 2 — Ajouter le fichier de données

Télécharger `Reviews.csv` depuis Kaggle :
https://www.kaggle.com/snap/amazon-fine-food-reviews

Placer le fichier à la racine du projet.

### Étape 3 — Créer les dossiers Airflow

```powershell
mkdir dags, logs, plugins
```

### Étape 4 — Copier le DAG Airflow

```powershell
copy amazon_pipeline_dag.py dags\
```

### Étape 5 — Créer le fichier `run_spark.cmd` (Windows)

Créer un fichier `run_spark.cmd` à la racine du projet via PowerShell :

```powershell
@"
@echo off
set JAVA_HOME=C:\Program Files\Eclipse Adoptium\jdk-17.0.19.10-hotspot
set HADOOP_HOME=C:\hadoop
set PATH=%JAVA_HOME%\bin;%HADOOP_HOME%\bin;%PATH%
set JAVA_TOOL_OPTIONS=--add-opens=java.base/javax.security.auth=ALL-UNNAMED --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.text=ALL-UNNAMED --add-opens=java.sql/java.sql=ALL-UNNAMED
echo Lancement de Spark Streaming...
python spark_streaming.py
"@ | Out-File -FilePath run_spark.cmd -Encoding ascii
```

> Adapter le chemin `JAVA_HOME` selon ton installation Java 17.

### Étape 6 — Lancer les services Docker

```bash
docker-compose up -d
```

Vérifier que ces services sont Running :
- Zookeeper → port 2181
- Kafka → port 9092
- MongoDB → port 27017
- PostgreSQL (Airflow) → port interne
- Mongo Express → http://localhost:8081
- Kafka UI → http://localhost:8080
- **Airflow Webserver → http://localhost:8082**

### Étape 7 — Créer l'utilisateur Airflow

La première fois seulement :

```powershell
docker exec -it amazon-reviews-sentiment-analysis-bigdata-airflow-webserver-1 airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@admin.com
```

### Étape 8 — Créer le topic Kafka

Aller sur http://localhost:8080 → Topics → **+ Add a Topic**

- Topic Name : `amazon_reviews`
- Partitions : `1`
- Replication Factor : `1`

### Étape 9 — Lancer Spark Streaming

Ouvrir un nouveau terminal :

```powershell
conda activate spark_env
.\run_spark.cmd
```

Attendre que les JARs se téléchargent (première fois ~2 minutes).
Tu dois voir : `[BATCH X] ✅ product_metrics mis à jour`

### Étape 10 — Lancer le Producer

Ouvrir un autre terminal :

```powershell
conda activate spark_env
python producer.py
```

Tu verras :
```
[PRODUCER] ✅ Avis 511683 envoyé | Produit: B000FFLXPG | Score: 5/5
[PRODUCER] ✅ Avis 511684 envoyé | Produit: B000FFLXPG | Score: 4/5
...
```

### Étape 11 — Vérifier dans MongoDB

Ouvrir http://localhost:8081

Tu dois voir dans `amazon_db` :

| Collection | Description |
|---|---|
| `reviews` | Avis + prédictions en temps réel |
| `product_metrics` | Stats agrégées par produit |
| `model_metrics` | Accuracy du modèle |

---

## Relancer le projet après redémarrage PC

```
Terminal 1 → docker-compose up -d
Terminal 2 → conda activate spark_env && .\run_spark.cmd
Terminal 3 → conda activate spark_env && python producer.py
```

Airflow redémarre **automatiquement** avec Docker.

---

## Structure du projet

```
Amazon-Reviews-Sentiment-Analysis-BigData/
│
├── dags/
│   └── amazon_pipeline_dag.py     ← DAG Airflow
├── logs/                          ← Logs Airflow (auto-généré)
├── plugins/                       ← Plugins Airflow (vide)
│
├── models/
│   ├── preprocessing_pipeline/    ← Pipeline TF-IDF sauvegardé
│   └── final_best_model/          ← Modèle LogisticRegression sauvegardé
│
├── notebooks/
│   ├── 01_EDA.ipynb               ← Exploration des données
│   ├── 02_Preprocessing_P.ipynb   ← Préparation des données
│   └── 03_Model_Training_.ipynb   ← Entraînement du modèle
│
├── spark_streaming.py             ← Script principal Spark
├── producer.py                    ← Envoi des avis vers Kafka
├── docker-compose.yml             ← Configuration Docker (+ Airflow)
├── run_spark.cmd                  ← Lancement Spark (Windows)
└── README.md
```

---

## Apache Airflow — Orchestration du pipeline

### Accès

- URL : **http://localhost:8082**
- Login : `admin`
- Mot de passe : `admin`

### DAG : `amazon_reviews_pipeline`

Le DAG s'exécute automatiquement **toutes les heures** (`@hourly`) et effectue 4 vérifications :

```
check_kafka → check_mongodb → pipeline_health_report → check_product_metrics
```

| Tâche | Rôle |
|---|---|
| `check_kafka` | Vérifie que Kafka est accessible sur `kafka:29092` |
| `check_mongodb` | Vérifie MongoDB et compte les reviews traités |
| `pipeline_health_report` | Rapport complet : % positifs/négatifs/neutres + accuracy |
| `check_product_metrics` | Top 3 produits les plus commentés |

### Exemple de rapport dans les logs Airflow

```
=================================================
RAPPORT PIPELINE AMAZON REVIEWS SENTIMENT
=================================================
Total reviews    : 4552
Positifs         : 2277 (50.0%)
Negatifs         : 1252 (27.5%)
Neutres          : 1023 (22.5%)
Accuracy modele  : 0.75
=================================================

OK 306 produits dans product_metrics
  - B004HOLD60 : 106 reviews | 57% positifs
  - B001E4S88W :  29 reviews | 55% positifs
  - B001E5E3S0 :  28 reviews | 46% positifs
```

### Voir les logs d'une tâche

1. Aller sur http://localhost:8082
2. Cliquer sur `amazon_reviews_pipeline`
3. Cliquer sur l'onglet **Graph**
4. Cliquer sur une tâche verte
5. Cliquer sur **Logs**

### Architecture Docker Airflow

| Service | Rôle |
|---|---|
| `airflow-webserver` | Interface web → port 8082 |
| `airflow-scheduler` | Exécute les DAGs automatiquement |
| `airflow-init` | Initialise la base de données (premier démarrage) |
| `postgres` | Base de données interne Airflow |

> ⚠️ Airflow communique avec Kafka et MongoDB via les **noms de services Docker** (`kafka:29092`, `mongodb:27017`) et non via `localhost`.

---

## Collections MongoDB

### 1. `reviews` — Données brutes + prédictions

```json
{
  "Id": 568454,
  "ProductId": "B001E4KFG0",
  "UserId": "A3SGXH7AUHU8GW",
  "ProfileName": "delmartian",
  "Score": 5,
  "Time": 1303862400,
  "Summary": "Great product!",
  "Text": "Absolutely love this product...",
  "predicted_sentiment": "positive",
  "confidence": 0.94,
  "processed_at": "2026-05-04T10:30:00Z"
}
```

### 2. `product_metrics` — Stats par produit

```json
{
  "_id": "B001E4KFG0",
  "review_count": 150,
  "avg_sentiment": 1.8,
  "positive_ratio": 0.76,
  "negative_ratio": 0.12,
  "neutral_ratio": 0.12,
  "last_updated": "2026-05-04T10:30:00Z"
}
```

### 3. `model_metrics` — Performance du modèle

```json
{
  "accuracy": 0.87,
  "total_processed": 500,
  "correct_predictions": 435,
  "model_path": "models/final_best_model"
}
```

---

## Données

Source : https://www.kaggle.com/snap/amazon-fine-food-reviews

- 568 454 avis sur des produits alimentaires Amazon
- Période : plus de 10 ans
- Colonnes : Id, ProductId, UserId, ProfileName, Score, Time, Summary, Text

### Labels de sentiment

| Score | Sentiment |
|---|---|
| 1 ou 2 | Négatif |
| 3 | Neutre |
| 4 ou 5 | Positif |

---

## Modèle ML

- **Algorithme** : Logistic Regression (multinomiale)
- **Features** : TF-IDF unigrammes + bigrammes
- **Split** : 80% train / 10% validation / 10% test
- **Stockage** : `models/final_best_model/`

---

## Résolution des problèmes courants

### Erreur Java `getSubject is not supported`

Cause : Java 17+ requis. Utiliser `run_spark.cmd` au lieu de `python spark_streaming.py`.

### Erreur `UnsatisfiedLinkError` (winutils manquant)

```
java.lang.UnsatisfiedLinkError: NativeIO$Windows.access0
```

Solution : Télécharger `winutils.exe` et `hadoop.dll` depuis https://github.com/cdarlint/winutils/tree/master/hadoop-3.3.6/bin et les placer dans `C:\hadoop\bin`.

### Erreur `Python worker exited unexpectedly`

Cause : incompatibilité Python 3.11 + PySpark. Utiliser Python 3.10 via conda :

```powershell
conda create -n spark_env python=3.10 -y
conda activate spark_env
pip install pyspark==3.5.1
```

### Erreur `ChecksumException`

```bash
del /s /q models\preprocessing_pipeline\metadata\.*.crc
del /s /q models\preprocessing_pipeline\stages\*.crc
del /s /q models\final_best_model\metadata\.*.crc
```

### Kafka : topic introuvable

Créer le topic sur http://localhost:8080 → Topics → **+ Add a Topic**

### MongoDB vide après lancement

Vérifier que `producer.py` et `run_spark.cmd` tournent dans leurs terminaux respectifs.

### Airflow : Invalid login

Recréer l'utilisateur manuellement :

```powershell
docker exec -it amazon-reviews-sentiment-analysis-bigdata-airflow-webserver-1 airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@admin.com
```

### Airflow : DAG Import Error (pymongo manquant)

Ajouter dans `docker-compose.yml` sous `x-airflow-common > environment` :

```yaml
_PIP_ADDITIONAL_REQUIREMENTS: 'pymongo'
```

Puis relancer :

```powershell
docker-compose up -d --no-deps airflow-webserver airflow-scheduler
```

### Airflow : DAG failed (Kafka inaccessible)

Vérifier que le DAG utilise les noms de services Docker et non `localhost` :

```python
KAFKA_HOST = "kafka"      # correct
KAFKA_HOST = "localhost"  # incorrect depuis un conteneur Docker
```

---