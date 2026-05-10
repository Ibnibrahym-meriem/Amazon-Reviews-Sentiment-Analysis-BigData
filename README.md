# Amazon Reviews Sentiment Analysis — Big Data

Analyse des avis clients Amazon en temps réel avec Kafka, Spark Streaming, MongoDB, ML, Apache Airflow et MLflow.

---

## Architecture

```
Reviews.csv (568 454 avis)
        │
        ├── 80% → Entraînement ML (LogisticRegression + TF-IDF)
        ├── 10% → Validation (tuning hyperparamètres)
        └── 10% → Flux temps réel via Kafka
                        │
                producer.py (1 avis/sec)
                        │
                Apache Kafka (Topic: amazon_reviews)
                        │
                Spark Streaming (prédiction sentiment)
                │   └── Threshold Négatif = 0.35 (décision business)
                │
                MongoDB (3 collections)
                ├── reviews          → avis + prédictions
                ├── product_metrics  → stats par produit
                └── model_metrics    → performance + historique réentraînements
                        │
                        ├── Apache Airflow (orchestration)
                        │   ├── DAG 1 : amazon_reviews_pipeline (@hourly)
                        │   │   ├── check_kafka            → vérifie Kafka
                        │   │   ├── check_mongodb          → vérifie MongoDB
                        │   │   ├── pipeline_health_report → rapport sentiment
                        │   │   └── check_product_metrics  → top produits
                        │   │
                        │   └── DAG 2 : daily_sentiment_retraining (@daily 3h)
                        │       ├── extract_and_prepare_data → lit MongoDB
                        │       ├── retrain_model            → réentraîne LR
                        │       ├── evaluate_and_promote     → compare modèles
                        │       ├── update_model_metrics     → écrit résultats
                        │       └── cleanup                  → nettoyage
                        │
                        └── MLflow (tracking ML)
                            ├── Expérience : Amazon_Sentiment_Retraining
                            ├── Logs : params, metrics, modèle
                            └── UI : http://localhost:5001
                                │
                                └── Flask Dashboard (visualisation)
                                    ├── Page d'accueil : KPIs temps réel
                                    ├── Analyse par produit : stats détaillées
                                    ├── Flux live : avis en temps réel
                                    ├── Liste de surveillance : produits suivis
                                    └── Santé du modèle : métriques ML
                                        │
                                        └── UI : http://localhost:5050
```

---

## Technologies

| Technologie | Rôle |
|---|---|
| Apache Kafka | Collecte flux temps réel |
| Zookeeper | Gestion cluster Kafka |
| Apache Spark (PySpark) | Traitement distribué + ML |
| MongoDB | Stockage NoSQL des résultats |
| Apache Airflow | Orchestration — monitoring + réentraînement ML |
| MLflow | Tracking des expériences ML + versioning modèles |
| PostgreSQL | Base de données interne Airflow |
| Docker | Conteneurisation des services |
| Python 3.10 | Développement (via conda spark_env) |
| Java 17 | Requis pour Spark |

---

## Prérequis

Installer dans cet ordre :

1. **Docker Desktop** → https://www.docker.com/products/docker-desktop
2. **Anaconda** → https://www.anaconda.com/download
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
pip install mlflow==2.11.0
pip install flask
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

### Étape 3 — Créer les dossiers requis

```powershell
mkdir dags, logs, plugins, models
```

### Étape 4 — Copier les DAGs Airflow

```powershell
copy amazon_pipeline_dag.py dags\
copy dag_model_retraining.py dags\
```

> Les deux DAGs coexistent dans `dags/` et s'exécutent indépendamment.

### Étape 5 — Créer le fichier `run_spark.cmd` (Windows)

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

| Service | URL | Description |
|---|---|---|
| Zookeeper | port 2181 | Gestionnaire Kafka |
| Kafka | port 9092 | Message broker |
| MongoDB | port 27017 | Base de données |
| Mongo Express | http://localhost:8081 | Interface MongoDB |
| Kafka UI | http://localhost:8080 | Interface Kafka |
| Airflow | http://localhost:8082 | Orchestration |
| MLflow | http://localhost:5001 | Tracking ML |
| Flask Dashboard | http://localhost:5050 | Visualisation données |

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

```powershell
conda activate spark_env
.\run_spark.cmd
```

Attendre que les JARs se téléchargent (première fois ~2 minutes).
Tu dois voir : `[BATCH X] ✅ product_metrics mis à jour`

### Étape 10 — Lancer le Producer

```powershell
conda activate spark_env
python producer.py
```

Tu verras :
```
[PRODUCER] ✅ Avis 511683 envoyé | Produit: B000FFLXPG | Score: 5/5
[PRODUCER] ✅ Avis 511684 envoyé | Produit: B000FFLXPG | Score: 4/5
```

### Étape 12 — Lancer le Dashboard Flask

```powershell
conda activate spark_env
pip install flask
python app.py
```

Le dashboard sera accessible sur **http://localhost:5050**

---

## Relancer le projet après redémarrage PC

```
Terminal 1 → docker-compose up -d
Terminal 2 → conda activate spark_env && .\run_spark.cmd
Terminal 3 → conda activate spark_env && python producer.py
Terminal 4 → conda activate spark_env && python app.py
```

Airflow et MLflow redémarrent **automatiquement** avec Docker.

---

## Structure du projet

```
Amazon-Reviews-Sentiment-Analysis-BigData/
│
├── app.py                          ← Dashboard Flask (port 5050)
├── requirements.txt                ← Dépendances Flask
│
├── dags/
│   ├── amazon_pipeline_dag.py       ← DAG 1 : Monitoring (@hourly)
│   └── dag_model_retraining.py      ← DAG 2 : Réentraînement ML (@daily)
│
├── templates/                       ← Templates HTML du dashboard
│   ├── base.html
│   ├── index.html                   ← Page d'accueil + KPIs
│   ├── product.html                 ← Analyse par produit
│   ├── live.html                    ← Flux temps réel
│   ├── watchlist.html               ← Produits surveillés
│   └── model_health.html            ← Métriques ML
│
├── logs/                            ← Logs Airflow (auto-généré)
├── plugins/                         ← Plugins Airflow (vide)
│
├── models/
│   ├── preprocessing_pipeline/      ← Pipeline TF-IDF (Notebook 02)
│   └── final_best_model/            ← Modèle LogisticRegression (Notebook 03)
│
├── notebooks/
│   ├── 01_EDA.ipynb                 ← Exploration des données
│   ├── 02_Preprocessing_P.ipynb     ← Préparation + pipeline TF-IDF
│   └── 03_Model_Training_.ipynb     ← Entraînement + évaluation + export
│
├── spark_streaming.py               ← Script principal Spark
├── producer.py                      ← Envoi des avis vers Kafka
├── docker-compose.yml               ← Configuration Docker
├── run_spark.cmd                    ← Lancement Spark (Windows)
├── Dockerfile.airflow               ← Image Airflow personnalisée
├── Dockerfile.producer              ← Image Producer Kafka
├── Dockerfile.spark                 ← Image Spark Streaming
└── README.md
```

---

## Apache Airflow — Deux DAGs

### Accès

- URL : **http://localhost:8082**
- Login : `admin` / Mot de passe : `admin`

---

### DAG 1 : `amazon_reviews_pipeline` — Monitoring (@hourly)

S'exécute automatiquement **toutes les heures** pour vérifier que le pipeline fonctionne.

```
check_kafka → check_mongodb → pipeline_health_report → check_product_metrics
```

| Tâche | Rôle |
|---|---|
| `check_kafka` | Vérifie que Kafka est accessible sur `kafka:29092` |
| `check_mongodb` | Vérifie MongoDB et compte les reviews traités |
| `pipeline_health_report` | Rapport : % positifs/négatifs/neutres + accuracy |
| `check_product_metrics` | Top 3 produits les plus commentés |

**Exemple de rapport dans les logs :**

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
```

---

### DAG 2 : `daily_sentiment_retraining` — Réentraînement ML (@daily 3h)

S'exécute automatiquement **chaque nuit à 3h** pour améliorer le modèle ML.

```
extract_and_prepare_data
        │
        ▼
retrain_model ──────────────────► MLflow : params + métriques + modèle loggés
        │
        ▼
evaluate_and_promote ───────────► MLflow : tag promoted=True/False
        │  ├── F1 amélioré > 1.5% → remplace final_best_model/
        │  └── pas d'amélioration  → modèle actuel conservé
        ▼
update_model_metrics ───────────► MongoDB model_metrics + lien MLflow
        │
        ▼
cleanup (toujours exécuté)
```

| Tâche | Rôle |
|---|---|
| `extract_and_prepare_data` | Lit MongoDB `reviews`, nettoie le texte |
| `retrain_model` | Réentraîne LR, logue dans MLflow |
| `evaluate_and_promote` | Compare candidat vs production |
| `update_model_metrics` | Écrit résultats dans MongoDB + lien MLflow |
| `cleanup` | Supprime fichiers temporaires |

**Exemple de résumé dans les logs Airflow :**

```
====================================================
  RÉSUMÉ RÉENTRAÎNEMENT QUOTIDIEN
====================================================
  Run name           : retrain_20260511_0300
  Samples utilisés   : 4,552
  F1 Macro nouveau   : 0.8421
  F1 Macro ancien    : 0.8415
  Amélioration       : +0.07%
  Recall Négatif     : 0.7589
  Modèle promu       : ✅ OUI
  MLflow UI          : http://localhost:5001/#/experiments/1/runs/...
====================================================
```

**Déclencher manuellement pour la démo :**

Airflow UI → DAG `daily_sentiment_retraining` → bouton **▶ Trigger DAG**

> ⚠️ Condition : MongoDB doit contenir au moins **300 reviews** dans la collection `reviews`.
> Si vide, Task 1 s'arrête avec `300 samples minimum` — comportement normal.

### Voir les logs d'une tâche

1. Aller sur http://localhost:8082
2. Cliquer sur le DAG souhaité
3. Cliquer sur l'onglet **Graph**
4. Cliquer sur une tâche verte
5. Cliquer sur **Logs**

---

## MLflow — Tracking des expériences ML

### Accès

- URL : **http://localhost:5001**
- Aucun login requis

### Ce que MLflow enregistre (rempli automatiquement par DAG 2)

**Paramètres loggés par run :**

| Paramètre | Valeur |
|---|---|
| algorithm | LogisticRegression |
| maxIter | 60 |
| regParam | 0.05 |
| neg_threshold | 0.35 |
| n_training | nombre d'exemples utilisés |
| class_weight | N/(K*count) |

**Métriques loggées par run :**

| Métrique | Description |
|---|---|
| f1_macro | F1 Score macro (métrique principale) |
| accuracy | Accuracy globale |
| f1_negative | F1 Score classe Négatif |
| f1_neutral | F1 Score classe Neutre |
| f1_positive | F1 Score classe Positif |
| recall_negative | Recall Négatif (métrique business) |
| previous_f1 | F1 du modèle en production avant ce run |
| f1_improvement | Amélioration en % vs modèle précédent |
| train_time_seconds | Durée d'entraînement |

**Tags :**

| Tag | Valeur possible |
|---|---|
| promoted | True / False |
| decision | PROMOTED / REJECTED |

### Naviguer dans MLflow

1. Ouvrir http://localhost:5001
2. Cliquer sur l'expérience **Amazon_Sentiment_Retraining**
3. Voir la liste de tous les runs (un par nuit)
4. Cliquer sur un run pour voir :
   - Tous les paramètres et métriques
   - Le graphique d'évolution des métriques
   - Le modèle sauvegardé comme artifact
5. Comparer deux runs : sélectionner 2 runs → **Compare**

---

## Flask Dashboard — Visualisation des données

### Accès

- URL : **http://localhost:5050**
- Aucun login requis

### Pages disponibles

| Page | Description |
|---|---|
| **Accueil** (`/`) | KPIs temps réel : nombre total d'avis, distribution des sentiments, métriques du modèle |
| **Analyse Produit** (`/product`) | Statistiques détaillées par produit (recherche par ProductId) |
| **Flux Live** (`/live`) | Avis en temps réel avec prédictions de sentiment |
| **Liste de surveillance** (`/watchlist`) | Produits suivis avec alertes sur nouveaux avis |
| **Santé du modèle** (`/model-health`) | Historique des métriques ML et évolution des performances |

### APIs REST

Le dashboard expose plusieurs endpoints API pour l'intégration :

| Endpoint | Description |
|---|---|
| `/api/kpis` | KPIs globaux (total avis, % positifs/négatifs/neutres) |
| `/api/watchlist` | Liste des produits surveillés |
| `/api/model-current` | Métriques du modèle actuel |
| `/api/model-history` | Historique des réentraînements |
| `/stream` | Server-Sent Events pour le flux temps réel |

### Fonctionnalités

- **Temps réel** : Mise à jour automatique des données depuis MongoDB
- **Responsive** : Interface adaptée mobile et desktop
- **Interactive** : Graphiques et tableaux dynamiques
- **Filtrage** : Recherche par produit, date, sentiment

---

## Modèle ML

- **Algorithme** : Logistic Regression multinomiale (PySpark MLlib)
- **Features** : TF-IDF unigrammes + bigrammes (~20 000 features)
- **Split** : 80% train / 10% validation / 10% test (stratifié)
- **Class weights** : N / (K × count) — compense le déséquilibre 78% Positif
- **Threshold Négatif** : 0.35 au lieu de 0.50 par défaut
  - *Raison* : dans un système d'alerte temps réel, rater une review négative
    est plus coûteux que déclencher une fausse alerte
- **Stockage** : `models/final_best_model/`

### Labels de sentiment

| Score | Sentiment | Classe |
|---|---|---|
| 1 ou 2 | Négatif | 0 |
| 3 | Neutre | 1 |
| 4 ou 5 | Positif | 2 |

---

## Collections MongoDB

### 1. `reviews` — Données brutes + prédictions (écrit par Spark)

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

### 2. `product_metrics` — Stats par produit (écrit par Spark)

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

### 3. `model_metrics` — Performance du modèle (écrit par Spark + DAG 2)

**Document Spark (temps réel) :**
```json
{
  "accuracy": 0.87,
  "total_processed": 500,
  "correct_predictions": 435,
  "model_path": "models/final_best_model"
}
```

**Document DAG 2 (réentraînement) :**
```json
{
  "type": "retraining",
  "f1_macro": 0.8421,
  "accuracy": 0.8614,
  "recall_negative": 0.7589,
  "previous_f1": 0.8415,
  "improvement_pct": 0.07,
  "promoted": true,
  "mlflow_run_id": "a3f8b2c1d4e5f6a7",
  "mlflow_url": "http://localhost:5001/#/experiments/1/runs/a3f8b2c1d4e5f6a7",
  "run_name": "retrain_20260511_0300",
  "n_training_samples": 4552,
  "neg_threshold": 0.35,
  "retrained_at": "2026-05-11T03:02:14Z"
}
```

---

## Architecture Docker — Services

| Service | Port | Description |
|---|---|---|
| `zookeeper` | 2181 | Gestionnaire cluster Kafka |
| `kafka` | 9092 | Message broker |
| `kafka-ui` | 8080 | Interface web Kafka |
| `mongodb` | 27017 | Base de données NoSQL |
| `mongo-express` | 8081 | Interface web MongoDB |
| `producer` | — | Envoi reviews vers Kafka (Docker) |
| `spark-streaming` | — | Inférence temps réel (Docker) |
| `mlflow` | 5001 | Tracking ML |
| `postgres` | interne | Base de données Airflow |
| `airflow-webserver` | 8082 | Interface web Airflow |
| `airflow-scheduler` | — | Planificateur des DAGs |
| `airflow-init` | — | Initialisation (premier démarrage) |

---

## Architecture Docker — Images personnalisées

Le projet utilise 3 images Docker personnalisées pour les services complexes :

### Dockerfile.airflow

Image Airflow étendue avec :
- **Java 17** (requis pour PySpark)
- **Python packages** : `pymongo`, `pyspark`, `mlflow`
- **Configuration** : variables d'environnement pour Java et PySpark

### Dockerfile.producer

Image légère pour l'envoi des données :
- **Base** : `python:3.10-slim`
- **Packages** : `kafka-python`, `pandas`
- **Fichiers** : `producer.py`, `Reviews.csv`

### Dockerfile.spark

Image Spark complète pour le streaming :
- **Base** : `apache/spark:3.5.1`
- **Java 17** et dépendances Python
- **Connecteurs** : Kafka, MongoDB
- **Configuration** : UI Spark sur port 4040

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

Cause : incompatibilité Python 3.11 + PySpark. Utiliser Python 3.10 :

```powershell
conda create -n spark_env python=3.10 -y
conda activate spark_env
pip install pyspark==3.5.1
```

### Erreur `ChecksumException`

```powershell
del /s /q models\preprocessing_pipeline\metadata\.*.crc
del /s /q models\preprocessing_pipeline\stages\*.crc
del /s /q models\final_best_model\metadata\.*.crc
```

### Kafka : topic introuvable

Créer le topic sur http://localhost:8080 → Topics → **+ Add a Topic**

### MongoDB vide après lancement

Vérifier que `producer.py` et `run_spark.cmd` tournent dans leurs terminaux.

### MLflow : interface inaccessible

Vérifier que le service est lancé :

```powershell
docker-compose ps mlflow
```

Si absent, relancer :

```powershell
docker-compose up -d mlflow
```

### Dashboard Flask : port 5050 inaccessible

Vérifier que `app.py` tourne :

```powershell
netstat -ano | findstr :5050
```

Si rien, relancer :

```powershell
conda activate spark_env
python app.py
```

### Dashboard Flask : erreur de connexion MongoDB

Vérifier que MongoDB est accessible :

```powershell
docker-compose ps mongodb
```

Puis vérifier la connexion dans les logs de `app.py`.

### DAG 2 : `FileNotFoundError preprocessing_pipeline`

Cause : le volume `./models:/opt/models` n'est pas dans `docker-compose.yml`.

Solution : vérifier que `x-airflow-common` contient :

```yaml
volumes:
  - ./models:/opt/models
```

Puis relancer :

```powershell
docker-compose up -d --no-deps airflow-webserver airflow-scheduler
```

### DAG 2 : `Seulement X samples (minimum=300)`

Comportement **normal** si MongoDB est vide ou contient peu de données.
Attendre que Spark Streaming ait traité au moins 300 reviews, puis déclencher le DAG manuellement depuis l'UI Airflow.

### Airflow : DAG Import Error (module manquant)

Vérifier que `_PIP_ADDITIONAL_REQUIREMENTS` dans `docker-compose.yml` contient :

```yaml
_PIP_ADDITIONAL_REQUIREMENTS: 'pymongo pyspark==3.5.1 mlflow==2.11.0'
```

Puis relancer :

```powershell
docker-compose up -d --no-deps airflow-webserver airflow-scheduler
```

### Airflow : Invalid login

```powershell
docker exec -it amazon-reviews-sentiment-analysis-bigdata-airflow-webserver-1 airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@admin.com
```

### Airflow : DAG failed (Kafka inaccessible)

Vérifier que le DAG utilise les noms de services Docker :

```python
KAFKA_HOST = "kafka"      # ✅ correct depuis un conteneur Docker
KAFKA_HOST = "localhost"  # ❌ incorrect depuis un conteneur Docker
```

---

## Données

Source : https://www.kaggle.com/snap/amazon-fine-food-reviews

- 568 454 avis sur des produits alimentaires Amazon
- Période : plus de 10 ans
- Colonnes : Id, ProductId, UserId, ProfileName, Score, Time, Summary, Text
