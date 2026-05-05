# Amazon Reviews Sentiment Analysis — Big Data

Analyse des avis clients Amazon en temps réel avec Kafka, Spark Streaming, MongoDB et ML.

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
```

---

## Technologies

| Technologie | Rôle |
|---|---|
| Apache Kafka | Collecte flux temps réel |
| Zookeeper | Gestion cluster Kafka |
| Apache Spark (PySpark) | Traitement distribué + ML |
| MongoDB | Stockage NoSQL des résultats |
| Docker | Conteneurisation des services |
| Python 3.11 | Développement |
| Java 17 | Requis pour Spark |

---

## Prérequis

Installer dans cet ordre :

1. **Docker Desktop** → https://www.docker.com/products/docker-desktop
2. **Python 3.11** → https://www.python.org/downloads/release/python-3110/
3. **Java 17 (Temurin)** → https://adoptium.net/temurin/releases/?version=17&os=windows&arch=x64&package=jdk

### Packages Python requis

```bash
pip install pyspark==4.1.1
pip install kafka-python
pip install pandas
```

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

### Étape 3 — Créer le fichier `run_spark.cmd` (Windows)

Créer un fichier `run_spark.cmd` à la racine du projet :

```cmd
@echo off
set JAVA_HOME=C:\Program Files\Eclipse Adoptium\jdk-17.0.19.10-hotspot
set PATH=%JAVA_HOME%\bin;%PATH%
set JAVA_TOOL_OPTIONS=--add-opens=java.base/javax.security.auth=ALL-UNNAMED --add-opens=java.base/java.lang=ALL-UNNAMED --add-opens=java.base/java.net=ALL-UNNAMED --add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.util.concurrent=ALL-UNNAMED --add-opens=java.base/java.text=ALL-UNNAMED --add-opens=java.sql/java.sql=ALL-UNNAMED

echo Java version utilisee :
java -version
echo Lancement de Spark Streaming...
python spark_streaming.py
```

> Adapter le chemin `JAVA_HOME` selon ton installation Java 17.

### Étape 4 — Lancer les services Docker

```bash
docker-compose up
```

Vérifier que ces services sont Running :
- Zookeeper → port 2181
- Kafka → port 9092
- MongoDB → port 27017
- Mongo Express → http://localhost:8081
- Kafka UI → http://localhost:8080

### Étape 5 — Créer le topic Kafka

Aller sur http://localhost:8080 → Topics → **+ Add a Topic**

- Topic Name : `amazon_reviews`
- Partitions : `1`
- Replication Factor : `1`

### Étape 6 — Lancer Spark Streaming

Ouvrir un nouveau terminal :

```bash
run_spark.cmd
```

Attendre que les JARs se téléchargent (première fois ~2 minutes).

### Étape 7 — Lancer le Producer

Ouvrir un autre terminal :

```bash
python producer.py
```

Tu verras :
```
[PRODUCER] ✅ Avis 511683 envoyé | Produit: B000FFLXPG | Score: 5/5
[PRODUCER] ✅ Avis 511684 envoyé | Produit: B000FFLXPG | Score: 4/5
...
```

### Étape 8 — Vérifier dans MongoDB

Ouvrir http://localhost:8081

Tu dois voir dans `amazon_db` :

| Collection | Description |
|---|---|
| `reviews` | Avis + prédictions en temps réel |
| `product_metrics` | Stats agrégées par produit |
| `model_metrics` | Accuracy du modèle |

---

## Structure du projet

```
Amazon-Reviews-Sentiment-Analysis-BigData/
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
├── docker-compose.yml             ← Configuration Docker
├── run_spark.cmd                  ← Lancement Spark (Windows)
└── README.md
```

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

### Erreur `ChecksumException`

```bash
del /s /q models\preprocessing_pipeline\metadata\.*.crc
del /s /q models\preprocessing_pipeline\stages\*.crc
del /s /q models\final_best_model\metadata\.*.crc
```

### Kafka : topic introuvable

Créer le topic sur http://localhost:8080 → Topics → **+ Add a Topic**

### MongoDB vide après lancement

Vérifier que `producer.py` tourne dans un terminal séparé.

---

## Équipe

Projet réalisé dans le cadre du cours Big Data — IASD 2025-2026

Présentation : **lundi 11/05/2026**