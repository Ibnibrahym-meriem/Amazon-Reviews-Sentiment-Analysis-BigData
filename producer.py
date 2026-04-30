import pandas as pd
from kafka import KafkaProducer
import json
import time

# 1. Configuration du Producer Kafka
# 'localhost:9092' est le port exposé par votre conteneur Docker Kafka
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

# Nom du topic défini dans votre architecture [cite: 16]
TOPIC_NAME = 'amazon_reviews'

def start_producer():
    try:
        print("--- Chargement des données Amazon Fine Food Reviews ---")
        # Lecture du fichier CSV collecté sur Kaggle 
        # On utilise chunksize si le fichier est trop lourd (500 000 avis) [cite: 94]
        df = pd.read_csv('Reviews.csv')
        
        print(f"--- Début de l'envoi vers le topic : {TOPIC_NAME} ---")

        for index, row in df.iterrows():
            # Construction du message JSON respectant strictement vos données [cite: 97, 98, 99, 100, 101, 102]
            message = {
                "Id": int(row['Id']),
                "ProductId": str(row['ProductId']),
                "UserId": str(row['UserId']),
                "ProfileName": str(row['ProfileName']),
                "HelpfulnessNumerator": int(row['HelpfulnessNumerator']),
                "HelpfulnessDenominator": int(row['HelpfulnessDenominator']),
                "Score": int(row['Score']),
                "Time": int(row['Time']),
                "Summary": str(row['Summary']),
                "Text": str(row['Text']) # Contenu complet de l'avis (texte brut) [cite: 102]
            }

            # Envoi du message (Push) vers le broker Kafka [cite: 22]
            producer.send(TOPIC_NAME, value=message)
            
            # Affichage de contrôle tous les 10 messages
            if index % 10 == 0:
                print(f"[PRODUCER] Avis {row['Id']} envoyé (Produit: {row['ProductId']})")
            
            # Simulation du flux temps réel (pause de 1 seconde entre chaque avis) [cite: 6]
            time.sleep(1)

    except FileNotFoundError:
        print("Erreur : Le fichier 'Reviews.csv' n'est pas dans le dossier actuel.")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")

if __name__ == "__main__":
    start_producer()