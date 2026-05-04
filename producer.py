import pandas as pd
from kafka import KafkaProducer
import json
import time

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

TOPIC_NAME = 'amazon_reviews'

def start_producer():
    try:
        print("--- Chargement des données Amazon Fine Food Reviews ---")
        df = pd.read_csv('Reviews.csv')
        
        # ✅ Utiliser seulement les 10% de test (comme dans le notebook)
        # Reproduire le même split que lors de l'entraînement
        total = len(df)
        test_start = int(total * 0.90)  # 90% → début du test set
        df_test = df.iloc[test_start:].reset_index(drop=True)
        
        print(f"--- Total reviews : {total} ---")
        print(f"--- Test set : {len(df_test)} reviews ---")
        print(f"--- Début de l'envoi vers le topic : {TOPIC_NAME} ---")
        
        for index, row in df_test.iterrows():
            # Vérifier que les valeurs ne sont pas NaN
            if pd.isna(row['Text']) or pd.isna(row['ProfileName']):
                continue
                
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
                "Text": str(row['Text'])
            }
            
            producer.send(TOPIC_NAME, value=message)
            
            # Affichage de contrôle
            print(f"[PRODUCER] ✅ Avis {row['Id']} envoyé | "
                  f"Produit: {row['ProductId']} | "
                  f"Score: {row['Score']}/5")
            
            # 1 avis par seconde → simulation temps réel
            time.sleep(1)
            
        print("--- ✅ Tous les avis du test set ont été envoyés ---")
        
    except FileNotFoundError:
        print("Erreur : Le fichier 'Reviews.csv' n'est pas trouvé.")
    except Exception as e:
        print(f"Erreur : {e}")

if __name__ == "__main__":
    start_producer()