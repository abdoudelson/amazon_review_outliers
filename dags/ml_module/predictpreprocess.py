import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sentence_transformers import SentenceTransformer
import logging
from colorlog import ColoredFormatter

class PredictPreprocessor:
    def __init__(self):
        self.logger = self.setup_logger()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.scaler = StandardScaler()  # Initialiser sans fit pour les prédictions
        self.label_encoder = LabelEncoder()  # Initialiser sans fit pour les prédictions
        self.numerical_features = ['rating', 'helpful_vote']
        self.categorical_features = ['verified_purchase']

    def setup_logger(self):
        formatter = ColoredFormatter(
            "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "white",
                "INFO": "cyan",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            }
        )
        logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def load_transformers(self, scaler_path, label_encoder_path):
        try:
            self.logger.info('Loading transformers for prediction...')
            self.scaler = pd.read_pickle(scaler_path)
            self.label_encoder = pd.read_pickle(label_encoder_path)
            self.logger.info('Transformers loaded successfully.')
        except Exception as e:
            self.logger.error("Error loading transformers: %s", e)
            raise

    def preprocess(self, data):
        """
        Prétraite les données de prédiction en appliquant les transformations
        utilisées durant l'entraînement.
        
        Paramètres:
        - data (dict): Un dictionnaire avec les clés "rating", "helpful_vote", "verified_purchase", "text".

        Retourne:
        - pd.DataFrame: Un DataFrame avec les données transformées.
        """
        try:
            self.logger.info('Preprocessing data for prediction...')

            # Crée un DataFrame à partir des données d'entrée
            data_df = pd.DataFrame([data])

            # Appliquer le scaling pour les caractéristiques numériques
            self.logger.info('Applying scaler to numerical features...')
            data_df[self.numerical_features] = self.scaler.transform(data_df[self.numerical_features])

            # Encoder les caractéristiques catégorielles
            self.logger.info('Encoding categorical features...')
            for col in self.categorical_features:
                data_df[col] = self.label_encoder.transform(data_df[col].astype(str))

            # Générer les embeddings de texte
            self.logger.info('Generating text embeddings...')
            text_embedding = self.model.encode(data_df['text'].tolist(), show_progress_bar=False)
            embedded_df = pd.DataFrame(text_embedding)

            # Combiner les embeddings de texte avec les autres caractéristiques
            other_features = data_df[self.numerical_features + self.categorical_features].reset_index(drop=True)
            processed_data = pd.concat([embedded_df, other_features], axis=1)

            self.logger.info('Data preprocessed successfully for prediction.')
            return processed_data
        except Exception as e:
            self.logger.error("Error during prediction preprocessing: %s", e)
            raise
if __name__ == "__main__":
    # Initialise la classe
    preprocessor = PredictPreprocessor()
    
    # Charger les transformations entraînées
    preprocessor.load_transformers('scaler.pkl', 'label_encoder.pkl')
    
    # Exemple de données de prédiction
    sample_data = {
        "rating": 4.0,
        "helpful_vote": 20,
        "verified_purchase": "Yes",
        "text": "Great product, really enjoyed it!"
    }

    # Prétraitement des données de prédiction
    processed_data = preprocessor.preprocess(sample_data)
    print(processed_data)