import mlflow
import numpy as np


def load_latest_model():
    # Load the latest model from MLflow
    model = mlflow.sklearn.load_model("models:/IsolationForest_model/latest")
    return model


def predict_outliers(model, review_text):
    # Feature extraction for the model (example: text length)
    feature_vector = np.array([len(review_text)]).reshape(1, -1)
    prediction = model.predict(feature_vector)[0]
    return prediction == -1
