import os

import pandas as pd
import joblib
import numpy as np
from config import ARTIFACTS_DIR
import tensorflow as tf

class HeartAttackPredictor:
    def __init__(self):
        self.model = tf.keras.models.load_model(os.path.join(ARTIFACTS_DIR, 'heart_attack_model.h5'))
        self.poly = joblib.load(os.path.join(ARTIFACTS_DIR, 'poly_transformer.pkl'))
        self.quantile = joblib.load(os.path.join(ARTIFACTS_DIR, 'quantile_transformer.pkl'))
        self.selector = joblib.load(os.path.join(ARTIFACTS_DIR, 'selector_kbest.pkl'))
        self.scaler = joblib.load(os.path.join(ARTIFACTS_DIR, 'standard_scaler.pkl'))
        self.robust_scaler = joblib.load(os.path.join(ARTIFACTS_DIR, 'robust_scaler.pkl'))
        self.selected_indices = joblib.load(os.path.join(ARTIFACTS_DIR, 'selected_indices.pkl'))
        self.feature_names = joblib.load(os.path.join(ARTIFACTS_DIR, 'feature_names.pkl'))

    def predict(self, input_data):
        # Convert to DataFrame
        user_df = pd.DataFrame([input_data])

        # Apply all transformations
        user_poly = self.poly.transform(user_df)
        user_qt = self.quantile.transform(user_poly)
        user_selected = user_qt[:, self.selected_indices]
        user_scaled_temp = self.scaler.transform(user_selected)
        user_scaled = self.robust_scaler.transform(user_scaled_temp)

        # Make prediction
        prediction_proba = self.model.predict(user_scaled)[0][0]
        prediction_class = "positive" if prediction_proba > 0.5 else "negative"

        return {
            'probability': float(prediction_proba),
            'class': prediction_class
        }


def predict_from_cli():
    predictor = HeartAttackPredictor()
    print("\n=== Heart Attack Risk Prediction ===")
    print("Please enter the following health metrics:")

    user_data = {}
    for feature in predictor.feature_names:
        while True:
            try:
                value = float(input(f"Enter {feature}: "))
                user_data[feature] = value
                break
            except ValueError:
                print("Please enter a valid number.")

    result = predictor.predict(user_data)

    print("\n=== Prediction Result ===")
    print(f"Heart Attack Risk Probability: {result['probability']:.4f}")
    print(f"Prediction: {result['class']}")

    if result['probability'] > 0.5:
        print("Warning: High risk of heart attack detected. Please consult a doctor.")
    else:
        print("Low risk of heart attack detected. Maintain a healthy lifestyle.")


if __name__ == "__main__":
    predict_from_cli()