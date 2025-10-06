import os
import pandas as pd
import joblib
import numpy as np
from config import ARTIFACTS_DIR


class HeartAttackPredictor:
    def __init__(self, model_type='auto'):
        """
        Initialize HeartAttackPredictor

        Args:
            model_type: 'auto', 'sklearn_mlp', or 'deep_nn'
        """
        # Load preprocessors
        print("Loading preprocessors...")
        self.poly = joblib.load(os.path.join(ARTIFACTS_DIR, 'poly_transformer.pkl'))
        self.quantile = joblib.load(os.path.join(ARTIFACTS_DIR, 'quantile_transformer.pkl'))
        self.selector = joblib.load(os.path.join(ARTIFACTS_DIR, 'selector_kbest.pkl'))
        self.scaler = joblib.load(os.path.join(ARTIFACTS_DIR, 'standard_scaler.pkl'))
        self.robust_scaler = joblib.load(os.path.join(ARTIFACTS_DIR, 'robust_scaler.pkl'))
        self.selected_indices = joblib.load(os.path.join(ARTIFACTS_DIR, 'selected_indices.pkl'))
        self.feature_names = joblib.load(os.path.join(ARTIFACTS_DIR, 'feature_names.pkl'))

        # Auto-detect models type
        if model_type == 'auto':
            if os.path.exists(os.path.join(ARTIFACTS_DIR, 'heart_attack_model.h5')):
                model_type = 'deep_nn'
                model_path = os.path.join(ARTIFACTS_DIR, 'heart_attack_model.h5')
            else:
                model_type = 'sklearn_mlp'
                model_path = os.path.join(ARTIFACTS_DIR, 'heart_attack_model.pkl')
        else:
            if model_type == 'deep_nn':
                model_path = os.path.join(ARTIFACTS_DIR, 'heart_attack_model.h5')
            else:
                model_path = os.path.join(ARTIFACTS_DIR, 'heart_attack_model.pkl')

        # Load models
        print(f"Loading {model_type.upper()} models from {model_path}...")
        if model_type == 'deep_nn':
            import tensorflow as tf
            self.model = tf.keras.models.load_model(model_path)
            self.model_type = 'Deep Neural Network'
        else:
            self.model = joblib.load(model_path)
            self.model_type = 'Scikit-learn MLP'

        print(f"✅ Successfully loaded {self.model_type} models")
        print(f"✅ Available features: {self.feature_names}")

    def predict(self, input_data):
        """
        Make heart attack risk prediction

        Args:
            input_data: Dictionary of feature values

        Returns:
            Dictionary with prediction results
        """
        try:
            # Convert to DataFrame
            user_df = pd.DataFrame([input_data])

            # Validate input features
            missing_features = set(self.feature_names) - set(user_df.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")

            # Apply all transformations
            user_poly = self.poly.transform(user_df)
            user_qt = self.quantile.transform(user_poly)
            user_selected = user_qt[:, self.selected_indices]
            user_scaled_temp = self.scaler.transform(user_selected)
            user_scaled = self.robust_scaler.transform(user_scaled_temp)

            # Make prediction
            print("Making prediction...")

            if self.model_type == 'Deep Neural Network':
                # TensorFlow models
                prediction_proba = self.model.predict(user_scaled, verbose=0)
                probability = float(prediction_proba[0][0])
            else:
                # Scikit-learn models
                prediction_proba = self.model.predict_proba(user_scaled)
                probability = float(prediction_proba[0][1])

            prediction_class = "positive" if probability > 0.5 else "negative"

            # Risk level classification
            if probability < 0.3:
                risk_level = "Low"
                recommendation = "Maintain healthy lifestyle"
            elif probability < 0.7:
                risk_level = "Medium"
                recommendation = "Consult doctor for checkup"
            else:
                risk_level = "High"
                recommendation = "Immediate medical consultation recommended"

            return {
                'probability': probability,
                'class': prediction_class,
                'risk_level': risk_level,
                'recommendation': recommendation,
                'model_type': self.model_type,
                'confidence': 'High' if (probability > 0.8 or probability < 0.2) else 'Medium'
            }

        except Exception as e:
            return {
                'error': str(e),
                'probability': None,
                'class': 'error',
                'risk_level': 'Unknown',
                'recommendation': 'Please check input data'
            }

    def get_feature_info(self):
        """Get information about required features"""
        return {
            'features': self.feature_names,
            'model_type': self.model_type,
            'total_features': len(self.feature_names)
        }


def predict_from_cli():
    """Command line interface for prediction"""
    try:
        predictor = HeartAttackPredictor()

        print("\n" + "=" * 70)
        print("🎯 HEART ATTACK RISK PREDICTION SYSTEM")
        print("=" * 70)
        print(f"Model Type: {predictor.model_type}")
        print(f"Required Features: {len(predictor.feature_names)}")
        print("\nPlease enter the following health metrics:")
        print("-" * 50)

        user_data = {}
        for feature in predictor.feature_names:
            while True:
                try:
                    value = float(input(f"Enter {feature}: "))
                    user_data[feature] = value
                    break
                except ValueError:
                    print("❌ Please enter a valid number.")
                except KeyboardInterrupt:
                    print("\n\nPrediction cancelled.")
                    return

        print("\n" + "=" * 50)
        print("ANALYZING YOUR DATA...")
        print("=" * 50)

        result = predictor.predict(user_data)

        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            return

        print("\n" + "=" * 70)
        print("📊 PREDICTION RESULTS")
        print("=" * 70)
        print(f"🤖 Model Used: {result['model_type']}")
        print(f"📈 Heart Attack Risk Probability: {result['probability']:.4f}")
        print(f"🔍 Prediction: {result['class'].upper()}")
        print(f"⚠️  Risk Level: {result['risk_level']}")
        print(f"💡 Recommendation: {result['recommendation']}")
        print(f"🎯 Confidence: {result['confidence']}")

        # Detailed interpretation
        print("\n" + "=" * 70)
        print("📋 DETAILED INTERPRETATION")
        print("=" * 70)
        if result['class'] == 'positive':
            print("🔴 HIGH RISK DETECTED")
            print("   - Probability of heart attack: {:.2f}%".format(result['probability'] * 100))
            print("   - Immediate medical attention recommended")
            print("   - Contact healthcare provider")
        else:
            print("🟢 LOW RISK DETECTED")
            print("   - Probability of heart attack: {:.2f}%".format(result['probability'] * 100))
            print("   - Continue maintaining healthy lifestyle")
            print("   - Regular checkups recommended")

        print(f"\n💡 Note: This prediction is based on machine learning analysis.")
        print(f"   Always consult with healthcare professionals for medical decisions.")

    except Exception as e:
        print(f"❌ Error initializing predictor: {e}")
        print("💡 Make sure the models has been trained first by running main.py")


if __name__ == "__main__":
    predict_from_cli()
