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
        self.label_encoders = joblib.load(os.path.join(ARTIFACTS_DIR, 'label_encoders.pkl'))

        # Load model
        model_path = os.path.join(ARTIFACTS_DIR, 'heart_attack_model.pkl')
        print(f"Loading model from {model_path}...")
        self.model = joblib.load(model_path)
        self.model_type = 'Scikit-learn MLP'

        print(f"Successfully loaded {self.model_type} model")
        print(f"Available features: {self.feature_names}")
        if self.label_encoders:
            print(f"Categorical encodings: {self.label_encoders}")

    def preprocess_input(self, input_data):
        """Preprocess user input similar to training pipeline"""
        # Convert to DataFrame with correct column order
        user_df = pd.DataFrame([input_data])[self.feature_names]

        # Encode categorical variables
        for col, encoder in self.label_encoders.items():
            if col in user_df.columns:
                # Handle both string and numeric inputs
                if user_df[col].dtype == 'object':
                    user_df[col] = encoder.transform(user_df[col])
                else:
                    # If numeric, assume it's already encoded but validate
                    unique_values = set(encoder.classes_)
                    if user_df[col].iloc[0] not in range(len(unique_values)):
                        raise ValueError(f"Invalid value for {col}. Expected values: {list(encoder.classes_)}")

        return user_df

    def predict(self, input_data):

        try:
            # Preprocess input
            user_df = self.preprocess_input(input_data)

            # Apply all transformations
            user_poly = self.poly.transform(user_df)
            user_qt = self.quantile.transform(user_poly)
            user_selected = user_qt[:, self.selected_indices]
            user_scaled_temp = self.scaler.transform(user_selected)
            user_scaled = self.robust_scaler.transform(user_scaled_temp)

            # Make prediction
            print("Making prediction...")
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
        feature_info = {}
        for feature in self.feature_names:
            if feature in self.label_encoders:
                encoder = self.label_encoders[feature]
                feature_info[feature] = {
                    'type': 'categorical',
                    'mapping': dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
                }
            else:
                feature_info[feature] = {'type': 'numerical'}

        return {
            'features': feature_info,
            'model_type': self.model_type,
            'total_features': len(self.feature_names)
        }


def predict_from_cli():
    """Command line interface for prediction"""
    try:
        predictor = HeartAttackPredictor()
        feature_info = predictor.get_feature_info()

        print("\n" + "=" * 70)
        print("HEART ATTACK RISK PREDICTION SYSTEM")
        print("=" * 70)
        print(f"Model Type: {predictor.model_type}")
        print(f"Required Features: {len(predictor.feature_names)}")
        print("\nPlease enter the following health metrics:")
        print("-" * 50)

        user_data = {}
        for feature, info in feature_info['features'].items():
            while True:
                try:
                    if info['type'] == 'categorical':
                        print(f"\n{feature} options: {info['mapping']}")
                        value = input(f"Enter {feature} ({'/'.join(info['mapping'].keys())}): ").strip()
                        # Convert to proper encoding
                        if value in info['mapping']:
                            user_data[feature] = info['mapping'][value]
                            break
                        else:
                            print(f"Invalid option. Please choose from: {list(info['mapping'].keys())}")
                    else:
                        value = float(input(f"Enter {feature}: "))
                        user_data[feature] = value
                        break
                except ValueError:
                    print("Please enter a valid number.")
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
        print("PREDICTION RESULTS")
        print("=" * 70)
        print(f"Model Used: {result['model_type']}")
        print(f"Heart Attack Risk Probability: {result['probability']:.4f} ({result['probability'] * 100:.2f}%)")
        print(f"Prediction: {result['class'].upper()}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Recommendation: {result['recommendation']}")
        print(f"Confidence: {result['confidence']}")

        # Detailed interpretation
        print("\n" + "=" * 70)
        print("DETAILED INTERPRETATION")
        print("=" * 70)
        if result['class'] == 'positive':
            print("HIGH RISK DETECTED")
            print("   - Probability of heart attack: {:.2f}%".format(result['probability'] * 100))
            print("   - Immediate medical attention recommended")
            print("   - Contact healthcare provider")
        else:
            print("LOW RISK DETECTED")
            print("   - Probability of heart attack: {:.2f}%".format(result['probability'] * 100))
            print("   - Continue maintaining healthy lifestyle")
            print("   - Regular checkups recommended")

        print(f"\nNote: This prediction is based on machine learning analysis.")
        print(f"   Always consult with healthcare professionals for medical decisions.")

    except Exception as e:
        print(f"Error initializing predictor: {e}")
        print("Make sure the model has been trained first by running main.py")


if __name__ == "__main__":
    predict_from_cli()
