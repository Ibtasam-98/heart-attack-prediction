import os
import pandas as pd
import numpy as np
from data_preprocessing import load_data, engineer_features, save_preprocessors
from sklearn.model_selection import train_test_split
from model import build_model, get_callbacks, cross_validate
from evaluation import evaluate_model, plot_cv_metrics, plot_learning_curves, print_comprehensive_metrics
from sklearn.utils import class_weight
import joblib
import time
from config import ARTIFACTS_DIR, MODEL_CONFIG


def train_model():
    """Train the heart attack prediction model"""
    print("STARTING HEART ATTACK PREDICTION MODEL TRAINING")
    print("=" * 70)

    # Load and prepare data
    X, y = load_data()
    feature_names = X.columns.tolist()

    # Print dataset characteristics
    print("\n" + "=" * 60)
    print("DATASET CHARACTERISTICS AND CLASS DISTRIBUTION")
    print("=" * 60)
    print(f"Total samples: {len(X)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Feature names: {feature_names}")
    print(f"\nClass Distribution:")
    class_counts = y.value_counts()
    class_percentages = y.value_counts(normalize=True) * 100
    for class_val, count in class_counts.items():
        percentage = class_percentages[class_val]
        class_name = "Positive" if class_val == 1 else "Negative"
        print(f"  {class_name} (Class {class_val}): {count} samples ({percentage:.2f}%)")

    print(f"\nData Types:")
    print(X.dtypes)
    print(f"\nBasic Statistics:")
    print(X.describe())

    # Split data
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=MODEL_CONFIG['test_size'],
        random_state=MODEL_CONFIG['random_state'], stratify=y
    )

    print(f"\nTraining set size: {len(X_train_raw)} samples")
    print(f"Test set size: {len(X_test_raw)} samples")

    # Feature engineering
    X_train_scaled, X_test_scaled, poly, quantile, selector, scaler, robust_scaler, selected_indices = \
        engineer_features(X_train_raw.values, X_test_raw.values, y_train, feature_names)

    # Cross-validation
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 60)
    cv_individual_fold_metrics, cv_mean_metrics, fold_times = cross_validate(X_train_scaled, y_train)

    print("\nCROSS-VALIDATION SUMMARY (Mean across folds):")
    print("-" * 50)
    for name, value in cv_mean_metrics.items():
        if name in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            print(f"{name.upper():<12} {value:<10.4f}")

    print(f"\nTraining Time Analysis:")
    print(f"  Average Training Time per fold: {np.mean(fold_times):.2f} seconds")
    print(f"  Total CV Training Time: {np.sum(fold_times):.2f} seconds")
    print(f"  Fastest fold: {np.min(fold_times):.2f} seconds")
    print(f"  Slowest fold: {np.max(fold_times):.2f} seconds")

    plot_cv_metrics(cv_individual_fold_metrics, fold_times)

    # Final model training
    print("\n" + "=" * 60)
    print("FINAL MODEL TRAINING")
    print("=" * 60)
    model = build_model(X_train_scaled.shape[1])

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class weights for Final Model: {class_weight_dict}")

    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    print(f"Final model training completed in {training_time:.2f} seconds")

    # Evaluation
    print("\n" + "=" * 60)
    print("FINAL MODEL EVALUATION ON TEST SET")
    print("=" * 60)
    final_metrics = evaluate_model(model, X_test_scaled, y_test, training_time)

    # Print comprehensive metrics
    print_comprehensive_metrics(final_metrics, cv_mean_metrics)

    plot_learning_curves(model)

    # Save artifacts
    print(f"\nSaving model and preprocessors to '{ARTIFACTS_DIR}' directory...")
    joblib.dump(model, os.path.join(ARTIFACTS_DIR, 'heart_attack_model.pkl'))
    save_preprocessors(poly, quantile, selector, scaler, robust_scaler, selected_indices, feature_names, final_metrics)

    print("Training completed successfully!")
    return model, feature_names


def make_prediction(model, feature_names):
    """Make predictions using the trained model"""
    print("\n" + "=" * 70)
    print("🎯 HEART ATTACK RISK PREDICTION")
    print("=" * 70)

    # Load label encoders for categorical data
    label_encoders = joblib.load(os.path.join(ARTIFACTS_DIR, 'label_encoders.pkl'))

    print(f"Required Features: {len(feature_names)}")
    print("\nPlease enter the following health metrics:")
    print("-" * 50)

    user_data = {}
    for feature in feature_names:
        while True:
            try:
                if feature in label_encoders:
                    # Handle categorical features
                    encoder = label_encoders[feature]
                    mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
                    print(f"\n{feature} options: {mapping}")
                    value = input(f"Enter {feature} ({'/'.join(mapping.keys())}): ").strip()

                    if value in mapping:
                        user_data[feature] = mapping[value]
                        break
                    else:
                        print(f"Invalid option. Please choose from: {list(mapping.keys())}")
                else:
                    # Handle numerical features
                    value = float(input(f"Enter {feature}: "))
                    user_data[feature] = value
                    break

            except ValueError:
                print("Please enter a valid number.")
            except KeyboardInterrupt:
                print("\n\nPrediction cancelled.")
                return None

    return user_data


def predict_single_user(model, user_data, feature_names):
    """Predict heart attack risk for a single user"""
    try:
        # Load preprocessors
        poly = joblib.load(os.path.join(ARTIFACTS_DIR, 'poly_transformer.pkl'))
        quantile = joblib.load(os.path.join(ARTIFACTS_DIR, 'quantile_transformer.pkl'))
        selector = joblib.load(os.path.join(ARTIFACTS_DIR, 'selector_kbest.pkl'))
        scaler = joblib.load(os.path.join(ARTIFACTS_DIR, 'standard_scaler.pkl'))
        robust_scaler = joblib.load(os.path.join(ARTIFACTS_DIR, 'robust_scaler.pkl'))
        selected_indices = joblib.load(os.path.join(ARTIFACTS_DIR, 'selected_indices.pkl'))

        # Convert to DataFrame with correct column order
        user_df = pd.DataFrame([user_data])[feature_names]

        # Apply all transformations
        user_poly = poly.transform(user_df)
        user_qt = quantile.transform(user_poly)
        user_selected = user_qt[:, selected_indices]
        user_scaled_temp = scaler.transform(user_selected)
        user_scaled = robust_scaler.transform(user_scaled_temp)

        # Make prediction
        prediction_proba = model.predict_proba(user_scaled)
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


def display_prediction_result(result):
    """Display prediction results in a user-friendly format"""
    if 'error' in result:
        print(f"Error: {result['error']}")
        return

    print("\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)
    print(f" Heart Attack Risk Probability: {result['probability']:.4f} ({result['probability'] * 100:.2f}%)")
    print(f" Prediction: {result['class'].upper()}")
    print(f" Risk Level: {result['risk_level']}")
    print(f" Recommendation: {result['recommendation']}")
    print(f" Confidence: {result['confidence']}")

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


def main():
    """Main function to run the complete workflow"""
    try:
        # Train the model
        model, feature_names = train_model()

        # Ask user if they want to make predictions
        while True:
            print("\n" + "=" * 70)
            response = input("\nDo you want to make a heart attack risk prediction? (yes/no): ").strip().lower()

            if response in ['yes', 'y']:
                # Get user input
                user_data = make_prediction(model, feature_names)

                if user_data is None:  # User cancelled
                    break

                # Make prediction
                print("\n" + "=" * 50)
                print("ANALYZING YOUR DATA...")
                print("=" * 50)

                result = predict_single_user(model, user_data, feature_names)
                display_prediction_result(result)

            elif response in ['no', 'n']:
                print("\nThank you for using the Heart Attack Prediction System!")
                break
            else:
                print("Please enter 'yes' or 'no'")

    except Exception as e:
        print(f"Error in main execution: {e}")
        print("Make sure you have the dataset in the correct location")


if __name__ == "__main__":
    main()
