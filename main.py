import os
from data_preprocessing import load_data, engineer_features, save_preprocessors
from sklearn.model_selection import train_test_split
from model import build_model, get_callbacks, cross_validate
from evaluation import evaluate_model, plot_cv_metrics, plot_learning_curves, print_comprehensive_metrics
from sklearn.utils import class_weight
import numpy as np
import joblib
import time
from config import ARTIFACTS_DIR, MODEL_CONFIG


def main():
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


if __name__ == "__main__":
    main()
