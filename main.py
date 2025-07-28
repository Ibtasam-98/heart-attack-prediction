import os
from data_preprocessing import load_data, engineer_features, save_preprocessors
from sklearn.model_selection import train_test_split

from model import build_model, get_callbacks, cross_validate
from evaluation import evaluate_model, plot_cv_metrics, plot_learning_curves
from sklearn.utils import class_weight
import numpy as np
import joblib
import tensorflow as tf
from config import ARTIFACTS_DIR, MODEL_CONFIG


def main():
    # Load and prepare data
    X, y = load_data()
    feature_names = X.columns.tolist()

    # Split data
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=MODEL_CONFIG['test_size'],
        random_state=MODEL_CONFIG['random_state'], stratify=y
    )

    # Feature engineering
    X_train_scaled, X_test_scaled, poly, quantile, selector, scaler, robust_scaler, selected_indices = \
        engineer_features(X_train_raw.values, X_test_raw.values, y_train, feature_names)

    # Cross-validation
    print("\n--- Cross-Validation Results ---")
    cv_individual_fold_metrics, cv_mean_metrics = cross_validate(X_train_scaled, y_train)
    print("\nCV Results (Mean across folds):")
    for name, value in cv_mean_metrics.items():
        print(f"{name:<10} {value:<10.4f}")

    plot_cv_metrics(cv_individual_fold_metrics)

    # Final model training
    print("\n--- Final Model Training ---")
    model = build_model(X_train_scaled.shape[1])

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class weights for Final Model: {class_weight_dict}")

    history = model.fit(
        X_train_scaled, y_train,
        epochs=MODEL_CONFIG['epochs'],
        batch_size=MODEL_CONFIG['batch_size'],
        validation_split=MODEL_CONFIG['validation_split'],
        callbacks=get_callbacks(),
        verbose=1,
        class_weight=class_weight_dict
    )

    # Evaluation
    print("\n--- Evaluating Final Model on Test Set ---")
    final_metrics = evaluate_model(model, X_test_scaled, y_test)
    print("\n--- Final Test Metrics ---")
    for name, value in final_metrics.items():
        print(f"{name:<10} {value:<10.4f}")

    plot_learning_curves(history)

    # Save artifacts
    print(f"\nSaving model and preprocessors to '{ARTIFACTS_DIR}' directory...")
    model.save(os.path.join(ARTIFACTS_DIR, 'heart_attack_model.h5'))
    save_preprocessors(poly, quantile, selector, scaler, robust_scaler, selected_indices, feature_names, final_metrics)
    print("Training completed successfully!")


if __name__ == "__main__":
    main()