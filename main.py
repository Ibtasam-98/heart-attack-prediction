import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PolynomialFeatures, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.regularizers import l1_l2
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import os
import joblib # For saving scikit-learn objects

# Define directory for saving model and preprocessors
ARTIFACTS_DIR = 'model_artifacts'
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Create a directory for plots if it doesn't exist
PLOTS_DIR = 'model_plots'
os.makedirs(PLOTS_DIR, exist_ok=True)


# Enhanced data loading with improved outlier handling
def load_data():
    data = pd.read_csv('dataset.csv')
    data['Result'] = data['Result'].map({'negative': 0, 'positive': 1})

    print("Data Description:\n", data.describe())
    print("\nClass Distribution:\n", data['Result'].value_counts(normalize=True))

    # Handle outliers using IQR method
    numeric_cols = data.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)

    X = data.drop('Result', axis=1)
    y = data['Result']
    return X, y

# Advanced feature engineering
def engineer_features(X_train, X_test, y_train, feature_names):
    # Convert to DataFrame to maintain column names for PolynomialFeatures
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    # Add Polynomial Features
    poly = PolynomialFeatures(degree=2, include_bias=False) # Degree 2 is a good start
    X_train_poly = poly.fit_transform(X_train_df)
    X_test_poly = poly.transform(X_test_df)

    print(f"Original features: {X_train.shape[1]}, Polynomial features: {X_train_poly.shape[1]}")

    # Quantile transformation for non-linear relationships
    # Apply after polynomial features
    quantile = QuantileTransformer(output_distribution='normal', random_state=42)
    X_train_qt = quantile.fit_transform(X_train_poly)
    X_test_qt = quantile.transform(X_test_poly)

    # Mutual information for feature selection
    # Adjust k if polynomial features drastically increase feature count
    selector = SelectKBest(mutual_info_classif, k='all') # Start with 'all' to inspect scores
    selector.fit(X_train_qt, y_train)

    # Get top 80% of features after polynomial transformation and quantile
    # You might want to experiment with this percentage
    k = int(0.80 * X_train_qt.shape[1])
    # If k becomes 0 for some reason (e.g., very few original features), set a minimum
    if k == 0 and X_train_qt.shape[1] > 0:
        k = 1
    elif k == 0: # If X_train_qt.shape[1] is also 0, this indicates an issue
        raise ValueError("No features available after transformations.")

    selected_features_indices = np.argsort(selector.scores_)[-k:]
    X_train_selected = X_train_qt[:, selected_features_indices]
    X_test_selected = X_test_qt[:, selected_features_indices]

    print(f"Features after selection: {X_train_selected.shape[1]}")

    # Robust scaling (less sensitive to outliers)
    scaler = StandardScaler()
    X_train_scaled_temp = scaler.fit_transform(X_train_selected)
    X_test_scaled_temp = scaler.transform(X_test_selected)

    # Optional: Add RobustScaler for another layer of robust scaling if needed
    robust_scaler = RobustScaler()
    X_train_scaled = robust_scaler.fit_transform(X_train_scaled_temp)
    X_test_scaled = robust_scaler.transform(X_test_scaled_temp)

    # Return all necessary preprocessors and selected indices
    return X_train_scaled, X_test_scaled, poly, quantile, selector, scaler, robust_scaler, selected_features_indices


# Optimized model architecture with slight modifications
def build_optimized_model(input_shape):
    model = Sequential([
        Dense(512, activation='gelu', input_shape=(input_shape,),
              kernel_regularizer=l1_l2(l1=0.0001, l2=0.001)), # Reduced regularization
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='gelu', kernel_regularizer=l1_l2(l1=0.0001, l2=0.001)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation='gelu'),
        BatchNormalization(),
        Dropout(0.3), # Added another dropout layer
        Dense(64, activation='gelu'), # Added another dense layer
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])

    optimizer = Adam(learning_rate=0.0001, clipnorm=1.0) # Keep a low learning rate
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy',
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall'),
                           tf.keras.metrics.AUC(name='auc')])
    return model

# Enhanced evaluation (now saves plot instead of showing directly)
def evaluate_model(model, X_test, y_test):
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    y_proba = model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5)) # Create a new figure for the confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    # Save the plot instead of showing it
    plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'))
    plt.close() # Close the plot to free memory

    print(f"Confusion Matrix saved to {os.path.join(PLOTS_DIR, 'confusion_matrix.png')}")


    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    return metrics

# Cross-validation training
def cross_validate(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) # Added shuffle and random_state
    cv_metrics = []

    # Compute class weights for imbalanced dataset
    class_weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y), y=y
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class weights for CV: {class_weight_dict}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = build_optimized_model(X_train.shape[1])

        callbacks_cv = [
            EarlyStopping(monitor='val_auc', patience=25, mode='max', restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=15, mode='max', min_lr=1e-6, verbose=0),
            TerminateOnNaN()
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=200, # Increased epochs for CV
            batch_size=32,
            verbose=0, # Set to 0 for less verbose CV output
            callbacks=callbacks_cv,
            class_weight=class_weight_dict
        )
        # We collect metrics for CV but do NOT plot confusion matrix for each fold
        # as it will be overwhelming and is intended for final model evaluation
        y_pred_cv = (model.predict(X_val) > 0.5).astype(int)
        y_proba_cv = model.predict(X_val)

        metrics = {
            'accuracy': accuracy_score(y_val, y_pred_cv),
            'precision': precision_score(y_val, y_pred_cv),
            'recall': recall_score(y_val, y_pred_cv),
            'f1': f1_score(y_val, y_pred_cv),
            'roc_auc': roc_auc_score(y_val, y_proba_cv)
        }
        print(f"Fold {fold + 1} Metrics: {metrics}")
        cv_metrics.append(metrics)

    return pd.DataFrame(cv_metrics).mean().to_dict()


if __name__ == "__main__":
    # Load and prepare data
    X, y = load_data()
    feature_names = X.columns.tolist() # Get original feature names

    # Split data before feature engineering to prevent data leakage
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    # Feature engineering applied separately to train and test
    X_train_scaled, X_test_scaled, poly_transformer, quantile_transformer, \
    selector_kbest, standard_scaler, robust_scaler, selected_indices = \
        engineer_features(X_train_raw.values, X_test_raw.values, y_train, feature_names)

    # Cross-validation
    print("\n=== Cross-Validation ===")
    cv_results = cross_validate(X_train_scaled, y_train, n_splits=7) # Increased folds for more robust CV
    print("\nCV Results (Mean across folds):")
    for name, value in cv_results.items():
        print(f"{name:10}: {value:.4f}")

    # Final model training
    print("\n=== Final Model Training ===")
    model = build_optimized_model(X_train_scaled.shape[1])

    # Compute class weights for final model training
    class_weights_final = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )
    class_weight_dict_final = dict(enumerate(class_weights_final))
    print(f"Class weights for Final Model: {class_weight_dict_final}")


    callbacks_final = [
        EarlyStopping(monitor='val_auc', patience=30, mode='max', restore_best_weights=True, verbose=1), # Increased patience
        ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=15, mode='max', min_lr=1e-7, verbose=1), # Adjusted patience and min_lr
        TerminateOnNaN()
    ]

    history = model.fit(
        X_train_scaled, y_train,
        epochs=300, # Increased epochs
        batch_size=32,
        validation_split=0.15,
        callbacks=callbacks_final,
        verbose=1,
        class_weight=class_weight_dict_final # Apply class weights
    )

    # Evaluation
    print("\n=== Evaluating Final Model on Test Set ===")
    final_metrics = evaluate_model(model, X_test_scaled, y_test)
    print("\n=== Final Test Metrics ===")
    for name, value in final_metrics.items():
        print(f"{name:10}: {value:.4f}")

    # Plot training history (Accuracy and Loss learning curves)
    plt.figure(figsize=(12, 6))

    # Plot Accuracy
    plt.subplot(1, 2, 1) # 1 row, 2 columns, first plot
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot Loss
    plt.subplot(1, 2, 2) # 1 row, 2 columns, second plot
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout() # Adjust layout to prevent overlapping titles/labels
    # Save the plot instead of showing it
    plt.savefig(os.path.join(PLOTS_DIR, 'learning_curves.png'))
    plt.close() # Close the plot to free memory

    print(f"Learning Curves saved to {os.path.join(PLOTS_DIR, 'learning_curves.png')}")

    print(f"\nAll visualizations are saved in the '{PLOTS_DIR}' directory.")

    # --- Save the trained model and preprocessors ---
    print(f"\nSaving model and preprocessors to '{ARTIFACTS_DIR}' directory...")
    model.save(os.path.join(ARTIFACTS_DIR, 'heart_attack_model.h5'))
    joblib.dump(poly_transformer, os.path.join(ARTIFACTS_DIR, 'poly_transformer.pkl'))
    joblib.dump(quantile_transformer, os.path.join(ARTIFACTS_DIR, 'quantile_transformer.pkl'))
    joblib.dump(selector_kbest, os.path.join(ARTIFACTS_DIR, 'selector_kbest.pkl'))
    joblib.dump(standard_scaler, os.path.join(ARTIFACTS_DIR, 'standard_scaler.pkl'))
    joblib.dump(robust_scaler, os.path.join(ARTIFACTS_DIR, 'robust_scaler.pkl'))
    joblib.dump(selected_indices, os.path.join(ARTIFACTS_DIR, 'selected_indices.pkl'))
    joblib.dump(feature_names, os.path.join(ARTIFACTS_DIR, 'feature_names.pkl'))
    joblib.dump(final_metrics, os.path.join(ARTIFACTS_DIR, 'final_metrics.pkl'))  # Save final metrics
    print("Model and preprocessors saved successfully!")


    # Function to get user input and make prediction - NOT USED IN FLASK APP DIRECTLY
    # This part can be removed or kept for command line testing if desired.
    def predict_user_input(model, feature_names, poly_transformer, quantile_transformer,
                           selector_kbest, standard_scaler, robust_scaler, selected_indices):
        print("\n=== Heart Attack Risk Prediction ===")
        print("Please enter the following health metrics:")

        user_data = {}
        for feature in feature_names:
            while True:
                try:
                    value = float(input(f"Enter {feature}: "))
                    user_data[feature] = value
                    break
                except ValueError:
                    print("Please enter a valid number.")

        # Convert to DataFrame
        user_df = pd.DataFrame([user_data])

        # Apply all the same transformations as training data
        # 1. Polynomial features
        user_poly = poly_transformer.transform(user_df)

        # 2. Quantile transformation
        user_qt = quantile_transformer.transform(user_poly)

        # 3. Feature selection
        user_selected = user_qt[:, selected_indices]

        # 4. Scaling
        user_scaled_temp = standard_scaler.transform(user_selected)
        user_scaled = robust_scaler.transform(user_scaled_temp)

        # Make prediction
        prediction_proba = model.predict(user_scaled)[0][0]
        prediction_class = "positive" if prediction_proba > 0.5 else "negative"

        print("\n=== Prediction Result ===")
        print(f"Heart Attack Risk Probability: {prediction_proba:.4f}")
        print(f"Prediction: {prediction_class}")

        if prediction_proba > 0.5:
            print("Warning: High risk of heart attack detected. Please consult a doctor.")
        else:
            print("Low risk of heart attack detected. Maintain a healthy lifestyle.")

        return prediction_proba, prediction_class

