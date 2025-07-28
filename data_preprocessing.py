import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PolynomialFeatures, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import joblib
from config import ARTIFACTS_DIR, PLOTS_DIR, MODEL_CONFIG

def load_data():
    data = pd.read_csv('dataset/dataset.csv')
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

def engineer_features(X_train, X_test, y_train, feature_names):
    # Convert to DataFrame to maintain column names for PolynomialFeatures
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    # Add Polynomial Features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_df)
    X_test_poly = poly.transform(X_test_df)

    print(f"Original features: {X_train.shape[1]}, Polynomial features: {X_train_poly.shape[1]}")

    # Quantile transformation
    quantile = QuantileTransformer(output_distribution='normal', random_state=MODEL_CONFIG['random_state'])
    X_train_qt = quantile.fit_transform(X_train_poly)
    X_test_qt = quantile.transform(X_test_poly)

    # Mutual information for feature selection
    selector = SelectKBest(mutual_info_classif, k='all')
    selector.fit(X_train_qt, y_train)

    # Get top 80% of features
    k = int(0.80 * X_train_qt.shape[1])
    if k == 0 and X_train_qt.shape[1] > 0:
        k = 1
    elif k == 0:
        raise ValueError("No features available after transformations.")

    selected_features_indices = np.argsort(selector.scores_)[-k:]
    X_train_selected = X_train_qt[:, selected_features_indices]
    X_test_selected = X_test_qt[:, selected_features_indices]

    print(f"Features after selection: {X_train_selected.shape[1]}")

    # Robust scaling
    scaler = StandardScaler()
    X_train_scaled_temp = scaler.fit_transform(X_train_selected)
    X_test_scaled_temp = scaler.transform(X_test_selected)

    robust_scaler = RobustScaler()
    X_train_scaled = robust_scaler.fit_transform(X_train_scaled_temp)
    X_test_scaled = robust_scaler.transform(X_test_scaled_temp)

    return X_train_scaled, X_test_scaled, poly, quantile, selector, scaler, robust_scaler, selected_features_indices

def save_preprocessors(poly, quantile, selector, scaler, robust_scaler, selected_indices, feature_names, metrics):
    joblib.dump(poly, os.path.join(ARTIFACTS_DIR, 'poly_transformer.pkl'))
    joblib.dump(quantile, os.path.join(ARTIFACTS_DIR, 'quantile_transformer.pkl'))
    joblib.dump(selector, os.path.join(ARTIFACTS_DIR, 'selector_kbest.pkl'))
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, 'standard_scaler.pkl'))
    joblib.dump(robust_scaler, os.path.join(ARTIFACTS_DIR, 'robust_scaler.pkl'))
    joblib.dump(selected_indices, os.path.join(ARTIFACTS_DIR, 'selected_indices.pkl'))
    joblib.dump(feature_names, os.path.join(ARTIFACTS_DIR, 'feature_names.pkl'))
    joblib.dump(metrics, os.path.join(ARTIFACTS_DIR, 'final_metrics.pkl'))
    print("Preprocessors saved successfully!")