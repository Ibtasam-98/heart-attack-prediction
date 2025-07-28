import streamlit as st
import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

ARTIFACTS_DIR = 'model_artifacts'

st.set_page_config(
    page_title="Heart Attack Risk Prediction",
    page_icon="",
    layout="wide",
    initial_sidebar_state="auto"
)


@st.cache_resource
def load_artifacts():
    try:
        model = tf.keras.models.load_model(os.path.join(ARTIFACTS_DIR, 'heart_attack_model.h5'))
        poly_transformer = joblib.load(os.path.join(ARTIFACTS_DIR, 'poly_transformer.pkl'))
        quantile_transformer = joblib.load(os.path.join(ARTIFACTS_DIR, 'quantile_transformer.pkl'))
        selector_kbest = joblib.load(os.path.join(ARTIFACTS_DIR, 'selector_kbest.pkl'))
        standard_scaler = joblib.load(os.path.join(ARTIFACTS_DIR, 'standard_scaler.pkl'))
        robust_scaler = joblib.load(os.path.join(ARTIFACTS_DIR, 'robust_scaler.pkl'))
        selected_indices = joblib.load(os.path.join(ARTIFACTS_DIR, 'selected_indices.pkl'))
        feature_names = joblib.load(os.path.join(ARTIFACTS_DIR, 'feature_names.pkl'))
        final_metrics = joblib.load(os.path.join(ARTIFACTS_DIR, 'final_metrics.pkl'))

        return model, poly_transformer, quantile_transformer, selector_kbest, \
            standard_scaler, robust_scaler, selected_indices, feature_names, final_metrics
    except Exception as e:
        st.error(
            f"Error loading model or preprocessors. Please ensure 'train_model.py' has been run and artifacts are in '{ARTIFACTS_DIR}'. Error: {e}")
        return None, None, None, None, None, None, None, None, {
            'accuracy': 'N/A', 'precision': 'N/A', 'recall': 'N/A', 'f1': 'N/A', 'roc_auc': 'N/A'
        }


model, poly_transformer, quantile_transformer, selector_kbest, \
    standard_scaler, robust_scaler, selected_indices, feature_names, final_metrics = load_artifacts()

# --- Application Title and Description ---
st.title("Heart Attack Risk Prediction App")
st.write("Enter the patient's health metrics below to predict their heart attack risk.")

# Create two main columns
col1, col2 = st.columns([1, 1], gap="large")

# Left column - Patient Health Metrics
with col1:
    st.subheader("Patient Health Metrics")

    if feature_names is None:
        st.warning("Cannot display input form as feature names could not be loaded.")
    else:
        with st.form("prediction_form"):
            form_col1, form_col2 = st.columns(2)
            user_data = {}

            for i, feature in enumerate(feature_names):
                if feature == 'Gender':
                    with form_col1 if i % 2 == 0 else form_col2:
                        gender_map = {'Male': 1, 'Female': 0}
                        selected_gender = st.selectbox(
                            "Gender",
                            options=list(gender_map.keys()),
                            index=0
                        )
                        user_data[feature] = gender_map[selected_gender]
                else:
                    with form_col1 if i % 2 == 0 else form_col2:
                        default_value = 0.0
                        if feature == 'Age':
                            default_value = 45.0
                        elif feature == 'Heart rate':
                            default_value = 70.0
                        elif feature == 'Systolic blood pressure':
                            default_value = 120.0
                        elif feature == 'Diastolic blood pressure':
                            default_value = 80.0
                        elif feature == 'Blood sugar':
                            default_value = 100.0
                        elif feature == 'CK-MB':
                            default_value = 2.0
                        elif feature == 'Troponin':
                            default_value = 0.01

                        user_data[feature] = st.number_input(
                            f"{feature.replace('_', ' ').title()}",
                            value=float(default_value),
                            step=0.01,
                            format="%.2f"
                        )

            submitted = st.form_submit_button("Predict Risk")

# Right column - Model Performance Metrics
with col2:
    st.subheader("Model Performance Metrics (on Test Set)")

    metrics_df_horizontal = pd.DataFrame([final_metrics])
    st.dataframe(metrics_df_horizontal.style.format("{:.4f}"), use_container_width=True)

    st.markdown("---")
    st.markdown("**Metric Definitions:**")
    st.markdown("- **Accuracy:** Overall correctness of the model")
    st.markdown("- **Precision:** Proportion of positive identifications that were correct")
    st.markdown("- **Recall:** Proportion of actual positives that were identified correctly")
    st.markdown("- **F1:** Balance between precision and recall")
    st.markdown("- **ROC AUC:** Ability to distinguish between classes")

# Prediction results - appears right below the form in the left column
if submitted:
    with col1:  # Display results in the left column below the form
        if model is None:
            st.error("Prediction cannot be made because the model was not loaded successfully.")
        else:
            try:
                # Convert user input to DataFrame
                user_df = pd.DataFrame([user_data])

                # Apply transformations
                user_poly = poly_transformer.transform(user_df)
                user_qt = quantile_transformer.transform(user_poly)
                user_selected = user_qt[:, selected_indices]
                user_scaled_temp = standard_scaler.transform(user_selected)
                user_scaled = robust_scaler.transform(user_scaled_temp)

                # Make prediction
                prediction_proba = model.predict(user_scaled)[0][0]
                prediction_class = "Positive" if prediction_proba > 0.5 else "Negative"

                st.subheader("Prediction Result")
                if prediction_class == "Positive":
                    st.error(
                        f"High Heart Attack Risk Detected: {prediction_class} (Probability: {prediction_proba:.4f})")
                    st.warning("Please consult a doctor immediately for further evaluation.")
                else:
                    st.success(
                        f"Low Heart Attack Risk Detected: {prediction_class} (Probability: {prediction_proba:.4f})")
                    st.info("Maintain a healthy lifestyle and regular check-ups.")

                st.write("---")
                st.subheader("Input Data Provided:")
                st.json(user_data)

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.warning("Please ensure all input fields are filled correctly with valid numerical values.")