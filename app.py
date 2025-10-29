# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, \
    precision_score, recall_score, f1_score
from sklearn.utils import class_weight
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Heart Attack Prediction System",
    page_icon=":heart:",
    layout="wide",
    initial_sidebar_state="expanded"
)


class HeartAttackPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.results = {}
        self.feature_names = [
            'Age', 'Gender', 'Heart rate', 'Systolic blood pressure',
            'Diastolic blood pressure', 'Blood sugar', 'CK-MB', 'Troponin'
        ]

    def load_data(self):
        """Load and preprocess the dataset"""
        try:
            data = pd.read_csv('dataset/dataset.csv')

            # Convert Result column
            data['Result'] = data['Result'].map({'negative': 0, 'positive': 1})

            # Convert Gender: 0 for Male, 1 for Female
            data['Gender'] = data['Gender'].map({0: 'Male', 1: 'Female'})
            le_gender = LabelEncoder()
            data['Gender'] = le_gender.fit_transform(data['Gender'])

            # Handle outliers using IQR method
            numeric_cols = ['Age', 'Heart rate', 'Systolic blood pressure',
                            'Diastolic blood pressure', 'Blood sugar', 'CK-MB', 'Troponin']

            for col in numeric_cols:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)

            X = data[self.feature_names]
            y = data['Result']

            return X, y, le_gender

        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None, None, None

    def train_models(self, X, y):
        """Train multiple models including deep learning"""
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.scalers['standard'] = scaler

        # Define models - using the same algorithms as in main.py
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Support Vector Machine': SVC(probability=True, random_state=42),
            'Neural Network (MLP)': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                random_state=42,
                max_iter=1000,
                learning_rate_init=0.001,
                alpha=0.01
            )
        }

        # Calculate class weights
        class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(enumerate(class_weights))

        results = {}

        for i, (name, model) in enumerate(models.items()):
            status_text.text(f"Training {name}...")

            # Set class weights if supported
            if hasattr(model, 'class_weight'):
                model.class_weight = class_weight_dict

            # Train model
            start_time = time.time()
            model.fit(X_train_scaled, y_train)
            training_time = time.time() - start_time

            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba)

            # Store results
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'training_time': training_time,
                'y_pred': y_pred,
                'y_proba': y_proba,
                'y_test': y_test,
                'X_test': X_test_scaled
            }

            self.models[name] = model

            progress_bar.progress((i + 1) / len(models))

        status_text.text("Training completed!")
        self.results = results
        return results

    def predict_risk_all_models(self, input_data):
        """Predict heart attack risk using all trained models"""
        if not self.models:
            return None

        predictions = {}
        scaler = self.scalers['standard']

        # Scale input data
        input_scaled = scaler.transform([input_data])

        for model_name, model in self.models.items():
            # Make prediction
            probability = model.predict_proba(input_scaled)[0][1]
            prediction = "Positive" if probability > 0.5 else "Negative"

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

            predictions[model_name] = {
                'probability': probability,
                'prediction': prediction,
                'risk_level': risk_level,
                'recommendation': recommendation,
                'confidence': 'High' if (probability > 0.8 or probability < 0.2) else 'Medium'
            }

        return predictions


def main():
    # Main heading
    st.title("Heart Attack Prediction System")

    # Create 3 main tabs
    tab1, tab2, tab3 = st.tabs(["Home", "Train Models", "Predict"])

    # Initialize predictor in session state
    if 'predictor' not in st.session_state:
        st.session_state.predictor = HeartAttackPredictor()

    predictor = st.session_state.predictor

    with tab1:
        show_home_page()

    with tab2:
        show_train_page(predictor)

    with tab3:
        show_predict_page(predictor)


def show_home_page():
    st.header("Welcome to the Heart Attack Prediction System")

    st.markdown("""
    This application uses machine learning to predict the risk of heart attack based on various health parameters.

    ### Machine Learning Algorithms Used:
    - **Random Forest**: Ensemble learning method using multiple decision trees
    - **Logistic Regression**: Statistical model for binary classification
    - **Support Vector Machine**: Finds optimal hyperplane for classification
    - **Neural Network (MLP)**: Deep learning model with multiple hidden layers

    ### Health Parameters Used:
    - **Age**: Patient's age in years
    - **Gender**: Male (0) or Female (1)
    - **Heart Rate**: Beats per minute
    - **Systolic Blood Pressure**: Upper number in blood pressure reading
    - **Diastolic Blood Pressure**: Lower number in blood pressure reading
    - **Blood Sugar**: Glucose level in mg/dL
    - **CK-MB**: Creatine kinase-MB enzyme level
    - **Troponin**: Cardiac troponin level

    ### How to Use:
    1. Go to the **Train Models** tab to train all machine learning algorithms
    2. Navigate to the **Predict** tab to input health parameters and get risk assessment from all models

    **Note**: This tool is for educational purposes only. Always consult healthcare professionals for medical decisions.
    """)

    # # Show dataset preview if available
    # try:
    #     st.subheader("Dataset Overview")
    #     data = pd.read_csv('dataset.csv')
    #
    #     col1, col2 = st.columns(2)
    #
    #     with col1:
    #         st.write("Dataset Preview:")
    #         st.dataframe(data.head(10), use_container_width=True)
    #
    #     with col2:
    #         st.write("Class Distribution:")
    #         result_counts = data['Result'].value_counts()
    #         result_counts.index = result_counts.index.map({'negative': 'Negative', 'positive': 'Positive'})
    #         fig = px.pie(values=result_counts.values, names=result_counts.index,
    #                      title="Heart Attack Result Distribution")
    #         st.plotly_chart(fig, use_container_width=True)
    #
    # except Exception as e:
    #     st.warning("Could not load dataset. Please make sure 'dataset.csv' is in the correct location.")


def show_train_page(predictor):
    st.header("Train Machine Learning Models")

    if st.button("Load Data and Train Models", type="primary"):
        with st.spinner("Loading data..."):
            X, y, le_gender = predictor.load_data()

        if X is not None and y is not None:
            st.success(f"Data loaded successfully! {len(X)} samples with {len(X.columns)} features.")

            # Data overview
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Class Distribution")
                class_counts = y.value_counts()
                class_counts.index = ['Negative', 'Positive']
                fig = px.bar(x=class_counts.index, y=class_counts.values,
                             labels={'x': 'Result', 'y': 'Count'},
                             title="Heart Attack Cases Distribution")
                fig.update_traces(marker_color=['blue', 'red'])
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Dataset Summary")
                summary_data = {
                    'Metric': ['Total Samples', 'Positive Cases', 'Negative Cases', 'Features'],
                    'Value': [len(X), sum(y), len(X) - sum(y), len(X.columns)]
                }
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)

            # Feature correlation
            st.subheader("Feature Correlation")
            data_corr = X.copy()
            data_corr['Result'] = y
            corr_matrix = data_corr.corr()
            fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                                 title="Feature Correlation Matrix")
            st.plotly_chart(fig_corr, use_container_width=True)

            # Train models
            st.subheader("Model Training Progress")
            results = predictor.train_models(X, y)

            # Performance comparison
            st.subheader("Model Performance Comparison")

            # Performance table
            performance_data = []
            for model_name, result in results.items():
                performance_data.append({
                    'Model': model_name,
                    'Accuracy': f"{result['accuracy']:.4f}",
                    'Precision': f"{result['precision']:.4f}",
                    'Recall': f"{result['recall']:.4f}",
                    'F1-Score': f"{result['f1']:.4f}",
                    'ROC AUC': f"{result['roc_auc']:.4f}",
                    'Training Time (s)': f"{result['training_time']:.2f}"
                })

            perf_df = pd.DataFrame(performance_data)
            st.dataframe(perf_df, use_container_width=True)

            # Performance chart
            fig = go.Figure()
            metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
            for metric in metrics_to_plot:
                fig.add_trace(go.Bar(name=metric, x=perf_df['Model'],
                                     y=pd.to_numeric(perf_df[metric])))
            fig.update_layout(title="Model Performance Comparison", barmode='group')
            st.plotly_chart(fig, use_container_width=True)

            # Detailed results for each model
            st.subheader("Detailed Model Results")

            for model_name, result in results.items():
                with st.expander(f"{model_name} - Detailed Analysis"):
                    col1, col2 = st.columns(2)

                    with col1:
                        # Classification report
                        st.write("Classification Report:")
                        report = classification_report(result['y_test'], result['y_pred'], output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

                    with col2:
                        # Confusion matrix
                        st.write("Confusion Matrix:")
                        cm = confusion_matrix(result['y_test'], result['y_pred'])
                        fig_cm = px.imshow(cm, text_auto=True,
                                           labels=dict(x="Predicted", y="Actual", color="Count"),
                                           x=['Negative', 'Positive'],
                                           y=['Negative', 'Positive'],
                                           title=f"Confusion Matrix - {model_name}")
                        st.plotly_chart(fig_cm, use_container_width=True)

                    # ROC Curve
                    fpr, tpr, _ = roc_curve(result['y_test'], result['y_proba'])
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                                 name=f'ROC curve (AUC = {result["roc_auc"]:.4f})'))
                    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                                 name='Random Classifier', line=dict(dash='dash')))
                    fig_roc.update_layout(title=f"ROC Curve - {model_name}",
                                          xaxis_title="False Positive Rate",
                                          yaxis_title="True Positive Rate")
                    st.plotly_chart(fig_roc, use_container_width=True)

                    # Feature importance for Random Forest
                    if model_name == 'Random Forest':
                        st.write("Feature Importance:")
                        feature_importance = result['model'].feature_importances_
                        feature_imp_df = pd.DataFrame({
                            'Feature': predictor.feature_names,
                            'Importance': feature_importance
                        }).sort_values('Importance', ascending=True)

                        fig_imp = px.bar(feature_imp_df, x='Importance', y='Feature',
                                         orientation='h', title="Feature Importance - Random Forest")
                        st.plotly_chart(fig_imp, use_container_width=True)


def show_predict_page(predictor):
    st.header("Heart Attack Risk Prediction")

    if not predictor.models:
        st.warning("Please train models first in the 'Train Models' tab.")
        return

    st.write("Adjust the health parameters using the sliders below to get predictions from all trained models:")

    # Create sliders for input parameters
    st.subheader("Health Parameters")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age (years)", 18, 100, 50)
        gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
        heart_rate = st.slider("Heart Rate (bpm)", 40, 200, 72)
        systolic_bp = st.slider("Systolic Blood Pressure (mmHg)", 80, 220, 120)

    with col2:
        diastolic_bp = st.slider("Diastolic Blood Pressure (mmHg)", 40, 140, 80)
        blood_sugar = st.slider("Blood Sugar (mg/dL)", 50, 600, 100)
        ck_mb = st.slider("CK-MB (ng/mL)", 0.0, 300.0, 2.0, step=0.1)
        troponin = st.slider("Troponin (ng/mL)", 0.0, 10.0, 0.01, step=0.001, format="%.3f")

    # Prepare input data
    input_data = [age, gender, heart_rate, systolic_bp, diastolic_bp, blood_sugar, ck_mb, troponin]

    # Make prediction
    if st.button("Predict Heart Attack Risk", type="primary"):
        with st.spinner("Analyzing health parameters with all models..."):
            predictions = predictor.predict_risk_all_models(input_data)

        if predictions:
            st.subheader("Prediction Results from All Models")

            # Display results for each model
            for model_name, result in predictions.items():
                with st.expander(f"{model_name} Prediction", expanded=True):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Risk Probability", f"{result['probability']:.4f}")
                        st.metric("Prediction", result['prediction'])

                    with col2:
                        st.metric("Risk Level", result['risk_level'])
                        st.metric("Confidence", result['confidence'])

                    with col3:
                        st.write("Recommendation:")
                        st.info(result['recommendation'])

                    # Risk visualization for each model
                    fig = go.Figure()
                    fig.add_trace(go.Indicator(
                        mode="gauge+number",
                        value=result['probability'] * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': f"Risk Percentage - {model_name}"},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "green"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50}}))

                    fig.update_layout(height=250)
                    st.plotly_chart(fig, use_container_width=True)

            # Overall consensus
            st.subheader("Overall Consensus")

            # Calculate average probability
            avg_probability = np.mean([result['probability'] for result in predictions.values()])
            positive_predictions = sum(1 for result in predictions.values() if result['prediction'] == 'Positive')
            total_models = len(predictions)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Average Risk Probability", f"{avg_probability:.4f}")

            with col2:
                st.metric("Models Predicting Positive", f"{positive_predictions}/{total_models}")

            with col3:
                consensus = "HIGH RISK" if positive_predictions > total_models / 2 else "LOW RISK"
                st.metric("Overall Consensus", consensus)

            # Detailed interpretation based on consensus
            st.subheader("Detailed Interpretation")
            if positive_predictions > total_models / 2:
                st.error("MAJORITY OF MODELS DETECT HIGH RISK")
                st.write(f"- Average probability of heart attack: {avg_probability * 100:.2f}%")
                st.write("- {}/{} models indicate high risk".format(positive_predictions, total_models))
                st.write("- Immediate medical attention recommended")
                st.write("- Contact healthcare provider")
            else:
                st.success("MAJORITY OF MODELS DETECT LOW RISK")
                st.write(f"- Average probability of heart attack: {avg_probability * 100:.2f}%")
                st.write("- {}/{} models indicate low risk".format(total_models - positive_predictions, total_models))
                st.write("- Continue maintaining healthy lifestyle")
                st.write("- Regular checkups recommended")

            st.write(
                "Note: This prediction is based on machine learning analysis. Always consult with healthcare professionals for medical decisions.")


if __name__ == "__main__":
    main()
