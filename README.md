# Heart Attack Risk Prediction System

![Heart Attack Prediction](https://img.shields.io/badge/Predictive-Healthcare-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-success)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)

A machine learning system that predicts the risk of heart attack based on patient health metrics, featuring:
- Advanced neural network model
- Comprehensive feature engineering
- Interactive web interface
- Detailed performance metrics

## Table of Contents
- [Features](#features)
- [Dataset](#dataset)
- [Technical Approach](#technical-approach)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Features

### Model Architecture
- Deep neural network with GELU activation
- L1/L2 regularization and dropout layers
- Batch normalization for stable training
- Class weighting for imbalanced data

### Feature Engineering
- Polynomial feature expansion (degree=2)
- Quantile transformation for non-linear relationships
- Mutual information feature selection
- Robust scaling pipeline

### Web Interface
- Interactive input form for health metrics
- Real-time risk prediction
- Model performance visualization
- Mobile-responsive design

## Dataset

The model was trained on clinical data containing the following features:

| Feature | Description | Normal Range |
|---------|-------------|--------------|
| Age | Patient age in years | - |
| Gender | Biological sex (0: Female, 1: Male) | - |
| Heart rate | Beats per minute | 60-100 |
| Systolic BP | Blood pressure (mm Hg) | <120 |
| Diastolic BP | Blood pressure (mm Hg) | <80 |
| Blood sugar | Glucose level (mg/dL) | <140 |
| CK-MB | Cardiac enzyme (ng/mL) | 0-5 |
| Troponin | Cardiac biomarker (ng/mL) | <0.04 |

## Technical Approach

1. **Data Preprocessing**:
   - Outlier handling with IQR method
   - Stratified train-test split (85-15)
   - Class balancing with inverse frequency weighting

2. **Feature Engineering**:
   ```python
   PolynomialFeatures → QuantileTransformer → SelectKBest → StandardScaler → RobustScaler
