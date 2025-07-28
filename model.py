import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.regularizers import l1_l2
import tensorflow as tf
from config import MODEL_CONFIG
import pandas as pd

def build_model(input_shape):
    model = Sequential([
        Dense(512, activation='gelu', input_shape=(input_shape,),
              kernel_regularizer=l1_l2(l1=0.0001, l2=0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='gelu', kernel_regularizer=l1_l2(l1=0.0001, l2=0.001)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation='gelu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='gelu'),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])

    optimizer = Adam(learning_rate=MODEL_CONFIG['learning_rate'], clipnorm=1.0)
    model.compile(optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy',
                         tf.keras.metrics.Precision(name='precision'),
                         tf.keras.metrics.Recall(name='recall'),
                         tf.keras.metrics.AUC(name='auc')])
    return model

def get_callbacks():
    return [
        EarlyStopping(monitor='val_auc', patience=MODEL_CONFIG['early_stopping_patience'],
                     mode='max', restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=MODEL_CONFIG['reduce_lr_patience'],
                         mode='max', min_lr=MODEL_CONFIG['min_lr'], verbose=1),
        TerminateOnNaN()
    ]

def cross_validate(X, y, n_splits=MODEL_CONFIG['cv_splits']):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=MODEL_CONFIG['random_state'])
    cv_metrics_list = []

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class weights for CV: {class_weight_dict}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = build_model(X_train.shape[1])
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=200,
            batch_size=MODEL_CONFIG['batch_size'],
            verbose=0,
            callbacks=get_callbacks(),
            class_weight=class_weight_dict
        )

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
        cv_metrics_list.append(metrics)

    return cv_metrics_list, pd.DataFrame(cv_metrics_list).mean().to_dict()