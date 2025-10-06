import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
import pandas as pd
import time
from config import MODEL_CONFIG


def build_model(input_shape):
    """Build MLP classifier with similar architecture to the original neural network"""
    return MLPClassifier(
        hidden_layer_sizes=(512, 256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=MODEL_CONFIG['batch_size'],
        learning_rate='adaptive',
        learning_rate_init=MODEL_CONFIG['learning_rate'],
        max_iter=MODEL_CONFIG['epochs'],
        early_stopping=True,
        validation_fraction=MODEL_CONFIG['validation_split'],
        n_iter_no_change=MODEL_CONFIG['early_stopping_patience'],
        random_state=MODEL_CONFIG['random_state']
    )


def get_callbacks():
    return []


def cross_validate(X, y, n_splits=MODEL_CONFIG['cv_splits']):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=MODEL_CONFIG['random_state'])
    cv_metrics_list = []
    fold_times = []

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class weights for CV: {class_weight_dict}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = build_model(X_train.shape[1])

        start_time = time.time()
        model.fit(X_train, y_train)
        fold_time = time.time() - start_time
        fold_times.append(fold_time)

        y_pred_cv = model.predict(X_val)
        y_proba_cv = model.predict_proba(X_val)[:, 1]

        # Calculate confusion matrix components
        cm = confusion_matrix(y_val, y_pred_cv)
        tn, fp, fn, tp = cm.ravel()
        error_rate = (fp + fn) / (tp + tn + fp + fn)

        metrics = {
            'accuracy': accuracy_score(y_val, y_pred_cv),
            'precision': precision_score(y_val, y_pred_cv),
            'recall': recall_score(y_val, y_pred_cv),
            'f1': f1_score(y_val, y_pred_cv),
            'roc_auc': roc_auc_score(y_val, y_proba_cv),
            'training_time': fold_time,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'error_rate': error_rate
        }

        print(f"Fold {fold + 1} Detailed Results:")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        print(f"  F1-Score:    {metrics['f1']:.4f}")
        print(f"  ROC-AUC:     {metrics['roc_auc']:.4f}")
        print(f"  Training Time: {metrics['training_time']:.2f}s")
        print(f"  Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
        print(f"  Error Rate:      {metrics['error_rate']:.4f}")

        cv_metrics_list.append(metrics)

    return cv_metrics_list, pd.DataFrame(cv_metrics_list).mean().to_dict(), fold_times
