
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score, roc_curve
import os
import pandas as pd
import numpy as np
from config import PLOTS_DIR


def evaluate_model(model, X_test, y_test, training_time):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

    print("\n" + "=" * 50)
    print("CONFUSION MATRIX")
    print("=" * 50)
    cm = confusion_matrix(y_test, y_pred)
    print(f"True Negatives (TN): {cm[0, 0]}")
    print(f"False Positives (FP): {cm[0, 1]}")
    print(f"False Negatives (FN): {cm[1, 0]}")
    print(f"True Positives (TP): {cm[1, 1]}")
    print(f"\nConfusion Matrix:")
    print(cm)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'))
    plt.close()

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'roc_curve.png'))
    plt.close()

    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    error_rate = (fp + fn) / (tp + tn + fp + fn)
    efficiency = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc,
        'training_time': training_time,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'error_rate': error_rate,
        'efficiency': efficiency
    }

    plot_final_metrics(metrics)
    return metrics


def plot_final_metrics(metrics):
    # Main metrics bar chart
    plt.figure(figsize=(10, 6))
    main_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    main_values = [metrics[m] for m in main_metrics]

    plt.subplot(1, 2, 1)
    bars = plt.bar(main_metrics, main_values, color=['blue', 'green', 'orange', 'red', 'purple'])
    plt.title('Main Performance Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1.0)
    for bar, value in zip(bars, main_values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f'{value:.4f}',
                 ha='center', va='bottom', fontweight='bold')
    plt.xticks(rotation=45)

    # Confusion matrix metrics
    plt.subplot(1, 2, 2)
    cm_metrics = ['TP', 'TN', 'FP', 'FN', 'Error Rate']
    cm_values = [metrics['tp'], metrics['tn'], metrics['fp'], metrics['fn'], metrics['error_rate']]
    colors = ['green', 'blue', 'orange', 'red', 'gray']

    bars = plt.bar(cm_metrics, cm_values, color=colors)
    plt.title('Confusion Matrix Metrics')
    plt.ylabel('Count / Rate')
    for bar, value in zip(bars, cm_values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + max(cm_values) * 0.01, f'{value}',
                 ha='center', va='bottom', fontweight='bold')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'comprehensive_metrics.png'))
    plt.close()


def plot_cv_metrics(cv_individual_fold_metrics, fold_times):
    # Metrics across folds
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    cv_df = pd.DataFrame(cv_individual_fold_metrics)
    cv_df['Fold'] = range(1, len(cv_df) + 1)
    cv_melted_df = cv_df.melt(id_vars=['Fold'], value_vars=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                              var_name='Metric', value_name='Value')

    sns.lineplot(data=cv_melted_df, x='Fold', y='Value', hue='Metric', marker='o', markersize=8, linewidth=2)
    plt.title('Cross-Validation Metrics Across Folds', fontsize=14)
    plt.xlabel('Fold Number')
    plt.ylabel('Metric Value')
    plt.ylim(0, 1.05)
    plt.xticks(range(1, len(cv_df) + 1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Training time vs accuracy
    plt.subplot(2, 2, 2)
    accuracies = [fold['accuracy'] for fold in cv_individual_fold_metrics]
    plt.scatter(fold_times, accuracies, s=100, alpha=0.7)
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Accuracy')
    plt.title('Training Time vs Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    for i, (time_val, acc) in enumerate(zip(fold_times, accuracies)):
        plt.annotate(f'Fold {i + 1}', (time_val, acc), xytext=(5, 5), textcoords='offset points')

    # Model performance rankings
    plt.subplot(2, 2, 3)
    mean_metrics = pd.DataFrame(cv_individual_fold_metrics)[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']].mean()
    mean_metrics.plot(kind='bar', color=['blue', 'green', 'orange', 'red', 'purple'])
    plt.title('Average Performance Across Metrics')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.ylim(0, 1.0)

    # Training time distribution
    plt.subplot(2, 2, 4)
    plt.bar(range(1, len(fold_times) + 1), fold_times, color='skyblue')
    plt.xlabel('Fold Number')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time per Fold')
    plt.xticks(range(1, len(fold_times) + 1))
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'comprehensive_cv_analysis.png'))
    plt.close()


def plot_learning_curves(model):
    if hasattr(model, 'loss_curve_'):
        plt.figure(figsize=(10, 6))
        plt.plot(model.loss_curve_, label='Training Loss', color='red', linewidth=2)
        plt.title('Model Learning Curve - Training Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'learning_curve.png'))
        plt.close()


def print_comprehensive_metrics(final_metrics, cv_metrics):
    print("\n" + "=" * 80)
    print("COMPREHENSIVE MODEL PERFORMANCE AND ANALYSIS METRICS")
    print("=" * 80)

    print("\nMODEL PERFORMANCE METRICS:")
    print("-" * 60)
    print(f"{'Metric':<15} {'Test Set':<12} {'CV Mean':<12} {'Difference':<12}")
    print("-" * 60)
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        test_val = final_metrics[metric]
        cv_val = cv_metrics.get(metric, 0)
        diff = test_val - cv_val
        print(f"{metric.upper():<15} {test_val:<12.4f} {cv_val:<12.4f} {diff:<12.4f}")

    print("\nCONFUSION MATRIX ANALYSIS:")
    print("-" * 40)
    print(f"True Positives (TP):  {final_metrics['tp']}")
    print(f"True Negatives (TN):  {final_metrics['tn']}")
    print(f"False Positives (FP): {final_metrics['fp']}")
    print(f"False Negatives (FN): {final_metrics['fn']}")
    print(f"Error Rate:           {final_metrics['error_rate']:.4f}")
    print(f"Efficiency (Recall):  {final_metrics['efficiency']:.4f}")

    print("\nPERFORMANCE SUMMARY:")
    print("-" * 40)
    print(f"Training Time:        {final_metrics['training_time']:.2f} seconds")
    print(f"Best CV AUC:          {cv_metrics.get('roc_auc', 0):.4f}")
    print(f"Final Test AUC:       {final_metrics['roc_auc']:.4f}")

    print("\nMODEL PERFORMANCE RANKINGS:")
    print("-" * 40)
    metrics_rank = {
        'Accuracy': final_metrics['accuracy'],
        'Precision': final_metrics['precision'],
        'Recall': final_metrics['recall'],
        'F1-Score': final_metrics['f1'],
        'ROC-AUC': final_metrics['roc_auc']
    }
    ranked_metrics = sorted(metrics_rank.items(), key=lambda x: x[1], reverse=True)
    for i, (metric, value) in enumerate(ranked_metrics, 1):
        print(f"{i}. {metric}: {value:.4f}")

    print("\n" + "=" * 80)
