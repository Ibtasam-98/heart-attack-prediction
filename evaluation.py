import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score
import os
import pandas as pd
from config import PLOTS_DIR


def evaluate_model(model, X_test, y_test):
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    y_proba = model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(PLOTS_DIR, 'confusion_matrix.png'))
    plt.close()

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }

    plot_final_metrics(metrics)
    return metrics


def plot_final_metrics(metrics):
    plt.figure(figsize=(8, 6))
    metrics_names = list(metrics.keys())
    metrics_values = list(metrics.values())
    sns.barplot(x=metrics_names, y=metrics_values, palette='viridis')
    plt.title('Final Model Test Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1.0)
    for index, value in enumerate(metrics_values):
        plt.text(index, value + 0.02, f"{value:.4f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'final_test_metrics_bar_chart.png'))
    plt.close()


def plot_cv_metrics(cv_individual_fold_metrics):
    plt.figure(figsize=(12, 7))
    cv_df = pd.DataFrame(cv_individual_fold_metrics)
    cv_df['Fold'] = range(1, len(cv_df) + 1)
    cv_melted_df = cv_df.melt(id_vars=['Fold'], var_name='Metric', value_name='Value')

    sns.lineplot(data=cv_melted_df, x='Fold', y='Value', hue='Metric', marker='o', markersize=8, linewidth=2)

    plt.title('Cross-Validation Metrics Across Folds', fontsize=16)
    plt.xlabel('Fold Number', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.ylim(0, 1.05)
    plt.xticks(range(1, len(cv_df) + 1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'cv_metrics_across_folds.png'))
    plt.close()


def plot_learning_curves(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='skyblue', linestyle='--')

    ax2 = plt.gca().twinx()
    ax2.plot(history.history['loss'], label='Train Loss', color='red')
    ax2.plot(history.history['val_loss'], label='Validation Loss', color='salmon', linestyle='--')

    plt.title('Model Accuracy and Loss Learning Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy', color='blue')
    ax2.set_ylabel('Loss', color='red')

    lines, labels = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='center right')

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'combined_learning_curves.png'))
    plt.close()