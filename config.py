import os

ARTIFACTS_DIR = 'model_artifacts'
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

PLOTS_DIR = 'model_plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    'model_type': 'sklearn_mlp',
    'input_shape': None,
    'learning_rate': 0.0001,
    'epochs': 300,
    'batch_size': 32,
    'validation_split': 0.15,
    'early_stopping_patience': 30,
    'reduce_lr_patience': 15,
    'min_lr': 1e-7,
    'cv_splits': 7,
    'test_size': 0.15,
    'random_state': 42,
}
