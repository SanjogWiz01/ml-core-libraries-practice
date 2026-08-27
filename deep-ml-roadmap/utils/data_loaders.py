"""Dataset loading helpers shared across all library folders."""

import os
import numpy as np


def load_csv_split(path, target_col, test_size=0.2, seed=42):
    """Load a CSV and return (X_train, X_test, y_train, y_test).

    Requires pandas and scikit-learn; kept separate so other utils stay lightweight.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(path)
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    return train_test_split(X, y, test_size=test_size, random_state=seed)


def load_mnist_numpy():
    """Return MNIST as numpy arrays via keras without TF/PyTorch dependency."""
    from tensorflow.keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return X_train, X_test, y_train, y_test


def set_seed(seed=42):
    """Set seeds for reproducibility across numpy, random, and (optionally) torch/tf."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
