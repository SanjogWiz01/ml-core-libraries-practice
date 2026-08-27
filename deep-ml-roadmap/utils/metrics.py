"""Shared evaluation metrics."""

import numpy as np


def accuracy(y_true, y_pred):
    """Fraction of correct predictions."""
    return np.mean(np.array(y_true) == np.array(y_pred))


def f1_macro(y_true, y_pred):
    """Macro-averaged F1 score without sklearn dependency."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    classes = np.unique(y_true)
    f1s = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1s.append(2 * precision * recall / (precision + recall + 1e-9))
    return float(np.mean(f1s))


def rmse(y_true, y_pred):
    """Root mean squared error."""
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))
