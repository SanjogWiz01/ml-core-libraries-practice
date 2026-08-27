"""Shared plotting utilities for the deep-ml-roadmap."""

import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curves(history, title="Training"):
    """Plot train/val loss and accuracy from a Keras history object or dict."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.get("loss", []), label="train loss")
    axes[0].plot(history.get("val_loss", []), label="val loss")
    axes[0].set_title(f"{title} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    if "accuracy" in history or "val_accuracy" in history:
        axes[1].plot(history.get("accuracy", []), label="train acc")
        axes[1].plot(history.get("val_accuracy", []), label="val acc")
        axes[1].set_title(f"{title} — Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    """Plot a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_title(title)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    plt.show()
