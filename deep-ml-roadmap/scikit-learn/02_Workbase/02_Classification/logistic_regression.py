"""Logistic Regression — binary classification with full metrics."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, roc_curve
)


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_xticklabels(class_names)
    ax.set_yticks([0, 1]); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout(); plt.show()


def plot_roc_curve(y_test, y_proba, auc):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, 'b-', label=f'ROC (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.show()


def main():
    print("=" * 55)
    print("LOGISTIC REGRESSION — Breast Cancer Classification")
    print("=" * 55)

    data = load_breast_cancer()
    X, y = data.data, data.target
    class_names = data.target_names  # ['malignant', 'benign']

    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {dict(zip(class_names, np.bincount(y)))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale — mandatory for logistic regression
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)

    y_pred  = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]  # P(benign)

    # --- Metrics ---
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=class_names))

    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC: {auc:.4f}")

    # --- Cross-validation ---
    cv_scores = cross_val_score(
        LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        scaler.fit_transform(X), y,   # full dataset for CV
        cv=5, scoring='roc_auc'
    )
    print(f"\n5-Fold CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # --- Effect of C (regularization) ---
    print("\n--- Effect of C (inverse regularization strength) ---")
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        m = LogisticRegression(C=C, max_iter=1000, random_state=42)
        m.fit(X_train_s, y_train)
        test_acc = accuracy_score(y_test, m.predict(X_test_s))
        print(f"  C={C:6}: acc={test_acc:.4f}")

    # --- Plots ---
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, class_names)
    plot_roc_curve(y_test, y_proba, auc)

    # --- Top features by coefficient magnitude ---
    print("\n--- Top 10 most important features ---")
    coef_abs = np.abs(model.coef_[0])
    top_idx = np.argsort(coef_abs)[::-1][:10]
    for i in top_idx:
        print(f"  {data.feature_names[i]:35s}: {model.coef_[0][i]:+.4f}")


if __name__ == "__main__":
    main()
