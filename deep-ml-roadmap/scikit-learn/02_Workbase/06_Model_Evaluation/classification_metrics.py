"""Classification metrics — accuracy, precision, recall, F1, ROC-AUC, confusion matrix."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score
)


def plot_confusion_matrix(cm, class_names, ax, title=""):
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title or "Confusion Matrix")
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black', fontsize=12)


def demonstrate_threshold_tradeoff(model, X_test, y_test):
    """Show precision/recall tradeoff by varying the decision threshold."""
    y_proba = model.predict_proba(X_test)[:, 1]

    thresholds = np.linspace(0.1, 0.9, 17)
    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        p = precision_score(y_test, y_pred_t, zero_division=0)
        r = recall_score(y_test, y_pred_t, zero_division=0)
        f = f1_score(y_test, y_pred_t, zero_division=0)
        precisions.append(p); recalls.append(r); f1s.append(f)

    plt.figure(figsize=(8, 4))
    plt.plot(thresholds, precisions, 'b-o', markersize=4, label='Precision')
    plt.plot(thresholds, recalls,    'r-o', markersize=4, label='Recall')
    plt.plot(thresholds, f1s,        'g-o', markersize=4, label='F1')
    plt.axvline(0.5, color='k', linestyle='--', linewidth=0.8, label='Default 0.5')
    best_t = thresholds[np.argmax(f1s)]
    plt.axvline(best_t, color='g', linestyle=':', linewidth=1.5, label=f'Best F1 @ {best_t:.2f}')
    plt.xlabel("Classification Threshold"); plt.ylabel("Score")
    plt.title("Precision / Recall / F1 vs Decision Threshold")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.show()


def compare_all_models(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'KNN(k=7)':           KNeighborsClassifier(n_neighbors=7),
        'RandomForest':       RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    print(f"\n  {'Model':22s}  {'Acc':>6}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'AUC':>6}")
    print(f"  {'-'*22}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")

    fig, axes = plt.subplots(1, len(models), figsize=(16, 4))

    for ax, (name, model) in zip(axes, models.items()):
        model.fit(X_tr_s, y_train)
        y_pred   = model.predict(X_te_s)
        y_proba  = model.predict_proba(X_te_s)[:, 1]

        acc   = accuracy_score(y_test, y_pred)
        prec  = precision_score(y_test, y_pred)
        rec   = recall_score(y_test, y_pred)
        f1    = f1_score(y_test, y_pred)
        auc   = roc_auc_score(y_test, y_proba)

        print(f"  {name:22s}  {acc:6.4f}  {prec:6.4f}  {rec:6.4f}  {f1:6.4f}  {auc:6.4f}")

        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm, ['Malignant', 'Benign'], ax, title=name)

    plt.tight_layout(); plt.show()

    return models['GradientBoosting']


def plot_roc_and_pr_curves(models_dict, X_tr_s, X_te_s, y_train, y_test):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for name, model in models_dict.items():
        model.fit(X_tr_s, y_train)
        y_proba = model.predict_proba(X_te_s)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax1.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')

        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        ax2.plot(rec, prec, label=f'{name} (AP={ap:.3f})')

    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR"); ax1.set_title("ROC Curves")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision"); ax2.set_title("PR Curves")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout(); plt.show()


def main():
    print("=" * 55)
    print("CLASSIFICATION METRICS — Full Suite")
    print("=" * 55)

    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    print("\n--- Comparing All Models ---")
    compare_all_models(X_train, X_test, y_train, y_test)

    print("\n--- Threshold Tradeoff (LogisticRegression) ---")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_tr_s, y_train)
    demonstrate_threshold_tradeoff(lr, X_te_s, y_test)

    models = {
        'LogReg': LogisticRegression(max_iter=1000, random_state=42),
        'RF':     RandomForestClassifier(n_estimators=100, random_state=42),
        'GB':     GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    plot_roc_and_pr_curves(models, X_tr_s, X_te_s, y_train, y_test)

    print("\n--- When to use each metric ---")
    print("  Accuracy:  only when classes are balanced and errors cost the same")
    print("  Precision: when false positives are costly (spam filter, loan approval)")
    print("  Recall:    when false negatives are costly (cancer detection, fraud)")
    print("  F1:        balanced tradeoff, especially on imbalanced data")
    print("  ROC-AUC:   overall ranking quality, threshold-independent")


if __name__ == "__main__":
    main()
