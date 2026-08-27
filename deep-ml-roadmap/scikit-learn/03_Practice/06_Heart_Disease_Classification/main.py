"""Heart Disease Classification — feature selection, class imbalance, full metrics."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, f1_score, precision_score, recall_score
)

# Using breast_cancer as a real-world binary classification proxy
# (31 numeric features, binary target, medically meaningful)


def load_data():
    data = load_breast_cancer()
    print(f"\nDataset: {data.data.shape}")
    print(f"Classes: {list(data.target_names)} = [1, 0]")
    print(f"Class distribution: {dict(zip(data.target_names, np.bincount(data.target)))}")
    print(f"\nNote: Using breast cancer dataset as a medical classification proxy.")
    print("Target 0=malignant (positive/disease class), 1=benign (negative)")
    return data.data, (data.target == 0).astype(int), np.array(data.feature_names)


def feature_selection_comparison(X_train, y_train, feature_names):
    """Compare models with all features vs SelectKBest vs RFECV."""
    print("\n--- Feature Selection Comparison ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)

    # All features
    pipe_all = Pipeline([('scaler', StandardScaler()), ('model', base_lr)])
    cv_all = cross_val_score(pipe_all, X_train, y_train, cv=skf, scoring='f1').mean()
    print(f"\n  All {X_train.shape[1]} features: F1={cv_all:.4f}")

    # SelectKBest
    for k in [5, 10, 15, 20]:
        pipe = Pipeline([
            ('sel',   SelectKBest(f_classif, k=k)),
            ('scaler', StandardScaler()),
            ('model', base_lr)
        ])
        cv = cross_val_score(pipe, X_train, y_train, cv=skf, scoring='f1').mean()
        print(f"  Top-{k:2d} features:     F1={cv:.4f}")

    # RFECV
    X_s = StandardScaler().fit_transform(X_train)
    rfecv = RFECV(estimator=LogisticRegression(max_iter=1000, class_weight='balanced'),
                  cv=skf, scoring='f1', n_jobs=-1)
    rfecv.fit(X_s, y_train)
    print(f"\n  RFECV optimal features: {rfecv.n_features_}")
    selected = feature_names[rfecv.support_]
    print(f"  Selected: {list(selected)[:10]} ...")

    return rfecv


def train_models(X_train, X_test, y_train, y_test):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Always use class_weight='balanced' or similar for medical diagnosis
    # because missing a positive case (FN) is more costly than a false alarm (FP)
    models = {
        'LogReg (balanced)':  Pipeline([('s', StandardScaler()),
                                         ('m', LogisticRegression(max_iter=1000,
                                                                   class_weight='balanced',
                                                                   random_state=42))]),
        'RandomForest (bal)': Pipeline([('m', RandomForestClassifier(n_estimators=100,
                                                                       class_weight='balanced',
                                                                       random_state=42))]),
        'GradientBoosting':   Pipeline([('m', GradientBoostingClassifier(n_estimators=100,
                                                                           random_state=42))]),
    }

    print(f"\n{'Model':22s}  {'CV AUC':>8}  {'CV F1':>7}  {'Test AUC':>9}  {'Recall':>7}")
    print(f"{'-'*22}  {'-'*8}  {'-'*7}  {'-'*9}  {'-'*7}")

    best_pipe = None
    best_auc  = -1
    all_results = {}

    for name, pipe in models.items():
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=skf, scoring='roc_auc', n_jobs=-1)
        cv_f1  = cross_val_score(pipe, X_train, y_train, cv=skf, scoring='f1', n_jobs=-1)

        pipe.fit(X_train, y_train)
        y_pred  = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        test_auc = roc_auc_score(y_test, y_proba)
        rec = recall_score(y_test, y_pred)
        print(f"{name:22s}  {cv_auc.mean():8.4f}  {cv_f1.mean():7.4f}  {test_auc:9.4f}  {rec:7.4f}")

        all_results[name] = (pipe, y_proba, y_pred)
        if cv_auc.mean() > best_auc:
            best_auc = cv_auc.mean()
            best_pipe = (name, pipe, y_proba, y_pred)

    return best_pipe, all_results


def plot_all(best_name, best_proba, best_pred, y_test, all_results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Confusion matrix
    cm = confusion_matrix(y_test, best_pred)
    axes[0].imshow(cm, cmap='Blues')
    axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(['Negative', 'Positive'])
    axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(['Negative', 'Positive'])
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
    axes[0].set_title(f"Confusion Matrix — {best_name}")
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, cm[i, j], ha='center', va='center',
                         color='white' if cm[i, j] > thresh else 'black', fontsize=14)

    # ROC curves
    for name, (pipe, proba, _) in all_results.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        axes[1].plot(fpr, tpr, label=f'{name} ({auc:.3f})')
    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
    axes[1].set_title("ROC Curves"); axes[1].legend(fontsize=7)

    # Threshold vs recall (critical for medical)
    thresholds = np.linspace(0.1, 0.9, 20)
    recalls = [recall_score(y_test, (best_proba >= t).astype(int)) for t in thresholds]
    precs   = [precision_score(y_test, (best_proba >= t).astype(int), zero_division=0)
               for t in thresholds]
    axes[2].plot(thresholds, recalls, 'r-', label='Recall')
    axes[2].plot(thresholds, precs,   'b-', label='Precision')
    axes[2].set_xlabel("Threshold"); axes[2].set_ylabel("Score")
    axes[2].set_title("Recall/Precision vs Threshold\n(for medical: prioritize high recall)")
    axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout(); plt.show()


def main():
    print("=" * 55)
    print("HEART DISEASE CLASSIFICATION — Medical Diagnosis")
    print("=" * 55)

    X, y, feature_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    feature_selection_comparison(X_train, y_train, feature_names)

    (best_name, best_pipe, best_proba, best_pred), all_results = train_models(
        X_train, X_test, y_train, y_test
    )

    print(f"\nBest model: {best_name}")
    print("\nClassification Report (Best):")
    print(classification_report(y_test, best_pred,
                                  target_names=['Negative', 'Positive (Disease)']))

    print("\n--- Class Imbalance Notes ---")
    print("  class_weight='balanced': each sample weighted inversely to class freq")
    print("  Use F1 and Recall (not accuracy) for evaluation")
    print("  Lower threshold → higher recall → fewer missed positives")
    print("  For medical: Recall (sensitivity) is more important than Precision")

    plot_all(best_name, best_proba, best_pred, y_test, all_results)


if __name__ == "__main__":
    main()
