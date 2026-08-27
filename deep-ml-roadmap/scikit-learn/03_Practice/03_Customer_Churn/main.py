"""Customer Churn Classification — encoding, class imbalance, ROC-AUC."""

from pathlib import Path
import subprocess, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, f1_score, precision_score, recall_score
)

ROOT = Path(__file__).parent.parent.parent
DATA_PATH = ROOT / "data" / "raw" / "customer_churn.csv"


def ensure_data():
    if not DATA_PATH.exists():
        subprocess.run([sys.executable, str(ROOT / "data" / "raw" / "generate_datasets.py")],
                       check=True)


def load_and_explore(path):
    df = pd.read_csv(path)
    print(f"\nDataset: {df.shape}")
    print(f"Missing:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    churn_rate = df['churn'].mean()
    print(f"\nChurn rate: {churn_rate:.2%}  (class imbalance ratio: {1/churn_rate:.1f}:1)")
    return df


def build_pipeline(model):
    numeric_cols     = ['tenure', 'monthly_charges', 'total_charges', 'support_calls']
    nominal_cols     = ['contract', 'payment_method', 'internet_service']
    passthrough_cols = ['paperless_billing']

    numeric_t = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler())
    ])
    nominal_t = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe',     OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num',  numeric_t,     numeric_cols),
        ('nom',  nominal_t,     nominal_cols),
        ('pass', 'passthrough', passthrough_cols),
    ])

    return Pipeline([('preprocessor', preprocessor), ('model', model)])


def evaluate_model(name, pipe, X_train, X_test, y_train, y_test):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_auc = cross_val_score(pipe, X_train, y_train, cv=skf, scoring='roc_auc', n_jobs=-1)
    cv_f1  = cross_val_score(pipe, X_train, y_train, cv=skf, scoring='f1',     n_jobs=-1)

    pipe.fit(X_train, y_train)
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    test_auc = roc_auc_score(y_test, y_proba)
    test_f1  = f1_score(y_test, y_pred)
    prec     = precision_score(y_test, y_pred)
    rec      = recall_score(y_test, y_pred)

    print(f"\n  {name}:")
    print(f"    CV ROC-AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
    print(f"    CV F1:      {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
    print(f"    Test AUC:   {test_auc:.4f}  |  F1={test_f1:.4f}  "
          f"|  Prec={prec:.4f}  |  Rec={rec:.4f}")

    return pipe, y_pred, y_proba


def plot_confusion_matrix(cm, ax, title):
    ax.imshow(cm, cmap='Blues')
    classes = ['Stay', 'Churn']
    ax.set_xticks([0, 1]); ax.set_xticklabels(classes)
    ax.set_yticks([0, 1]); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')


def main():
    print("=" * 55)
    print("CUSTOMER CHURN — Binary Classification")
    print("=" * 55)

    ensure_data()
    df = load_and_explore(DATA_PATH)

    X = df.drop(columns=['churn'])
    y = df['churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        'LogisticRegression': LogisticRegression(C=1.0, max_iter=1000,
                                                   class_weight='balanced', random_state=42),
        'RandomForest':       RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                                      random_state=42),
        'GradientBoosting':   GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                                           random_state=42),
    }

    print("\n--- Model Comparison ---")
    best_pipe = None
    best_auc  = -1
    best_name = None
    all_probas = {}

    for name, model in models.items():
        pipe = build_pipeline(model)
        fitted_pipe, y_pred, y_proba = evaluate_model(
            name, pipe, X_train, X_test, y_train, y_test
        )
        auc = roc_auc_score(y_test, y_proba)
        all_probas[name] = y_proba
        if auc > best_auc:
            best_auc  = auc
            best_name = name
            best_pipe = fitted_pipe

    print(f"\nBest model: {best_name} (AUC={best_auc:.4f})")

    # --- Full report ---
    print("\n--- Classification Report (Best Model) ---")
    y_pred_best = best_pipe.predict(X_test)
    print(classification_report(y_test, y_pred_best, target_names=['Stay', 'Churn']))

    # --- Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Confusion matrices for all models
    for name, proba in all_probas.items():
        pass  # skip — show only best

    cm_best = confusion_matrix(y_test, y_pred_best)
    plot_confusion_matrix(cm_best, axes[0], f"Confusion Matrix — {best_name}")

    # ROC curves
    for name, proba in all_probas.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        axes[1].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
    axes[1].set_title("ROC Curves"); axes[1].legend(fontsize=8)

    # Feature importance
    best_model = best_pipe.named_steps['model']
    if hasattr(best_model, 'feature_importances_'):
        X_enc = best_pipe.named_steps['preprocessor'].transform(X_test)
        imp = best_model.feature_importances_
        top_n = min(10, len(imp))
        idx = np.argsort(imp)[::-1][:top_n]
        axes[2].barh(range(top_n), imp[idx][::-1])
        axes[2].set_yticks(range(top_n))
        axes[2].set_yticklabels([f"f_{i}" for i in idx][::-1])
        axes[2].set_title("Feature Importance")

    plt.tight_layout(); plt.show()

    # --- Threshold analysis ---
    print("\n--- Threshold Impact (Best Model) ---")
    best_proba = all_probas[best_name]
    print(f"  {'Threshold':>10}  {'Precision':>10}  {'Recall':>7}  {'F1':>7}  {'% Flagged':>10}")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_at_t = (best_proba >= t).astype(int)
        p = precision_score(y_test, y_at_t, zero_division=0)
        r = recall_score(y_test, y_at_t, zero_division=0)
        f = f1_score(y_test, y_at_t, zero_division=0)
        pct = y_at_t.mean() * 100
        print(f"  {t:>10.2f}  {p:>10.4f}  {r:>7.4f}  {f:>7.4f}  {pct:>9.1f}%")


if __name__ == "__main__":
    main()
