"""Cross-validation — KFold, StratifiedKFold, cross_val_score, cross_validate."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import (
    cross_val_score, cross_validate,
    KFold, StratifiedKFold, LeaveOneOut
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def demonstrate_kfold_splits():
    """Visualize how KFold divides the dataset."""
    n = 20
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    indices = np.arange(n)

    print("\n--- KFold splits on 20 samples ---")
    for fold, (train_idx, test_idx) in enumerate(kf.split(indices), 1):
        visual = ['T' if i in test_idx else '.' for i in range(n)]
        print(f"  Fold {fold}: [{''.join(visual)}]  test={list(test_idx)}")

    print("\n  (T = test, . = train, shuffled)")


def cv_classification():
    print("\n=== Classification CV ===")

    data = load_breast_cancer()
    X, y = data.data, data.target

    # StratifiedKFold preserves class proportions per fold — always use for classification
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model',  LogisticRegression(max_iter=1000, random_state=42))
    ])

    # cross_val_score: simplest — returns one score per fold
    scores = cross_val_score(pipe, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
    print(f"\n  cross_val_score (accuracy): {scores}")
    print(f"  Mean: {scores.mean():.4f} ± {scores.std():.4f}")

    # cross_validate: returns multiple metrics + fit/score times
    results = cross_validate(
        pipe, X, y,
        cv=skf,
        scoring=['accuracy', 'f1', 'roc_auc'],
        return_train_score=True,
        n_jobs=-1
    )

    print("\n  cross_validate results:")
    for key in ['train_accuracy', 'test_accuracy', 'test_f1', 'test_roc_auc']:
        vals = results[key]
        print(f"    {key:22s}: {vals.mean():.4f} ± {vals.std():.4f}")

    # Overfitting check: train score >> test score
    print(f"\n  Train accuracy: {results['train_accuracy'].mean():.4f}")
    print(f"  Test  accuracy: {results['test_accuracy'].mean():.4f}")
    gap = results['train_accuracy'].mean() - results['test_accuracy'].mean()
    print(f"  Gap: {gap:.4f}  {'← Overfitting' if gap > 0.05 else '← OK'}")

    return scores


def cv_regression():
    print("\n=== Regression CV ===")

    X, y = fetch_california_housing(return_X_y=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model',  RandomForestRegressor(n_estimators=50, random_state=42))
    ])

    # Note: sklearn minimizes score, so use negative for metrics where lower=better
    scores = cross_val_score(pipe, X, y, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1)
    rmse_scores = -scores  # flip sign back to positive

    print(f"\n  RMSE per fold: {rmse_scores.round(4)}")
    print(f"  Mean RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")

    r2_scores = cross_val_score(pipe, X, y, cv=kf, scoring='r2', n_jobs=-1)
    print(f"  Mean R²:   {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")


def compare_cv_strategies():
    print("\n=== CV Strategy Comparison ===")

    data = load_breast_cancer()
    X, y = data.data, data.target

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model',  LogisticRegression(max_iter=1000))
    ])

    strategies = {
        'KFold(5)':           KFold(n_splits=5, shuffle=True, random_state=42),
        'KFold(10)':          KFold(n_splits=10, shuffle=True, random_state=42),
        'StratifiedKFold(5)': StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        'StratifiedKFold(10)':StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
    }

    print(f"\n  {'Strategy':25s}  {'Mean Acc':>9}  {'Std':>8}  {'Variance':>10}")
    print(f"  {'-'*25}  {'-'*9}  {'-'*8}  {'-'*10}")
    for name, cv in strategies.items():
        s = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
        print(f"  {name:25s}  {s.mean():9.4f}  {s.std():8.4f}  {s.var():10.6f}")


def cv_with_full_pipeline():
    """Demonstrate that CV on Pipeline is the only correct approach."""
    print("\n=== CV Must Include the Full Pipeline ===")

    data = load_breast_cancer()
    X, y = data.data, data.target
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # WRONG: scale all data, then CV (leakage)
    X_scaled_all = StandardScaler().fit_transform(X)
    bad_scores = cross_val_score(
        LogisticRegression(max_iter=1000), X_scaled_all, y, cv=skf, scoring='accuracy'
    )
    print(f"\n  WRONG (scale before CV): {bad_scores.mean():.4f} ± {bad_scores.std():.4f}  ← leakage inflates score")

    # RIGHT: Pipeline inside CV
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model',  LogisticRegression(max_iter=1000))
    ])
    good_scores = cross_val_score(pipe, X, y, cv=skf, scoring='accuracy')
    print(f"  RIGHT (Pipeline in CV):  {good_scores.mean():.4f} ± {good_scores.std():.4f}  ← honest estimate")


def main():
    print("=" * 55)
    print("CROSS-VALIDATION — KFold, Stratified, cross_val_score")
    print("=" * 55)

    demonstrate_kfold_splits()
    cv_classification()
    cv_regression()
    compare_cv_strategies()
    cv_with_full_pipeline()


if __name__ == "__main__":
    main()
