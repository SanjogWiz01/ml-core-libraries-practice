"""Feature Selection — SelectKBest, RFE, and feature importance filtering."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import (
    SelectKBest, f_classif, chi2, mutual_info_classif,
    RFE, RFECV
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


def select_k_best_demo(X_train, X_test, y_train, y_test, feature_names):
    print("\n=== SelectKBest ===")

    # f_classif: ANOVA F-statistic (for numeric features)
    skb = SelectKBest(score_func=f_classif, k=10)
    X_train_sel = skb.fit_transform(X_train, y_train)
    X_test_sel  = skb.transform(X_test)

    selected_mask  = skb.get_support()
    selected_names = feature_names[selected_mask]
    scores         = skb.scores_[selected_mask]

    print(f"\n  Selected {X_train_sel.shape[1]} of {X_train.shape[1]} features:")
    for name, score in sorted(zip(selected_names, scores), key=lambda x: -x[1]):
        bar = '█' * int(score / skb.scores_.max() * 30)
        print(f"    {name:35s}: {score:8.2f}  {bar}")

    # Evaluate with selected vs all features
    pipe_all = Pipeline([
        ('scaler', StandardScaler()),
        ('model',  LogisticRegression(max_iter=1000))
    ])
    pipe_sel = Pipeline([
        ('select', SelectKBest(f_classif, k=10)),
        ('scaler', StandardScaler()),
        ('model',  LogisticRegression(max_iter=1000))
    ])

    cv_all = cross_val_score(pipe_all, X_train, y_train, cv=5, scoring='accuracy').mean()
    cv_sel = cross_val_score(pipe_sel, X_train, y_train, cv=5, scoring='accuracy').mean()
    print(f"\n  CV accuracy — all {X_train.shape[1]} features: {cv_all:.4f}")
    print(f"  CV accuracy — top 10 features:   {cv_sel:.4f}")

    # k sweep
    print("\n  k vs CV accuracy:")
    for k in [2, 5, 10, 15, 20, 25, X_train.shape[1]]:
        pipe = Pipeline([
            ('sel',   SelectKBest(f_classif, k=min(k, X_train.shape[1]))),
            ('scal',  StandardScaler()),
            ('model', LogisticRegression(max_iter=1000))
        ])
        cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean()
        print(f"    k={k:2d}: {cv:.4f}")

    # Different scoring functions
    print("\n  Scoring function comparison:")
    # chi2 requires non-negative features — use MinMaxScaler first
    X_nn = MinMaxScaler().fit_transform(X_train)
    for name, func in [('f_classif', f_classif),
                        ('mutual_info', mutual_info_classif)]:
        skb_f = SelectKBest(func, k=10)
        skb_f.fit(X_nn, y_train)
        sel = set(feature_names[skb_f.get_support()])
        print(f"    {name:20s}: {sorted(sel)[:5]} ...")


def rfe_demo(X_train, X_test, y_train, y_test, feature_names):
    print("\n=== RFE — Recursive Feature Elimination ===")

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    estimator = LogisticRegression(max_iter=1000)

    # RFE: eliminate least important features one by one
    rfe = RFE(estimator=estimator, n_features_to_select=10, step=1)
    rfe.fit(X_tr_s, y_train)

    selected = feature_names[rfe.support_]
    ranks    = rfe.ranking_[rfe.support_]

    print(f"\n  Selected 10 features (rank 1 = best):")
    for name, rank in sorted(zip(selected, ranks)):
        print(f"    {name}")

    acc = accuracy_score(y_test, rfe.predict(X_te_s))
    print(f"\n  Test accuracy with RFE-selected features: {acc:.4f}")

    # RFECV: automatically finds optimal n_features via cross-validation
    print("\n  RFECV — automatic n_features selection:")
    rfecv = RFECV(
        estimator=LogisticRegression(max_iter=1000),
        min_features_to_select=1,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    rfecv.fit(X_tr_s, y_train)

    print(f"  Optimal number of features: {rfecv.n_features_}")
    print(f"  Best CV accuracy: {rfecv.cv_results_['mean_test_score'].max():.4f}")

    # Plot n_features vs CV score
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1),
             rfecv.cv_results_['mean_test_score'], 'bo-')
    plt.axvline(rfecv.n_features_, color='r', linestyle='--',
                label=f'Optimal n={rfecv.n_features_}')
    plt.xlabel("Number of Features Selected")
    plt.ylabel("CV Accuracy")
    plt.title("RFECV: Finding Optimal Number of Features")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.show()


def feature_importance_filter(X_train, X_test, y_train, y_test, feature_names):
    print("\n=== Feature Importance Filtering (RandomForest) ===")

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(StandardScaler().fit_transform(X_train), y_train)
    importances = rf.feature_importances_

    threshold = 0.03
    selected = feature_names[importances >= threshold]
    print(f"\n  Threshold={threshold}: keep {len(selected)} of {len(feature_names)} features")

    pipe_full = Pipeline([
        ('scaler', StandardScaler()),
        ('rf',     RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    cv_full = cross_val_score(pipe_full, X_train, y_train, cv=5).mean()

    # Keep only important features
    idx = np.where(importances >= threshold)[0]
    cv_sel = cross_val_score(
        Pipeline([('scaler', StandardScaler()),
                  ('rf', RandomForestClassifier(n_estimators=100, random_state=42))]),
        X_train[:, idx], y_train, cv=5
    ).mean()

    print(f"  CV accuracy — all {len(feature_names)} features: {cv_full:.4f}")
    print(f"  CV accuracy — {len(selected)} important features: {cv_sel:.4f}")


def main():
    print("=" * 55)
    print("FEATURE SELECTION — SelectKBest, RFE, Importance")
    print("=" * 55)

    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = np.array(data.feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    select_k_best_demo(X_train, X_test, y_train, y_test, feature_names)
    rfe_demo(X_train, X_test, y_train, y_test, feature_names)
    feature_importance_filter(X_train, X_test, y_train, y_test, feature_names)


if __name__ == "__main__":
    main()
