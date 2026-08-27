"""Model Comparison — CV-based multi-model benchmarking on the Wine dataset."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report


def build_models():
    return {
        'GaussianNB':          Pipeline([('m', GaussianNB())]),
        'LogisticRegression':  Pipeline([('s', StandardScaler()),
                                          ('m', LogisticRegression(max_iter=1000, random_state=42))]),
        'KNN(k=5)':            Pipeline([('s', StandardScaler()),
                                          ('m', KNeighborsClassifier(n_neighbors=5))]),
        'KNN(k=9)':            Pipeline([('s', StandardScaler()),
                                          ('m', KNeighborsClassifier(n_neighbors=9))]),
        'DecisionTree(d=5)':   Pipeline([('m', DecisionTreeClassifier(max_depth=5, random_state=42))]),
        'SVM(linear)':         Pipeline([('s', StandardScaler()),
                                          ('m', SVC(kernel='linear', C=1.0, probability=True,
                                                     random_state=42))]),
        'SVM(rbf)':            Pipeline([('s', StandardScaler()),
                                          ('m', SVC(kernel='rbf', C=10.0, gamma='scale',
                                                     probability=True, random_state=42))]),
        'RandomForest(100)':   Pipeline([('m', RandomForestClassifier(n_estimators=100,
                                                                        random_state=42))]),
        'ExtraTrees(100)':     Pipeline([('m', ExtraTreesClassifier(n_estimators=100,
                                                                      random_state=42))]),
        'GradientBoosting':    Pipeline([('m', GradientBoostingClassifier(n_estimators=100,
                                                                            learning_rate=0.1,
                                                                            random_state=42))]),
    }


def run_cv_comparison(models, X, y, cv):
    """Run cross-validate for all models and return a results DataFrame."""
    rows = []
    print(f"\n{'Model':22s}  {'Acc Mean':>9}  {'Acc Std':>8}  {'F1 Mean':>8}  {'Time':>6}")
    print(f"{'-'*22}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*6}")

    for name, pipe in models.items():
        results = cross_validate(
            pipe, X, y, cv=cv,
            scoring=['accuracy', 'f1_macro'],
            return_train_score=True,
            n_jobs=-1
        )
        row = {
            'Model':         name,
            'Train Acc':     results['train_accuracy'].mean(),
            'Val Acc Mean':  results['test_accuracy'].mean(),
            'Val Acc Std':   results['test_accuracy'].std(),
            'Val F1 Mean':   results['test_f1_macro'].mean(),
            'Val F1 Std':    results['test_f1_macro'].std(),
            'Fit Time (s)':  results['fit_time'].mean(),
        }
        rows.append(row)
        print(f"{name:22s}  {row['Val Acc Mean']:9.4f}  {row['Val Acc Std']:8.4f}  "
              f"{row['Val F1 Mean']:8.4f}  {row['Fit Time (s)']:6.3f}s")

    return pd.DataFrame(rows).sort_values('Val Acc Mean', ascending=False)


def plot_comparison(df_results):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Accuracy bar chart with error bars
    y = np.arange(len(df_results))
    axes[0].barh(y, df_results['Val Acc Mean'],
                  xerr=df_results['Val Acc Std'],
                  color='steelblue', alpha=0.8, capsize=4)
    axes[0].barh(y, df_results['Train Acc'],
                  alpha=0.3, color='firebrick')
    axes[0].set_yticks(y); axes[0].set_yticklabels(df_results['Model'])
    axes[0].set_xlabel("Accuracy")
    axes[0].set_title("CV Accuracy ± Std (blue=val, red=train)")
    axes[0].axvline(df_results['Val Acc Mean'].max(), color='k', linestyle='--', linewidth=0.8)
    axes[0].set_xlim(0.7, 1.05)

    # F1 scatter: mean vs std (low std = more stable)
    for _, row in df_results.iterrows():
        axes[1].scatter(row['Val F1 Mean'], row['Val F1 Std'], s=80, zorder=5)
        axes[1].annotate(row['Model'], (row['Val F1 Mean'], row['Val F1 Std']),
                          fontsize=7, ha='left', va='bottom',
                          xytext=(3, 3), textcoords='offset points')
    axes[1].set_xlabel("CV F1 Macro (higher=better)")
    axes[1].set_ylabel("CV F1 Std (lower=more stable)")
    axes[1].set_title("F1 Mean vs Stability")
    axes[1].grid(alpha=0.3)
    axes[1].invert_yaxis()  # lower std is better, so top-right = best

    plt.tight_layout(); plt.show()


def final_test_evaluation(best_model_name, models, X_train, X_test, y_train, y_test, class_names):
    print(f"\n--- Final Test Evaluation: {best_model_name} ---")
    best_pipe = models[best_model_name]
    best_pipe.fit(X_train, y_train)
    y_pred = best_pipe.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=class_names))


def main():
    print("=" * 60)
    print("MODEL COMPARISON — Wine Dataset (10 classifiers)")
    print("=" * 60)

    data = load_wine()
    X, y = data.data, data.target
    class_names = data.target_names

    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features, {len(class_names)} classes")
    print(f"Classes: {list(class_names)}")
    print(f"Distribution: {dict(zip(class_names, np.bincount(y)))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    models = build_models()

    print("\n--- 10-Fold Stratified CV Results ---")
    df_results = run_cv_comparison(models, X_train, y_train, skf)

    print(f"\n--- Top 3 Models ---")
    print(df_results[['Model', 'Val Acc Mean', 'Val Acc Std', 'Val F1 Mean']].head(3).to_string(index=False))

    best_name = df_results.iloc[0]['Model']
    final_test_evaluation(best_name, models, X_train, X_test, y_train, y_test, class_names)

    plot_comparison(df_results)

    print("\n--- Key Insights ---")
    print("  Ensemble methods (RF, GB) usually win on tabular data.")
    print("  SVM with RBF kernel is competitive but needs scaling and is slower.")
    print("  Logistic Regression is a strong baseline — if it's close, prefer it (interpretable).")
    print("  High train acc + low val acc → overfitting. Watch for Decision Trees.")
    print("  Use 10-fold CV for small datasets like Wine to reduce variance.")


if __name__ == "__main__":
    main()
