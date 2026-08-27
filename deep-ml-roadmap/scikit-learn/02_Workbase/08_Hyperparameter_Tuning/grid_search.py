"""GridSearchCV — exhaustive hyperparameter search with cross-validation."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


def basic_grid_search():
    print("\n=== Basic GridSearchCV ===")

    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Parameter names reference the pipeline step using __
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model',  RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'model__n_estimators':    [50, 100, 200],
        'model__max_depth':       [None, 5, 10],
        'model__min_samples_leaf': [1, 5, 10],
    }
    # Total: 3 × 3 × 3 = 27 combos × 5 folds = 135 fits

    grid = GridSearchCV(
        pipe, param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        refit=True  # refit on full train data with best params (default)
    )
    grid.fit(X_train, y_train)

    print(f"\n  Best params: {grid.best_params_}")
    print(f"  Best CV F1:  {grid.best_score_:.4f}")
    print(f"  Test F1:     {grid.score(X_test, y_test):.4f}")
    print(f"  Test report:")
    print(classification_report(y_test, grid.best_estimator_.predict(X_test),
                                  target_names=['Malignant', 'Benign']))

    return grid, X_train


def analyze_grid_results(grid, X_train):
    """Dig into the GridSearchCV results dataframe."""
    print("\n=== Analyzing CV Results ===")

    results = pd.DataFrame(grid.cv_results_)
    results = results.sort_values('rank_test_score')

    print("\n  Top 5 configurations:")
    cols = ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
    print(results[cols].head().to_string(index=False))

    # Heatmap: n_estimators vs max_depth (fix min_samples_leaf)
    pivot = results[results['param_model__min_samples_leaf'] == 1].pivot_table(
        values='mean_test_score',
        index='param_model__max_depth',
        columns='param_model__n_estimators'
    )

    plt.figure(figsize=(7, 4))
    plt.imshow(pivot.values, cmap='RdYlGn', aspect='auto',
               vmin=pivot.values.min(), vmax=pivot.values.max())
    plt.colorbar(label='CV F1')
    plt.xticks(range(len(pivot.columns)), pivot.columns.astype(str), rotation=0)
    plt.yticks(range(len(pivot.index)),   [str(i) for i in pivot.index])
    plt.xlabel("n_estimators"); plt.ylabel("max_depth")
    plt.title("GridSearch Heatmap (min_samples_leaf=1)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            plt.text(j, i, f"{pivot.iloc[i,j]:.3f}", ha='center', va='center', fontsize=9)
    plt.tight_layout(); plt.show()


def grid_search_svm():
    print("\n=== GridSearchCV on SVM ===")

    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([('scaler', StandardScaler()), ('model', SVC(probability=True))])

    # SVM has two main hyperparameters: C and kernel
    param_grid = [
        {'model__kernel': ['linear'], 'model__C': [0.01, 0.1, 1, 10]},
        {'model__kernel': ['rbf'],    'model__C': [0.1, 1, 10, 100],
         'model__gamma': ['scale', 'auto', 0.01]},
    ]

    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train, y_train)

    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV AUC: {grid.best_score_:.4f}")
    print(f"  Test AUC:    {grid.score(X_test, y_test):.4f}")

    return grid


def main():
    print("=" * 55)
    print("GRIDSEARCHCV — Exhaustive Hyperparameter Search")
    print("=" * 55)

    grid_rf, X_train = basic_grid_search()
    analyze_grid_results(grid_rf, X_train)
    grid_svm = grid_search_svm()

    print("\n--- GridSearchCV Key Points ---")
    print("  1. Use __ to access nested pipeline params: 'model__n_estimators'")
    print("  2. Cost grows fast: 3×3×3 grid + 5-fold = 135 model fits")
    print("  3. refit=True (default) refits best model on full X_train")
    print("  4. best_estimator_ is the fitted pipeline you save/deploy")
    print("  5. Look at cv_results_ — it shows the full landscape, not just the winner")


if __name__ == "__main__":
    main()
