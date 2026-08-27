"""RandomizedSearchCV — efficient hyperparameter search for large spaces."""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import randint, uniform, loguniform
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


def random_search_demo():
    print("\n=== RandomizedSearchCV — Gradient Boosting Regressor ===")

    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model',  GradientBoostingRegressor(random_state=42))
    ])

    # scipy distributions for continuous/discrete parameters
    param_dist = {
        'model__n_estimators':      randint(50, 500),        # discrete uniform [50, 500)
        'model__learning_rate':     loguniform(0.005, 0.5),  # log-uniform [0.005, 0.5]
        'model__max_depth':         randint(2, 8),
        'model__min_samples_leaf':  randint(1, 30),
        'model__subsample':         uniform(0.5, 0.5),       # uniform [0.5, 1.0]
        'model__max_features':      uniform(0.3, 0.7),       # uniform [0.3, 1.0]
    }

    rand_search = RandomizedSearchCV(
        pipe, param_dist,
        n_iter=40,          # 40 random combinations × 5 folds = 200 fits
        cv=5,
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=1,
        refit=True
    )

    t0 = time.time()
    rand_search.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"\n  Best params: {rand_search.best_params_}")
    print(f"  Best CV R²:  {rand_search.best_score_:.4f}")
    print(f"  Test R²:     {rand_search.best_estimator_.score(X_test, y_test):.4f}")
    print(f"  Time:        {elapsed:.1f}s")

    return rand_search


def compare_grid_vs_random():
    print("\n=== Grid vs Random Search — Efficiency Comparison ===")

    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model',  RandomForestRegressor(random_state=42))
    ])

    # Grid: 3×3×3 = 27 combos
    param_grid = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth':    [5, 10, None],
        'model__min_samples_leaf': [1, 10, 20],
    }

    param_dist = {
        'model__n_estimators':     randint(10, 300),
        'model__max_depth':        randint(3, 20),
        'model__min_samples_leaf': randint(1, 30),
    }

    t0 = time.time()
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)
    grid_time = time.time() - t0

    t0 = time.time()
    rand = RandomizedSearchCV(pipe, param_dist, n_iter=27, cv=5, scoring='r2',
                               n_jobs=-1, random_state=42)
    rand.fit(X_train, y_train)
    rand_time = time.time() - t0

    print(f"\n  GridSearchCV (27 combos):")
    print(f"    Best CV R²: {grid.best_score_:.4f}, Test R²: {grid.best_estimator_.score(X_test, y_test):.4f}")
    print(f"    Time: {grid_time:.1f}s")

    print(f"\n  RandomizedSearchCV (27 random combos):")
    print(f"    Best CV R²: {rand.best_score_:.4f}, Test R²: {rand.best_estimator_.score(X_test, y_test):.4f}")
    print(f"    Time: {rand_time:.1f}s")
    print(f"\n  → Equivalent quality, same time. But Random scales to 1000s of params without exploding.")


def plot_search_landscape(rand_search):
    """Visualize the relationship between learning_rate and test R²."""
    import pandas as pd

    results = pd.DataFrame(rand_search.cv_results_)
    lr_vals = [p['model__learning_rate'] for p in results['params']]
    n_est   = [p['model__n_estimators'] for p in results['params']]
    scores  = results['mean_test_score'].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].scatter(lr_vals, scores, c=n_est, cmap='viridis', alpha=0.7)
    axes[0].set_xscale('log')
    axes[0].set_xlabel("learning_rate (log scale)")
    axes[0].set_ylabel("CV R²")
    axes[0].set_title("Learning Rate vs R² (color=n_estimators)")
    plt.colorbar(axes[0].collections[0], ax=axes[0], label='n_estimators')

    axes[1].scatter(n_est, scores, alpha=0.7, c='steelblue')
    axes[1].set_xlabel("n_estimators")
    axes[1].set_ylabel("CV R²")
    axes[1].set_title("n_estimators vs R²")

    plt.tight_layout(); plt.show()


def two_stage_tuning():
    """Best practice: RandomizedSearch wide → GridSearch narrow."""
    print("\n=== Two-Stage Tuning (Wide → Narrow) ===")

    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([('scaler', StandardScaler()),
                     ('model', GradientBoostingRegressor(random_state=42))])

    # Stage 1: wide random search
    rand = RandomizedSearchCV(
        pipe,
        {'model__n_estimators': randint(50, 500),
         'model__learning_rate': loguniform(0.005, 0.5),
         'model__max_depth': randint(2, 8)},
        n_iter=20, cv=3, scoring='r2', n_jobs=-1, random_state=42
    )
    rand.fit(X_train, y_train)
    best = rand.best_params_
    print(f"  Stage 1 best: lr={best['model__learning_rate']:.4f}, "
          f"n={best['model__n_estimators']}, depth={best['model__max_depth']}")

    # Stage 2: narrow grid around best
    lr_best = best['model__learning_rate']
    n_best  = best['model__n_estimators']
    grid = GridSearchCV(
        pipe,
        {'model__n_estimators': [max(10, n_best-50), n_best, n_best+50],
         'model__learning_rate': [lr_best/2, lr_best, lr_best*2],
         'model__max_depth': [best['model__max_depth'] - 1, best['model__max_depth'],
                               best['model__max_depth'] + 1]},
        cv=5, scoring='r2', n_jobs=-1
    )
    grid.fit(X_train, y_train)

    print(f"  Stage 2 best: {grid.best_params_}")
    print(f"  Final test R²: {grid.best_estimator_.score(X_test, y_test):.4f}")


def main():
    print("=" * 55)
    print("RANDOMIZEDSEARCHCV — Efficient Hyperparameter Search")
    print("=" * 55)

    rand_search = random_search_demo()
    compare_grid_vs_random()
    plot_search_landscape(rand_search)
    two_stage_tuning()


if __name__ == "__main__":
    main()
