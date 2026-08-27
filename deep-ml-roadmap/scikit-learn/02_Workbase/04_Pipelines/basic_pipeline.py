"""Pipeline — chain preprocessing and model into one leakage-free object."""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from pathlib import Path


def demo_leakage_problem(X, y):
    """Show why preprocessing outside a pipeline causes leakage."""
    print("\n=== The Leakage Problem ===")

    # WRONG: scaler sees test data
    scaler = StandardScaler()
    X_scaled_all = scaler.fit_transform(X)  # leaks test into train stats
    Xtr, Xte, ytr, yte = train_test_split(X_scaled_all, y, test_size=0.2, random_state=42)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(Xtr, ytr)
    wrong_score = lr.score(Xte, yte)
    print(f"  WRONG (scaler fits all data): test_acc = {wrong_score:.4f}  ← artificially inflated")

    # RIGHT: scaler only sees train data
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)   # transform with train stats
    lr.fit(Xtr_s, ytr)
    right_score = lr.score(Xte_s, yte)
    print(f"  RIGHT (manual, fit train only): test_acc = {right_score:.4f}")

    # BEST: Pipeline does it automatically
    pipe = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(max_iter=1000))])
    pipe.fit(Xtr, ytr)
    pipe_score = pipe.score(Xte, yte)
    print(f"  BEST (Pipeline):              test_acc = {pipe_score:.4f}  ← same as right, guaranteed safe")


def build_and_evaluate_pipeline(X, y):
    """Build a Pipeline, evaluate with CV, and demonstrate its interface."""
    print("\n=== Pipeline API ===")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # step 1
        ('scaler',  StandardScaler()),                  # step 2
        ('model',   LogisticRegression(C=1.0, max_iter=1000, random_state=42))  # step 3
    ])

    # Fit the whole pipeline
    pipe.fit(X_train, y_train)

    # Access individual steps
    print(f"\n  Steps: {[name for name, _ in pipe.steps]}")
    print(f"  Scaler mean (first 3): {pipe.named_steps['scaler'].mean_[:3].round(3)}")
    print(f"  Model coef shape: {pipe.named_steps['model'].coef_.shape}")

    # Predict — preprocessing is automatically applied
    y_pred = pipe.predict(X_test)
    print(f"\n  Test accuracy: {pipe.score(X_test, y_test):.4f}")
    print(f"  Probabilities shape: {pipe.predict_proba(X_test).shape}")

    # Cross-validate the full pipeline — correct way
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy')
    print(f"  5-Fold CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return pipe, X_test, y_test


def pipeline_with_hyperparameter_tuning(X, y):
    """GridSearchCV on a Pipeline — tune model AND preprocessor params."""
    from sklearn.model_selection import GridSearchCV

    print("\n=== GridSearchCV on Pipeline ===")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ('imputer', SimpleImputer()),
        ('scaler',  StandardScaler()),
        ('model',   LogisticRegression(max_iter=1000))
    ])

    # Parameter names use __ to navigate the pipeline steps
    param_grid = {
        'imputer__strategy': ['mean', 'median'],
        'model__C':          [0.01, 0.1, 1.0, 10.0],
    }

    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)

    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV acc: {grid.best_score_:.4f}")
    print(f"  Test acc:    {grid.best_estimator_.score(X_test, y_test):.4f}")


def save_and_load_pipeline(pipe, path):
    """Demonstrate joblib save/load."""
    print("\n=== Save & Load Pipeline ===")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipe, path)
    print(f"  Saved: {path}")

    loaded = joblib.load(path)
    print(f"  Loaded pipeline type: {type(loaded).__name__}")
    return loaded


def main():
    print("=" * 55)
    print("PIPELINE — Leakage-Free Preprocessing + Model")
    print("=" * 55)

    data = load_breast_cancer()
    X, y = data.data, data.target

    demo_leakage_problem(X, y)
    pipe, X_test, y_test = build_and_evaluate_pipeline(X, y)
    pipeline_with_hyperparameter_tuning(X, y)

    models_dir = Path(__file__).parent.parent.parent / "models"
    loaded_pipe = save_and_load_pipeline(pipe, models_dir / "logistic_pipeline.joblib")
    print(f"  Loaded model acc: {loaded_pipe.score(X_test, y_test):.4f}  ← identical to saved")


if __name__ == "__main__":
    main()
