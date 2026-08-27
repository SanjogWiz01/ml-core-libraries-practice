"""Student Performance Prediction — regression with real preprocessing."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import sys
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).parent.parent.parent
DATA_PATH = ROOT / "data" / "raw" / "student_performance.csv"


def ensure_data():
    if not DATA_PATH.exists():
        print("Generating dataset...")
        subprocess.run([sys.executable, str(ROOT / "data" / "raw" / "generate_datasets.py")],
                       check=True)


def load_and_explore(path):
    df = pd.read_csv(path)
    print(f"\n--- Dataset shape: {df.shape} ---")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nTarget distribution:\n{df['final_score'].describe()}")
    return df


def eda(df):
    print("\n--- EDA ---")
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    correlations = df[numeric_cols].corr()['final_score'].drop('final_score').sort_values(ascending=False)
    print("\nCorrelations with final_score:")
    for col, corr in correlations.items():
        bar = '█' * int(abs(corr) * 20)
        sign = '+' if corr >= 0 else '-'
        print(f"  {col:25s}: {corr:+.4f}  {sign}{bar}")


def build_preprocessor(X_train):
    numeric_cols     = ['study_hours_per_day', 'attendance_pct', 'prev_score']
    ordinal_cols     = ['parent_education']
    ordinal_cats     = [['none', 'high_school', 'bachelors', 'masters']]
    nominal_cols     = ['gender']
    passthrough_cols = ['has_tutoring', 'internet_access']

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler())
    ])
    ordinal_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(categories=ordinal_cats))
    ])
    nominal_transformer = Pipeline([
        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])

    return ColumnTransformer([
        ('num', numeric_transformer,  numeric_cols),
        ('ord', ordinal_transformer,  ordinal_cols),
        ('nom', nominal_transformer,  nominal_cols),
        ('pass', 'passthrough',       passthrough_cols),
    ])


def train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor):
    models = {
        'LinearRegression':  LinearRegression(),
        'Ridge(α=10)':       Ridge(alpha=10),
        'RandomForest':      RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting':  GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    }

    print(f"\n{'Model':22s}  {'CV RMSE':>9}  {'Test MAE':>9}  {'Test R²':>9}")
    print(f"{'-'*22}  {'-'*9}  {'-'*9}  {'-'*9}")

    results = {}
    for name, model in models.items():
        pipe = Pipeline([('preprocessor', preprocessor), ('model', model)])
        cv = -cross_val_score(pipe, X_train, y_train, cv=5,
                               scoring='neg_root_mean_squared_error', n_jobs=-1)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        print(f"{name:22s}  {cv.mean():9.4f}  {mae:9.4f}  {r2:9.4f}")
        results[name] = {'pipe': pipe, 'r2': r2, 'mae': mae}

    best_name = max(results, key=lambda k: results[k]['r2'])
    print(f"\nBest model: {best_name} (R²={results[best_name]['r2']:.4f})")
    return results[best_name]['pipe'], results


def plot_results(y_test, y_pred):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(y_test, y_pred, alpha=0.4, s=15)
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    axes[0].plot([mn, mx], [mn, mx], 'r--')
    axes[0].set_xlabel("Actual Score"); axes[0].set_ylabel("Predicted Score")
    axes[0].set_title("Actual vs Predicted")

    residuals = y_test - y_pred
    axes[1].hist(residuals, bins=30, edgecolor='white')
    axes[1].axvline(0, color='r', linestyle='--')
    axes[1].set_xlabel("Residual"); axes[1].set_title("Residual Distribution")
    plt.tight_layout(); plt.show()


def main():
    print("=" * 55)
    print("STUDENT PERFORMANCE — Regression Project")
    print("=" * 55)

    ensure_data()
    df = load_and_explore(DATA_PATH)
    eda(df)

    X = df.drop(columns=['final_score'])
    y = df['final_score']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

    preprocessor = build_preprocessor(X_train)
    best_pipe, results = train_and_evaluate(X_train, X_test, y_train, y_test, preprocessor)

    y_pred = best_pipe.predict(X_test)
    plot_results(y_test.values, y_pred)

    # Feature importance from the best model (if tree-based)
    model = best_pipe.named_steps.get('model')
    if hasattr(model, 'feature_importances_'):
        X_trans = best_pipe.named_steps['preprocessor'].fit_transform(X_train)
        print(f"\nTop feature importances (shape: {X_trans.shape[1]} encoded features):")
        importances = model.feature_importances_
        for i in np.argsort(importances)[::-1][:10]:
            print(f"  feature_{i}: {importances[i]:.4f}")


if __name__ == "__main__":
    main()
