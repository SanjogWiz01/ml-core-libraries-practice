"""House Price Prediction — California Housing with full Pipeline, CV, and tuning."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import randint, loguniform
import joblib

ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT / "models"


def load_data():
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    print(f"\n--- California Housing Dataset ---")
    print(f"Shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"\nTarget (median house value in $100k):")
    print(f"  min={df['MedHouseVal'].min():.2f}, max={df['MedHouseVal'].max():.2f}, "
          f"mean={df['MedHouseVal'].mean():.2f}")
    return df


def feature_engineering(df):
    """Add derived features before splitting."""
    df = df.copy()
    df['RoomsPerHousehold']    = df['AveRooms'] / df['HouseAge'].clip(lower=1)
    df['BedroomsPerRoom']      = df['AveBedrms'] / df['AveRooms'].clip(lower=1)
    df['PopulationPerHousehold'] = df['Population'] / df['AveOccup'].clip(lower=1)
    print(f"\nAdded 3 derived features. New shape: {df.shape}")
    return df


def build_pipeline(model):
    numeric_features = [c for c in ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                                     'Population', 'AveOccup', 'Latitude', 'Longitude',
                                     'RoomsPerHousehold', 'BedroomsPerRoom',
                                     'PopulationPerHousehold'] if c != 'MedHouseVal']

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
    ], remainder='drop')

    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])


def compare_models(X_train, y_train):
    models = {
        'LinearRegression':  LinearRegression(),
        'Ridge(α=1)':        Ridge(alpha=1.0),
        'RandomForest':      RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting':  GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                                         max_depth=4, random_state=42),
    }

    print(f"\n{'Model':22s}  {'CV RMSE':>9}  {'CV R²':>8}")
    print(f"{'-'*22}  {'-'*9}  {'-'*8}")

    results = {}
    for name, model in models.items():
        pipe = build_pipeline(model)
        rmse_cv = -cross_val_score(pipe, X_train, y_train, cv=5,
                                    scoring='neg_root_mean_squared_error', n_jobs=-1)
        r2_cv   = cross_val_score(pipe, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
        print(f"{name:22s}  {rmse_cv.mean():9.4f}  {r2_cv.mean():8.4f}")
        results[name] = (r2_cv.mean(), model)

    best_name = max(results, key=lambda k: results[k][0])
    print(f"\nBest baseline: {best_name}")
    return results


def tune_best_model(X_train, y_train):
    print("\n--- Tuning GradientBoosting ---")
    pipe = build_pipeline(GradientBoostingRegressor(random_state=42))

    param_dist = {
        'model__n_estimators':     randint(100, 600),
        'model__learning_rate':    loguniform(0.01, 0.3),
        'model__max_depth':        randint(2, 7),
        'model__min_samples_leaf': randint(5, 30),
        'model__subsample':        [0.7, 0.8, 0.9, 1.0],
    }

    search = RandomizedSearchCV(
        pipe, param_dist,
        n_iter=30, cv=5, scoring='r2',
        n_jobs=-1, random_state=42, verbose=0
    )
    search.fit(X_train, y_train)

    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV R²:  {search.best_score_:.4f}")
    return search.best_estimator_


def final_evaluation(best_pipe, X_test, y_test):
    y_pred = best_pipe.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"\n--- Final Test Set Evaluation ---")
    print(f"  MAE:  {mae:.4f} ($100k)  ≈ ${mae*100:.0f}k")
    print(f"  RMSE: {rmse:.4f} ($100k)  ≈ ${rmse*100:.0f}k")
    print(f"  R²:   {r2:.4f}  ({r2*100:.1f}% variance explained)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(y_test, y_pred, alpha=0.2, s=6)
    mn, mx = 0, max(y_test.max(), y_pred.max())
    axes[0].plot([mn, mx], [mn, mx], 'r--')
    axes[0].set_xlabel("Actual ($100k)"); axes[0].set_ylabel("Predicted ($100k)")
    axes[0].set_title("House Price: Actual vs Predicted")

    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.2, s=6)
    axes[1].axhline(0, color='r', linestyle='--')
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Residuals")
    axes[1].set_title("Residual Plot")
    plt.tight_layout(); plt.show()


def main():
    print("=" * 55)
    print("HOUSE PRICE PREDICTION — Full ML Pipeline")
    print("=" * 55)

    df = load_data()
    df = feature_engineering(df)

    X = df.drop(columns=['MedHouseVal'])
    y = df['MedHouseVal']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

    compare_models(X_train, y_train)
    best_pipe = tune_best_model(X_train, y_train)
    final_evaluation(best_pipe, X_test, y_test)

    # Save model
    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / "house_price_model.joblib"
    joblib.dump(best_pipe, model_path)
    print(f"\nModel saved: {model_path}")

    # Load and predict
    loaded = joblib.load(model_path)
    sample = X_test.iloc[:3]
    preds  = loaded.predict(sample)
    print(f"\nSample predictions (loaded model):")
    for i, (actual, pred) in enumerate(zip(y_test.iloc[:3], preds)):
        print(f"  Sample {i+1}: actual=${actual*100:.0f}k, predicted=${pred*100:.0f}k")


if __name__ == "__main__":
    main()
