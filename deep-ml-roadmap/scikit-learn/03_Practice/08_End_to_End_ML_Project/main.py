"""
End-to-End ML Project — Employee Salary Prediction

Full lifecycle:
    raw CSV → EDA → cleaning → split → preprocessing → feature engineering
    → ColumnTransformer → Pipeline → multiple models → cross-validation
    → hyperparameter tuning → final evaluation → joblib save → load → predict
"""

from pathlib import Path
import subprocess, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from scipy.stats import randint, loguniform

from sklearn.model_selection import (
    train_test_split, cross_val_score, RandomizedSearchCV, KFold
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT      = Path(__file__).parent.parent.parent
DATA_PATH = ROOT / "data" / "raw" / "employee_salary.csv"
MODEL_PATH = ROOT / "models" / "employee_salary_model.joblib"


# ===========================================================================
# STEP 1: GET DATA
# ===========================================================================

def ensure_data():
    if not DATA_PATH.exists():
        print("[Step 1] Generating dataset...")
        subprocess.run(
            [sys.executable, str(ROOT / "data" / "raw" / "generate_datasets.py")],
            check=True
        )
    else:
        print("[Step 1] Dataset found.")


def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"  Shape: {df.shape}")
    return df


# ===========================================================================
# STEP 2: EDA
# ===========================================================================

def eda(df):
    print("\n[Step 2] EDA")
    print(f"\n  Data types:\n{df.dtypes.to_string()}")
    print(f"\n  Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0].to_string()}")
    print(f"\n  Target (salary) stats:\n{df['salary'].describe().round(0).to_string()}")

    numeric_cols = df.select_dtypes(include='number').columns.drop('salary', errors='ignore')
    correlations = df[numeric_cols].corrwith(df['salary']).sort_values(key=abs, ascending=False)
    print(f"\n  Correlations with salary:")
    for col, corr in correlations.items():
        bar = '█' * int(abs(corr) * 20)
        print(f"    {col:25s}: {corr:+.4f}  {bar}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Target distribution
    axes[0].hist(df['salary'], bins=40, edgecolor='white')
    axes[0].set_title("Salary Distribution"); axes[0].set_xlabel("Salary ($)")

    # Salary by education
    df.boxplot(column='salary', by='education', ax=axes[1])
    axes[1].set_title("Salary by Education"); axes[1].set_xlabel("")

    # Salary by job_role
    df.boxplot(column='salary', by='job_role', ax=axes[2])
    axes[2].set_title("Salary by Job Role"); axes[2].tick_params(axis='x', rotation=30)

    plt.suptitle("")
    plt.tight_layout(); plt.show()


# ===========================================================================
# STEP 3: CLEANING
# ===========================================================================

def clean(df):
    print("\n[Step 3] Cleaning")
    df = df.copy()

    # Remove exact duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"  Removed {before - len(df)} duplicate rows")

    # Drop rows where target is missing
    df = df.dropna(subset=['salary'])
    print(f"  Rows with valid target: {len(df)}")

    return df


# ===========================================================================
# STEP 4: SPLIT (BEFORE PREPROCESSING)
# ===========================================================================

def split(df):
    print("\n[Step 4] Train/Test Split (before any preprocessing)")
    X = df.drop(columns=['salary'])
    y = df['salary']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Target train: mean=${y_train.mean():.0f}, std=${y_train.std():.0f}")
    return X_train, X_test, y_train, y_test


# ===========================================================================
# STEPS 5-7: FEATURE ENGINEERING + PREPROCESSING + PIPELINE
# ===========================================================================

def add_features(X_train, X_test):
    """Feature engineering applied to both sets consistently."""
    print("\n[Step 5-7] Feature Engineering + Preprocessing + Pipeline")

    for X in [X_train, X_test]:
        X['experience_to_age_ratio'] = (
            X['years_experience'].fillna(X['years_experience'].median()) /
            X['age'].clip(lower=22)
        )
    print("  Added: experience_to_age_ratio")
    return X_train, X_test


def build_full_pipeline(model):
    numeric_cols = ['age', 'years_experience', 'skills_count', 'overtime_hrs_week',
                    'experience_to_age_ratio']
    ordinal_cols = ['education', 'performance']
    ordinal_cats = [
        ['associate', 'bachelor', 'master', 'phd'],
        ['low', 'medium', 'high', 'excellent']
    ]
    nominal_cols = ['job_role', 'industry', 'city_size', 'gender']

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler())
    ])
    ordinal_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(categories=ordinal_cats,
                                    handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    nominal_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer,  numeric_cols),
        ('ord', ordinal_transformer,  ordinal_cols),
        ('nom', nominal_transformer,  nominal_cols),
    ])

    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])


# ===========================================================================
# STEP 8: MULTIPLE MODELS — CROSS-VALIDATION
# ===========================================================================

def compare_models(X_train, y_train):
    print("\n[Step 8] Cross-Validation (multiple models)")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        'LinearRegression':  LinearRegression(),
        'Ridge(α=1)':        Ridge(alpha=1.0),
        'RandomForest':      RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting':  GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                                         max_depth=4, random_state=42),
    }

    print(f"\n  {'Model':22s}  {'CV RMSE':>10}  {'CV R²':>9}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*9}")

    results = {}
    for name, model in models.items():
        pipe = build_full_pipeline(model)
        rmse_cv = -cross_val_score(pipe, X_train, y_train, cv=kf,
                                    scoring='neg_root_mean_squared_error', n_jobs=-1)
        r2_cv   =  cross_val_score(pipe, X_train, y_train, cv=kf,
                                    scoring='r2', n_jobs=-1)
        print(f"  {name:22s}  {rmse_cv.mean():10.1f}  {r2_cv.mean():9.4f}")
        results[name] = r2_cv.mean()

    best_name = max(results, key=results.get)
    print(f"\n  Best baseline: {best_name}")
    return best_name


# ===========================================================================
# STEP 9: HYPERPARAMETER TUNING
# ===========================================================================

def tune(X_train, y_train):
    print("\n[Step 9] Hyperparameter Tuning (RandomizedSearchCV)")

    pipe = build_full_pipeline(GradientBoostingRegressor(random_state=42))
    param_dist = {
        'model__n_estimators':     randint(100, 500),
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


# ===========================================================================
# STEP 10: FINAL EVALUATION
# ===========================================================================

def evaluate(best_pipe, X_test, y_test):
    print("\n[Step 10] Final Evaluation on Test Set")
    y_pred = best_pipe.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"\n  MAE:  ${mae:,.0f}   → avg prediction error")
    print(f"  RMSE: ${rmse:,.0f}  → penalizes large errors more")
    print(f"  R²:   {r2:.4f}  → model explains {r2*100:.1f}% of salary variance")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(y_test, y_pred, alpha=0.3, s=10)
    mn = min(y_test.min(), y_pred.min()); mx = max(y_test.max(), y_pred.max())
    axes[0].plot([mn, mx], [mn, mx], 'r--')
    axes[0].set_xlabel("Actual Salary ($)"); axes[0].set_ylabel("Predicted Salary ($)")
    axes[0].set_title("End-to-End: Actual vs Predicted")

    residuals = y_test - y_pred
    axes[1].hist(residuals, bins=40, edgecolor='white')
    axes[1].axvline(0, color='r', linestyle='--')
    axes[1].set_xlabel("Residual ($)"); axes[1].set_title("Residual Distribution")
    plt.tight_layout(); plt.show()

    return y_pred


# ===========================================================================
# STEP 11: SAVE MODEL
# ===========================================================================

def save_model(best_pipe):
    print("\n[Step 11] Saving model with joblib")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, MODEL_PATH)
    size_kb = MODEL_PATH.stat().st_size / 1024
    print(f"  Saved: {MODEL_PATH}  ({size_kb:.1f} KB)")


# ===========================================================================
# STEP 12: LOAD AND PREDICT
# ===========================================================================

def load_and_predict():
    print("\n[Step 12] Load Model → Predict on New Data")
    loaded_pipe = joblib.load(MODEL_PATH)
    print(f"  Loaded: {type(loaded_pipe).__name__} pipeline")

    # New employees — raw data, same format as training
    new_employees = pd.DataFrame({
        'age':               [28, 45, 35],
        'years_experience':  [5.0, 20.0, 10.0],
        'education':         ['bachelor', 'phd', 'master'],
        'job_role':          ['engineer', 'manager', 'analyst'],
        'industry':          ['tech', 'finance', 'healthcare'],
        'city_size':         ['large', 'large', 'medium'],
        'gender':            ['M', 'F', 'M'],
        'skills_count':      [8, 12, 7],
        'overtime_hrs_week': [5.0, 3.0, np.nan],   # NaN handled by pipeline
        'performance':       ['high', 'excellent', 'medium'],
        'experience_to_age_ratio': [5/28, 20/45, 10/35]  # add engineered feature
    })

    predictions = loaded_pipe.predict(new_employees)

    print(f"\n  Salary predictions:")
    for i, (_, row) in enumerate(new_employees.iterrows()):
        print(f"    {row['education']:8s} {row['job_role']:10s} "
              f"{row['industry']:12s} ({row['city_size']}): "
              f"${predictions[i]:,.0f}")

    print("\n  Pipeline applies preprocessing (imputation, scaling, encoding) automatically.")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 60)
    print("END-TO-END ML PROJECT — Employee Salary Prediction")
    print("=" * 60)
    print("Workflow: raw CSV → EDA → clean → split → features")
    print("         → Pipeline → CV → tune → evaluate → save → predict")

    ensure_data()
    df = load_data()
    eda(df)
    df = clean(df)

    X_train, X_test, y_train, y_test = split(df)
    X_train, X_test = add_features(X_train, X_test)

    best_baseline = compare_models(X_train, y_train)
    best_pipe = tune(X_train, y_train)
    y_pred = evaluate(best_pipe, X_test, y_test)
    save_model(best_pipe)
    load_and_predict()

    print("\n" + "=" * 60)
    print("End-to-End project complete.")
    print("Key takeaways:")
    print("  1. Split BEFORE preprocessing — no exceptions.")
    print("  2. Pipeline handles imputation/scaling/encoding safely.")
    print("  3. RandomizedSearchCV + CV gives honest performance estimates.")
    print("  4. joblib.dump saves the full pipeline — preprocessing included.")
    print("  5. Loaded model handles raw data with NaNs automatically.")


if __name__ == "__main__":
    main()
