"""Missing Data Preprocessing — explicit demonstration of the full imputation pipeline."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def create_dataset_with_all_miss_types(n=800, seed=42):
    """Dataset with numeric, ordinal, and nominal missingness."""
    rng = np.random.default_rng(seed)

    age       = rng.integers(18, 65, n).astype(float)
    income    = rng.normal(55000, 20000, n)
    bmi       = rng.normal(26, 5, n).clip(15, 50)
    education = rng.choice(['high_school', 'bachelor', 'master', 'phd'], n,
                            p=[0.40, 0.35, 0.18, 0.07])
    city      = rng.choice(['urban', 'suburban', 'rural'], n, p=[0.5, 0.3, 0.2])
    smoker    = rng.choice([0, 1], n, p=[0.75, 0.25])
    exercise  = rng.choice(['none', 'light', 'moderate', 'intense'], n,
                            p=[0.25, 0.30, 0.30, 0.15])

    # Create target
    edu_map = {'high_school': 0, 'bachelor': 1, 'master': 2, 'phd': 3}
    edu_num = pd.Series(education).map(edu_map)
    logit = 0.02*(age-40) + income/100000 + edu_num*0.3 - smoker*0.8 + rng.normal(0, 0.5, n)
    target = (1 / (1 + np.exp(-logit)) > 0.5).astype(int)

    # Introduce missing values with different patterns
    # MCAR — missing completely at random
    age[rng.choice(n, int(n*0.08), replace=False)] = np.nan          # 8%
    income[rng.choice(n, int(n*0.12), replace=False)] = np.nan       # 12%

    # MAR — missing at random (slightly correlated with other vars)
    # Higher BMI → more likely to skip reporting
    bmi_miss_mask = (bmi > 35) & (rng.uniform(0, 1, n) < 0.4)
    bmi[bmi_miss_mask] = np.nan                                        # ~15% of high BMI

    # Categorical missing
    education_arr = np.array(education, dtype=object)
    education_arr[rng.choice(n, int(n*0.06), replace=False)] = np.nan # 6%
    city_arr = np.array(city, dtype=object)
    city_arr[rng.choice(n, int(n*0.04), replace=False)] = np.nan      # 4%
    exercise_arr = np.array(exercise, dtype=object)
    exercise_arr[rng.choice(n, int(n*0.09), replace=False)] = np.nan  # 9%

    df = pd.DataFrame({
        'age': age, 'income': income, 'bmi': bmi,
        'education': education_arr, 'city': city_arr,
        'exercise': exercise_arr, 'smoker': smoker,
        'target': target
    })
    return df


def analyze_missingness(df):
    print("\n--- Missing Value Analysis ---")
    miss = df.isnull().sum()
    miss_pct = df.isnull().mean() * 100

    print(f"\n  {'Column':15s}  {'Missing':>8}  {'%':>7}  {'Type':>10}")
    print(f"  {'-'*15}  {'-'*8}  {'-'*7}  {'-'*10}")
    type_map = {'age': 'numeric', 'income': 'numeric', 'bmi': 'numeric',
                'education': 'ordinal', 'city': 'nominal', 'exercise': 'ordinal',
                'smoker': 'binary'}
    for col in df.columns:
        if col == 'target':
            continue
        dtype = type_map.get(col, 'unknown')
        print(f"  {col:15s}  {miss[col]:8d}  {miss_pct[col]:6.1f}%  {dtype:>10}")

    total_complete = df.dropna().shape[0]
    print(f"\n  Complete rows (no NaN): {total_complete}/{len(df)} ({total_complete/len(df)*100:.1f}%)")
    print("  → Dropping rows would lose 30%+ of data. Imputation is necessary.")


def build_imputation_pipeline(model):
    """
    Full preprocessing pipeline:
    Missing values → impute → encode → scale → model
    """
    numeric_cols  = ['age', 'income', 'bmi']
    ordinal_cols  = ['education', 'exercise']
    ordinal_cats  = [
        ['high_school', 'bachelor', 'master', 'phd'],
        ['none', 'light', 'moderate', 'intense']
    ]
    nominal_cols  = ['city']
    binary_cols   = ['smoker']  # no imputation needed (no missing)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # robust to outliers
        ('scaler',  StandardScaler())
    ])

    ordinal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # fill with mode
        ('encoder', OrdinalEncoder(categories=ordinal_cats,
                                    handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    nominal_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer,  numeric_cols),
        ('ord', ordinal_transformer,  ordinal_cols),
        ('nom', nominal_transformer,  nominal_cols),
        ('bin', 'passthrough',        binary_cols),
    ])

    return Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])


def compare_strategies(X_train, y_train, numeric_cols):
    """Compare different imputation strategies for numeric columns."""
    print("\n--- Imputation Strategy Comparison (numeric only) ---")

    for strategy in ['mean', 'median', 'most_frequent']:
        imp = SimpleImputer(strategy=strategy)
        X_imp = imp.fit_transform(X_train[numeric_cols])
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        cv = cross_val_score(rf, X_imp, y_train, cv=5, scoring='f1').mean()
        print(f"  strategy='{strategy}': CV F1={cv:.4f}  (stats: {imp.statistics_.round(2)})")


def visualize_imputation_effect(df, numeric_cols):
    """Show distribution before/after imputation."""
    fig, axes = plt.subplots(2, len(numeric_cols), figsize=(12, 6))

    for i, col in enumerate(numeric_cols):
        original = df[col].dropna()
        imp = SimpleImputer(strategy='median')
        imputed_all = imp.fit_transform(df[[col]]).ravel()

        axes[0, i].hist(original, bins=30, edgecolor='white', color='steelblue')
        axes[0, i].set_title(f"{col} — original (no NaN)")

        axes[1, i].hist(imputed_all, bins=30, edgecolor='white', color='firebrick')
        axes[1, i].set_title(f"{col} — after imputation")

    plt.suptitle("Distribution Before/After Median Imputation")
    plt.tight_layout(); plt.show()


def main():
    print("=" * 60)
    print("MISSING DATA PREPROCESSING")
    print("Missing values → SimpleImputer → ColumnTransformer → Pipeline → Model")
    print("=" * 60)

    df = create_dataset_with_all_miss_types()
    analyze_missingness(df)

    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    compare_strategies(X_train, y_train, ['age', 'income', 'bmi'])
    visualize_imputation_effect(df, ['age', 'income', 'bmi'])

    # --- Full Pipeline: Missing → Impute → Encode → Model ---
    print("\n--- Full Pipeline Evaluation ---")
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest':       RandomForestClassifier(n_estimators=100, random_state=42),
    }

    for name, model in models.items():
        pipe = build_imputation_pipeline(model)
        cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring='f1', n_jobs=-1)
        pipe.fit(X_train, y_train)
        test_f1 = cross_val_score(pipe, X_test, y_test, cv=3, scoring='f1').mean()
        y_pred = pipe.predict(X_test)
        from sklearn.metrics import f1_score
        print(f"\n  {name}:")
        print(f"    CV F1:   {cv.mean():.4f} ± {cv.std():.4f}")
        print(f"    Test F1: {f1_score(y_test, y_pred):.4f}")

    # --- Show the imputer statistics learned from train ---
    print("\n--- Imputer Statistics (learned from X_train only) ---")
    pipe = build_imputation_pipeline(RandomForestClassifier(n_estimators=100, random_state=42))
    pipe.fit(X_train, y_train)

    num_imputer = pipe.named_steps['preprocessor'].named_transformers_['num'].named_steps['imputer']
    print(f"  Numeric imputer (median) statistics: {num_imputer.statistics_.round(2)}")
    print(f"  → X_test is imputed with THESE values (from X_train) — no leakage.")


if __name__ == "__main__":
    main()
