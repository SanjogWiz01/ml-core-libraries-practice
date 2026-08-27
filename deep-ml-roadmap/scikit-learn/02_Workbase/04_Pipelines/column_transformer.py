"""ColumnTransformer — the production-ready pattern for mixed-type data."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings("ignore")


def make_mixed_dataset(n=600, seed=42):
    """Realistic dataset with numeric, ordinal, and nominal features + missing values."""
    rng = np.random.default_rng(seed)

    age      = rng.integers(20, 65, n).astype(float)
    income   = rng.normal(55000, 20000, n)
    debt     = rng.exponential(10000, n)
    years    = rng.integers(0, 30, n).astype(float)
    edu      = rng.choice(['high_school', 'bachelors', 'masters', 'phd'], n)
    job      = rng.choice(['tech', 'finance', 'healthcare', 'education', 'other'], n)
    region   = rng.choice(['north', 'south', 'east', 'west'], n)
    married  = rng.choice(['yes', 'no'], n)

    # Introduce missing values
    age[rng.choice(n, 25)]   = np.nan
    income[rng.choice(n, 40)] = np.nan
    edu_arr = np.array(edu, dtype=object); edu_arr[rng.choice(n, 20)] = np.nan
    job_arr = np.array(job, dtype=object); job_arr[rng.choice(n, 15)] = np.nan

    # Target: credit approval
    edu_score = pd.Series(edu_arr).map({'high_school': 0, 'bachelors': 1, 'masters': 2, 'phd': 3}).fillna(1)
    target = ((income / 50000 + edu_score / 3 - debt / 30000 + years / 30) > 1.3).astype(int)

    return pd.DataFrame({
        'age': age, 'income': income, 'debt': debt, 'work_years': years,
        'education': edu_arr, 'job_sector': job_arr,
        'region': region, 'married': married,
        'target': target
    })


def build_preprocessor(numeric_cols, ordinal_cols, ordinal_categories, nominal_cols):
    """Build a ColumnTransformer for mixed-type features."""

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler())
    ])

    ordinal_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(categories=ordinal_categories,
                                   handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    nominal_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_cols),
        ('ord', ordinal_transformer, ordinal_cols),
        ('nom', nominal_transformer, nominal_cols),
    ], remainder='drop')  # drop any unlisted columns

    return preprocessor


def main():
    print("=" * 60)
    print("COLUMNTRANSFORMER — Mixed-Type Feature Preprocessing")
    print("=" * 60)

    df = make_mixed_dataset()
    print(f"\nDataset: {df.shape[0]} samples, {df.shape[1]-1} features")
    print(f"Target balance: {df['target'].value_counts().to_dict()}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Define column groups ---
    numeric_cols  = ['age', 'income', 'debt', 'work_years']
    ordinal_cols  = ['education']
    ordinal_cats  = [['high_school', 'bachelors', 'masters', 'phd']]
    nominal_cols  = ['job_sector', 'region', 'married']

    preprocessor = build_preprocessor(numeric_cols, ordinal_cols, ordinal_cats, nominal_cols)

    # --- Inspect what the preprocessor produces ---
    X_preprocessed = preprocessor.fit_transform(X_train)
    print(f"\n--- Shape after ColumnTransformer ---")
    print(f"  Input:  {X_train.shape}")
    print(f"  Output: {X_preprocessed.shape}")

    # Get transformed feature names (sklearn >= 1.0)
    try:
        feature_names_out = preprocessor.get_feature_names_out()
        print(f"\n  Feature names ({len(feature_names_out)}):")
        for name in feature_names_out:
            print(f"    {name}")
    except Exception:
        print("  (get_feature_names_out not available in this sklearn version)")

    # --- Full pipelines with different models ---
    print("\n--- Compare models with same preprocessor ---")

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        'RandomForest':       RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    best_score = -np.inf
    best_name  = None
    best_pipe  = None

    for model_name, model in models.items():
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='f1', n_jobs=-1)
        pipe.fit(X_train, y_train)
        test_score = pipe.score(X_test, y_test)
        print(f"  {model_name:22s}: CV F1={cv_scores.mean():.4f}±{cv_scores.std():.4f}, "
              f"test_acc={test_score:.4f}")
        if cv_scores.mean() > best_score:
            best_score = cv_scores.mean()
            best_name  = model_name
            best_pipe  = pipe

    print(f"\nBest model: {best_name}")
    print("\nClassification Report:")
    print(classification_report(y_test, best_pipe.predict(X_test),
                                 target_names=['Denied', 'Approved']))

    # --- make_column_selector: auto-detect column types ---
    print("\n--- Auto column detection with make_column_selector ---")
    auto_preprocessor = ColumnTransformer([
        ('num', StandardScaler(), make_column_selector(dtype_include=np.number)),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                make_column_selector(dtype_include=object)),
    ])
    auto_pipe = Pipeline([('prep', auto_preprocessor), ('model', RandomForestClassifier(random_state=42))])
    auto_cv = cross_val_score(auto_pipe, X_train, y_train, cv=3, scoring='accuracy').mean()
    print(f"  Auto-detected pipeline CV accuracy: {auto_cv:.4f}")
    print("  (Useful when you have many columns and just want dtype-based splitting)")


if __name__ == "__main__":
    main()
