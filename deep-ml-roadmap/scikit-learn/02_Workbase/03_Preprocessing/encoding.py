"""OneHotEncoder and OrdinalEncoder — categorical feature encoding."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def create_categorical_data(n=400, seed=42):
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        'age':          rng.integers(20, 60, n),
        'income':       rng.normal(50000, 15000, n),
        # Nominal — no order
        'city':         rng.choice(['London', 'Paris', 'Tokyo', 'NYC'], n),
        'job':          rng.choice(['engineer', 'teacher', 'doctor', 'artist'], n),
        # Ordinal — has a natural order
        'education':    rng.choice(['high_school', 'bachelors', 'masters', 'phd'], n),
        'satisfaction': rng.choice(['low', 'medium', 'high'], n),
    })

    # Target: roughly correlated with education and income
    edu_score = df['education'].map({'high_school': 0, 'bachelors': 1, 'masters': 2, 'phd': 3})
    df['target'] = ((df['income'] / 50000 + edu_score / 3) > 1.1).astype(int)
    return df


def main():
    print("=" * 55)
    print("CATEGORICAL ENCODING — OHE vs Ordinal")
    print("=" * 55)

    df = create_categorical_data()
    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ============================================================
    # OneHotEncoder — nominal columns
    # ============================================================
    print("\n=== OneHotEncoder (nominal: city, job) ===")
    nominal_cols = ['city', 'job']

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_ohe = ohe.fit_transform(X_train[nominal_cols])
    X_test_ohe  = ohe.transform(X_test[nominal_cols])

    print(f"Original shape:  {X_train[nominal_cols].shape}")
    print(f"Encoded shape:   {X_train_ohe.shape}")
    print(f"Feature names:   {ohe.get_feature_names_out(nominal_cols).tolist()}")

    # Show one example
    example = X_train[nominal_cols].iloc[0]
    example_enc = X_train_ohe[0]
    print(f"\nExample: city={example['city']}, job={example['job']}")
    print(f"Encoded: {example_enc}")

    # handle_unknown='ignore' → all zeros for unseen categories in test
    print("\n  handle_unknown='ignore' → unseen test categories → zero vector (safe)")

    # ============================================================
    # OrdinalEncoder — ordinal columns
    # ============================================================
    print("\n=== OrdinalEncoder (ordinal: education, satisfaction) ===")
    ordinal_cols = ['education', 'satisfaction']
    ordinal_categories = [
        ['high_school', 'bachelors', 'masters', 'phd'],
        ['low', 'medium', 'high']
    ]

    oe = OrdinalEncoder(categories=ordinal_categories)
    X_train_oe = oe.fit_transform(X_train[ordinal_cols])
    X_test_oe  = oe.transform(X_test[ordinal_cols])

    print("Education mapping:")
    for cat, val in zip(ordinal_categories[0], range(4)):
        print(f"  {cat:15s} → {val}")

    # ============================================================
    # Full pipeline: combine numeric + nominal + ordinal
    # ============================================================
    print("\n=== Full encoding pipeline ===")
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    numeric_cols = ['age', 'income']

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler())
    ])
    nominal_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe',     OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])
    ordinal_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('oe',      OrdinalEncoder(categories=ordinal_categories))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_cols),
        ('nom', nominal_transformer, nominal_cols),
        ('ord', ordinal_transformer, ordinal_cols),
    ])

    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    full_pipeline.fit(X_train, y_train)
    acc = accuracy_score(y_test, full_pipeline.predict(X_test))
    print(f"Full pipeline accuracy: {acc:.4f}")

    X_preprocessed = preprocessor.fit_transform(X_train)
    print(f"Input features: {X_train.shape[1]}  →  Output features: {X_preprocessed.shape[1]}")


if __name__ == "__main__":
    main()
