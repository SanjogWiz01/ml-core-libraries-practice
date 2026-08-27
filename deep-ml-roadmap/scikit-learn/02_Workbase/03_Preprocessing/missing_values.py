"""SimpleImputer — handling missing values with all strategies."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def create_dataset_with_missing(n=500, seed=42):
    """Synthetic dataset with realistic missingness patterns."""
    rng = np.random.default_rng(seed)

    age      = rng.integers(18, 70, n).astype(float)
    income   = rng.normal(50000, 15000, n)
    score    = rng.normal(60, 15, n).clip(0, 100)
    city     = rng.choice(['urban', 'suburban', 'rural'], n)
    target   = (0.3 * age + 0.0001 * income + 0.4 * score > 35).astype(int)

    # Introduce missingness (MCAR — missing completely at random)
    age[rng.choice(n, 40, replace=False)]    = np.nan   # 8% missing
    income[rng.choice(n, 60, replace=False)] = np.nan   # 12% missing
    city_arr = np.array(city, dtype=object)
    city_arr[rng.choice(n, 30, replace=False)] = np.nan  # 6% missing

    return pd.DataFrame({
        'age': age, 'income': income, 'score': score,
        'city': city_arr, 'target': target
    })


def main():
    print("=" * 55)
    print("MISSING VALUES — SimpleImputer Strategies")
    print("=" * 55)

    df = create_dataset_with_missing()

    print("\n--- Missing value summary ---")
    print(df.isnull().sum())
    print(f"\nDataset shape: {df.shape}")

    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ============================================================
    # Numeric imputation strategies
    # ============================================================
    numeric_cols = ['age', 'income', 'score']
    cat_col = 'city'

    print("\n--- Numeric imputation strategies ---")
    strategies = ['mean', 'median', 'most_frequent', ('constant', 0)]

    results = []
    for strategy in strategies:
        if isinstance(strategy, tuple):
            name, fill = strategy
            imp_num = SimpleImputer(strategy=name, fill_value=fill)
        else:
            name = strategy
            imp_num = SimpleImputer(strategy=name)

        imp_cat = SimpleImputer(strategy='most_frequent')

        # Fit on train, transform both
        X_train_num = imp_num.fit_transform(X_train[numeric_cols])
        X_test_num  = imp_num.transform(X_test[numeric_cols])
        X_train_cat = imp_cat.fit_transform(X_train[[cat_col]])
        X_test_cat  = imp_cat.transform(X_test[[cat_col]])

        # Simple encoding for the cat column
        from sklearn.preprocessing import OrdinalEncoder
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_train_cat_enc = enc.fit_transform(X_train_cat)
        X_test_cat_enc  = enc.transform(X_test_cat)

        import numpy as np
        X_tr = np.hstack([X_train_num, X_train_cat_enc])
        X_te = np.hstack([X_test_num,  X_test_cat_enc])

        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_tr, y_train)
        acc = accuracy_score(y_test, clf.predict(X_te))
        results.append((name, acc))
        print(f"  strategy='{name}': accuracy={acc:.4f}")

    # ============================================================
    # Show what imputer learned
    # ============================================================
    imp = SimpleImputer(strategy='median')
    imp.fit(X_train[numeric_cols])
    print(f"\n--- Imputer statistics (learned from train) ---")
    for col, stat in zip(numeric_cols, imp.statistics_):
        print(f"  {col}: {stat:.2f}")

    # ============================================================
    # Compare: drop rows vs impute
    # ============================================================
    print("\n--- Drop rows vs impute (numeric only) ---")
    X_num = X[numeric_cols]
    y_aligned = y

    # Drop NaN rows
    mask = X_num.notna().all(axis=1)
    X_clean = X_num[mask]
    y_clean = y[mask]
    Xtr, Xte, ytr, yte = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
    clf.fit(Xtr, ytr)
    acc_drop = accuracy_score(yte, clf.predict(Xte))
    print(f"  Drop rows (n={len(X_clean)}): accuracy={acc_drop:.4f}")

    imp_all = SimpleImputer(strategy='median')
    X_imp = imp_all.fit_transform(X_num)
    Xtr, Xte, ytr, yte = train_test_split(X_imp, y, test_size=0.2, random_state=42)
    clf.fit(Xtr, ytr)
    acc_imp = accuracy_score(yte, clf.predict(Xte))
    print(f"  Impute (n={len(X_num)}):  accuracy={acc_imp:.4f}")
    print("  → Imputing keeps all data and usually performs at least as well.")


if __name__ == "__main__":
    main()
