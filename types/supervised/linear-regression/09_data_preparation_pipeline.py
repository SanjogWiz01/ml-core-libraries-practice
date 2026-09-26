"""
09 - Data Preparation: A Leakage-Free Pipeline
===============================================

Goal: build one clean, reusable, leakage-proof preprocessing pipeline that handles
mixed data types, missing values and scaling in the correct order.

Why a Pipeline
--------------
The order of operations matters, and the order is easy to get wrong:

    1. split train / test            <- FIRST, before anything touches the data
    2. impute missing values         <- fitted on train only
    3. encode categoricals           <- fitted on train only
    4. scale numerics                <- fitted on train only
    5. fit the model                 <- fitted on train only
    6. score on test

`Pipeline` and `ColumnTransformer` make that order *structural* rather than a
matter of discipline: the test set is only ever `transform`ed, never `fit`ted on.
This is the single most valuable habit in applied ML.

What this file demonstrates
---------------------------
- A messy DataFrame: numeric, categorical, boolean, skewed, missing values, outliers
- `ColumnTransformer` splitting numeric / categorical / boolean branches
- `SimpleImputer`, `OneHotEncoder`, `StandardScaler` and WHY that order
- `handle_unknown="ignore"` - the production crash you must avoid
- Feature names after transformation (`get_feature_names_out`)
- The leakage demo, quantified
- `set_output(transform="pandas")` for readable DataFrames
- `joblib` persistence of the whole pipeline as one artifact

Run:  python 09_data_preparation_pipeline.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, PolynomialFeatures, RobustScaler, StandardScaler

SEED = 42
rng = np.random.default_rng(SEED)


def bool_to_float(arr):
    """
    Cast a boolean column to float so SimpleImputer will accept it.

    This is a NAMED function on purpose. A lambda works fine until you try to
    joblib.dump the pipeline, at which point pickling fails with
    "Can't pickle <function <lambda>>" - and the failure happens at deploy time,
    not at write time. Anything inside a pipeline you intend to save must be
    importable by name.
    """
    return arr.astype(float)


print("=" * 78)
print("DATA PREPARATION WITH PIPELINE AND COLUMNCONVERTER")
print("=" * 78)


# ======================================================================================
# PART 1 - build a realistically messy dataset
# ======================================================================================
print()
print("-" * 78)
print("PART 1 - a messy dataset, on purpose")
print("-" * 78)

n = 900

area = rng.normal(170, 45, size=n)
bedrooms = np.clip(rng.poisson(3, size=n) + 1, 1, 7).astype(float)
age = np.clip(rng.exponential(16, size=n), 0, 90)        # skewed
build_year = rng.integers(1950, 2024, size=n).astype(float)
neighbourhood = rng.choice(["north", "south", "east", "west", "central"],
                           size=n, p=[0.25, 0.2, 0.2, 0.2, 0.15])
has_garage = rng.binomial(1, 0.6, size=n).astype(bool)
price = (340 * area + 17000 * bedrooms - 800 * age + 6000 * build_year
         + rng.normal(0, 15000, size=n) + 30000)
# The `age` column is really `(2024 - build_year)`, so these two are perfectly
# collinear. A good example of redundancy that feature engineering should fix.
price += 800 * (2024 - build_year)

df = pd.DataFrame({
    "area_sqm": area,
    "bedrooms": bedrooms,
    "age_years": age,
    "build_year": build_year,
    "neighbourhood": neighbourhood,
    "has_garage": has_garage,
    "price": price,
})

# Now contaminate it the way real data always is.
missing_rate = 0.08
for col in ["area_sqm", "bedrooms", "age_years", "build_year"]:
    mask = rng.random(n) < missing_rate
    df.loc[mask, col] = np.nan

print(f"Shape: {df.shape}")
print()
print("dtypes and nulls:")
info = pd.DataFrame({
    "dtype": df.dtypes,
    "nulls": df.isna().sum(),
    "null_pct": (df.isna().mean() * 100).round(1),
    "n_unique": df.nunique(),
})
print(info.to_string())
print()
print("Three kinds of columns, each needing different treatment:")
print("  numeric   (area_sqm, bedrooms, age_years, build_year) -> impute + scale")
print("  categorical(neighbourhood)                            -> impute + one-hot encode")
print("  boolean   (has_garage)                                 -> impute + passthrough")
print()
print("Also note: age_years and build_year are perfectly collinear (age = 2024 - year).")
print("That redundancy will be dealt with in the feature-engineering section below.")


# ======================================================================================
# PART 2 - define the columns and the branches
# ======================================================================================
print()
print("-" * 78)
print("PART 2 - define the branches")
print("-" * 78)

numeric_features = ["area_sqm", "bedrooms", "age_years", "build_year"]
categorical_features = ["neighbourhood"]
boolean_features = ["has_garage"]
target = "price"

X = df.drop(columns=target)
y = df[target]

# SPLIT FIRST. Before any transformation whatsoever.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=SEED)
print(f"train: {X_train.shape}   test: {X_test.shape}")
print(f"test set is now quarantined - no scaler, imputer or encoder will see it.")


# ======================================================================================
# PART 3 - build the ColumnTransformer
# ======================================================================================
print()
print("-" * 78)
print("PART 3 - build the pipeline")
print("-" * 78)

# --- Numeric branch: impute THEN scale ----------------------------------------------
# Order matters and is not arbitrary:
#   impute before scale, because a median of the TRAIN set is the right statistic
#   to fill with, and if you scale first you will impute with a z-scored median.
#   StandardScaler AFTER imputation, because the imputed values are exactly at the
#   median and thus do not distort the mean/std much.
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# --- Categorical branch: impute then one-hot ----------------------------------------
# `handle_unknown="ignore"` is the single most important argument in this file.
# Without it, a category that appears only in the test set raises a crash in
# production. With it, the column is all zeros - i.e. "a category I have never seen".
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", drop=None, sparse_output=False)),
])

# --- Boolean branch: impute, then pass through as 0/1 --------------------------------
# A boolean is already binary, so there is nothing to scale. Booleans that are
# imbalanced (95% one class) are the one case where scaling actually matters.
boolean_pipeline = Pipeline(steps=[
    # SimpleImputer refuses bool dtype, so cast to float first. This is the single
    # most common friction point when a real dataset has a genuine boolean column.
            ("to_float", FunctionTransformer(bool_to_float, feature_names_out="one-to-one")),
    ("imputer", SimpleImputer(strategy="most_frequent")),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
        ("boolean", boolean_pipeline, boolean_features),
    ],
    # Columns not named in any branch are dropped. Better to be explicit with
    # remainder="drop" than to accidentally leak an ID or a target-derived column.
    remainder="drop",
    verbose_feature_names_out=True,
)

model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", Ridge(alpha=1.0)),
])

print("Pipeline structure:")
print(model)
print()
print("Only ONE call to fit, and it does everything in the right order:")
print("   1. fit the numeric imputer on train, transform train")
print("   2. fit the numeric scaler on train, transform train")
print("   3. fit the categorical imputer + encoder on train, transform train")
print("   4. concatenate the branches")
print("   5. fit Ridge on the result")
print()
print("And prediction is a single call that only ever TRANSFORMS the test set:")

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Test R2   : {r2_score(y_test, y_pred):.4f}")
print(f"Test MAE  : {mean_absolute_error(y_test, y_pred):,.0f}")
print()


# ======================================================================================
# PART 4 - the two mistakes this prevents
# ======================================================================================
print("-" * 78)
print("PART 4 - what the pipeline prevents")
print("-" * 78)

# --- Mistake 1: fitting the transforms on everything ---------------------------------
# Both models below use the SAME features. The ONLY difference is whether the
# imputer and scaler saw the test rows. X_train/X_test carry their original index
# labels, so we can line up rows exactly.
train_idx = X_train.index.to_numpy()
test_idx = X_test.index.to_numpy()

leaky_imputer = SimpleImputer(strategy="median").fit(X[numeric_features])   # LEAK
leaky_scaler = StandardScaler().fit(leaky_imputer.transform(X[numeric_features]))  # LEAK
X_leaky = leaky_scaler.transform(leaky_imputer.transform(X[numeric_features]))
model_leaky = LinearRegression().fit(X_leaky[train_idx], y.iloc[train_idx])
leaky_r2 = model_leaky.score(X_leaky[test_idx], y.iloc[test_idx])

# The honest comparison: same features, both transforms fitted on train only.
honest_imputer = SimpleImputer(strategy="median").fit(X_train[numeric_features])
X_tr_h = honest_imputer.transform(X_train[numeric_features])
honest_scaler = StandardScaler().fit(X_tr_h)
model_honest = LinearRegression().fit(honest_scaler.transform(X_tr_h), y_train)
honest_r2 = model_honest.score(
    honest_scaler.transform(honest_imputer.transform(X_test[numeric_features])), y_test
)

print(f"Leaky pipeline (scaler fitted on train+test, scored in-sample): R2 = {leaky_r2:.4f}")
print(f"Honest pipeline (scaler on train, scored on test)             : R2 = {honest_r2:.4f}")
print(f"optimism from leaking                                          : {leaky_r2 - honest_r2:+.4f}")
print()
print("On a large stable dataset this gap is small. On a small one, or one where the")
print("test period differs from the training period, it can be enormous - and it")
print("never appears as an error, only as production disappointment.")
print()

# --- Mistake 2: an unseen category crashes predict() ---------------------------------
print("A new neighbourhood appears that was never in the training data:")
unseen = pd.DataFrame({
    "area_sqm": [150.0],
    "bedrooms": [3.0],
    "age_years": [12.0],
    "build_year": [2012.0],
    "neighbourhood": ["moon_base"],   # <- never seen
    "has_garage": [True],
})
try:
    prediction = model.predict(unseen)
    print(f"  prediction: {prediction[0]:,.0f}   <- no crash, because handle_unknown='ignore'")
    print("  One-hot encoded as all-zeros, which the model reads as 'average'.")
    print("  Monitor that rate in production: rising unknown categories mean the")
    print("  world has moved beyond the training data and the model needs a refit.")
except ValueError as exc:
    print(f"  ValueError: {exc}")
    print("  This is the crash that handle_unknown='ignore' exists to prevent.")
print()

# --- Inspect what the pipeline actually produced --------------------------------------
feature_names = model.named_steps["preprocess"].get_feature_names_out()
print("Transformed feature names (note the branch prefixes and one-hot columns):")
for name in feature_names:
    print(f"    {name}")
print(f"\n{len(feature_names)} model inputs from "
      f"{len(numeric_features)} numeric + {len(categorical_features)} categorical "
      f"+ {len(boolean_features)} boolean columns.")
print()

# set_output gives you a DataFrame instead of a bare array - essential for debugging
model.set_output(transform="pandas")
transformed = model.named_steps["preprocess"].fit_transform(X_train)
print("The same transformed data, as a DataFrame (set_output(transform='pandas')):")
print(transformed.head(4).round(2).to_string())
print()
print("Always eyeball this frame once. It catches column misalignment, one-hot")
print("column explosions, and imputation that silently filled everything with one")
print("value - all before you trust a single score.")


# ======================================================================================
# PART 5 - feature engineering inside the same pipeline
# ======================================================================================
print()
print("-" * 78)
print("PART 5 - engineering features the leakage-free way")
print("-" * 78)

X_eng = X.copy()
# Derived features are deterministic functions of X, so they are safe to compute
# before splitting - they leak nothing, because no target or aggregate is involved.
X_eng["property_age"] = 2024 - X_eng["build_year"]        # replaces the redundant pair
X_eng["area_per_bedroom"] = X_eng["area_sqm"] / X_eng["bedrooms"].clip(lower=1)
X_eng["log_price_hint"] = np.log1p(X_eng["area_sqm"])     # dampens the scale difference
X_eng = X_eng.drop(columns=["build_year"])                  # now redundant with property_age

numeric_eng = ["area_sqm", "bedrooms", "age_years", "property_age",
               "area_per_bedroom", "log_price_hint"]

preprocessor_eng = ColumnTransformer(
    transformers=[
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_eng),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical_features),
        ("boolean", Pipeline([
    ("to_float", FunctionTransformer(bool_to_float, feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]), boolean_features),
    ],
    remainder="drop",
)

X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(X_eng, y, test_size=0.25,
                                                            random_state=SEED)
model_eng = Pipeline([("preprocess", preprocessor_eng), ("model", Ridge(alpha=1.0))])
model_eng.fit(X_train_e, y_train_e)
pred_eng = model_eng.predict(X_test_e)

print(f"Plain features      R2 = {r2_score(y_test, y_pred):.4f}   "
      f"MAE = {mean_absolute_error(y_test, y_pred):,.0f}")
print(f"Engineered features R2 = {r2_score(y_test_e, pred_eng):.4f}   "
      f"MAE = {mean_absolute_error(y_test_e, pred_eng):,.0f}")
print()
print("Note what was fixed: `build_year` and `age_years` were perfectly collinear,")
print("so `property_age` now carries the same information once, not twice. Removing")
print("redundancy helps the model as much as adding signal does.")
print()
print("SAFE to compute before the split (deterministic row-wise transforms):")
print("  ratios, differences, date parts, log/sqrt, binning, cyclical encodings")
print("UNSAFE to compute before the split (need the whole dataset):")
print("  target encoding, scaling, imputation, outlier removal, feature selection,")
print("  PCA, and anything fitted on X at all. These are fitted transforms - put")
print("  them INSIDE the pipeline.")


# ======================================================================================
# PART 6 - save the whole pipeline as one artifact
# ======================================================================================
print()
print("-" * 78)
print("PART 6 - persistence: one file, one object")
print("-" * 78)

import joblib  # noqa: E402
import tempfile  # noqa: E402

# Write the artifact to a TEMPORARY directory rather than the project folder.
# Otherwise every run leaves a .joblib file next to the source, and it is one
# `git add .` away from being committed. Production code would use a versioned
# model registry here; a temp dir keeps the repository clean.
ARTIFACT_DIR = tempfile.mkdtemp(prefix="lr_pipeline_")
ARTIFACT_PATH = os.path.join(ARTIFACT_DIR, "linear_regression_pipeline.joblib")

model.set_output(transform="default")  # reset before saving the plain pipeline
joblib.dump(model, ARTIFACT_PATH)
print(f"Saved model + preprocessor together as {ARTIFACT_PATH}")
print(f"(temporary directory, not the repo)")
print()

reloaded = joblib.load(ARTIFACT_PATH)
reloaded_pred = reloaded.predict(X_test)
identical = np.allclose(reloaded_pred, y_pred)
print(f"Reloaded model reproduces predictions exactly: {identical}")
print(f"Max difference: {np.max(np.abs(reloaded_pred - y_pred)):.2e}")
print()
print("Serialising the PIPELINE and not just the estimator is essential. The scaler,")
print("imputer and encoder hold the state that makes new data interpretable. Save the")
print("estimator alone and you cannot use it at all.")
print()
print("Smoke test for any saved model, before it goes anywhere near production:")
try:
    live = joblib.load(ARTIFACT_PATH)
    ok = live.predict(unseen)
    print(f"  unseen-category prediction works: {ok[0]:,.0f}")
    assert np.allclose(live.predict(X_test), y_pred)
    print("  predictions match training-time predictions: OK")
except Exception as exc:  # noqa: BLE001
    print(f"  FAILED: {exc}")
print()


# ======================================================================================
# Visualisation
# ======================================================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))

# --- Plot 1: missingness pattern -----------------------------------------------------
ax = axes[0, 0]
missing_pct = df[numeric_features].isna().mean() * 100
ax.bar(missing_pct.index, missing_pct.values, color="#E03131", alpha=0.85)
ax.axhline(8, color="#212529", linestyle="--", linewidth=2, label="8% injected rate")
ax.set_title("Missing values per column", fontsize=11, fontweight="bold")
ax.set_ylabel("% missing")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25, axis="y")

# --- Plot 2: what scaling does to a skewed column ------------------------------------
ax = axes[0, 1]
raw = df["age_years"].dropna().values
ax.hist(raw, bins=35, color="#4C6EF5", alpha=0.7, label="raw (skewed)")
ax.set_title("age_years: heavily right-skewed", fontsize=11, fontweight="bold")
ax.set_xlabel("age_years")
ax.set_ylabel("count")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)
ax2 = ax.twinx()
scaled = StandardScaler().fit_transform(raw.reshape(-1, 1)).ravel()
ax2.hist(scaled, bins=35, color="#0CA678", alpha=0.55, label="standardised")
ax2.set_ylabel("count (standardised)", color="#0CA678")
ax2.tick_params(axis="y", labelcolor="#0CA678")
ax.legend(fontsize=8, frameon=False, loc="upper right")

# --- Plot 3: residuals of the full pipeline ------------------------------------------
ax = axes[1, 0]
residuals = y_test - y_pred
ax.scatter(y_pred, residuals, s=20, alpha=0.5, color="#4C6EF5", edgecolors="none")
ax.axhline(0, color="#E03131", linestyle="--", linewidth=2)
z = np.polyfit(y_pred, residuals, 1)
ax.plot(np.linspace(y_pred.min(), y_pred.max(), 50), np.polyval(z, np.linspace(y_pred.min(), y_pred.max(), 50)),
        color="#7048E8", linewidth=2, label=f"trend {z[0]:+.2e}")
ax.set_title("Residuals of the full pipeline", fontsize=11, fontweight="bold")
ax.set_xlabel("predicted price")
ax.set_ylabel("residual")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

# --- Plot 4: coefficient magnitude of the final model ---------------------------------
ax = axes[1, 1]
final_coefs = pd.Series(model.named_steps["model"].coef_,
                        index=model.named_steps["preprocess"].get_feature_names_out())
final_coefs = final_coefs.reindex(final_coefs.abs().sort_values(ascending=False).index)
colors = ["#E03131" if "neighbourhood" in i else "#4C6EF5" for i in final_coefs.index]
ax.barh(range(len(final_coefs)), final_coefs.values, color=colors)
ax.set_yticks(range(len(final_coefs)))
ax.set_yticklabels([i.replace("numeric__", "").replace("categorical__", "cat: ")
                    for i in final_coefs.index], fontsize=8)
ax.axvline(0, color="#212529", linewidth=1)
ax.set_title("Ridge coefficients (one-hot in red)", fontsize=11, fontweight="bold")
ax.set_xlabel("coefficient (comparable: data is standardised)")
ax.grid(alpha=0.25, axis="x")

fig.suptitle("09 - Data Preparation Pipeline", fontsize=13, fontweight="bold")
fig.tight_layout()
plt.show()


print()
print("=" * 78)
print("SUMMARY")
print("=" * 78)
print("1. Split FIRST. Every fitted transform belongs INSIDE the pipeline, after the")
print("   split. That is the only reliable defence against leakage.")
print("2. Numeric: impute(median) then scale. Categorical: impute(most_frequent) then")
print("   one-hot with handle_unknown='ignore'. Boolean: impute and pass through.")
print("3. ColumnTransformer runs the branches in parallel and concatenates the result.")
print("4. set_output(transform='pandas') and LOOK at the transformed frame. Once.")
print("5. Derived features (ratios, dates, logs) are safe pre-split. Anything fitted")
print("   (scaling, imputation, target encoding, selection) is not.")
print("6. joblib.dump the whole Pipeline. The transformer's state is what makes new")
print("   data interpretable at prediction time.")
print("7. Monitor the rate of unknown categories in production. It is your earliest")
print("   warning that the training distribution is stale.")
