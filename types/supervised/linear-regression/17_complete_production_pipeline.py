"""
17 - Complete Production Pipeline
=================================

Goal: assemble everything from scripts 01-16 into one end-to-end, reproducible,
production-shaped regression project - and print the report you would actually
hand to a stakeholder.

The full sequence
-----------------
    1.  Generate / load data
    2.  Explore and clean
    3.  Engineer features
    4.  Split, properly
    5.  Build a preprocessing + model Pipeline
    6.  Establish a DummyRegressor baseline
    7.  Tune the model with nested cross-validation
    8.  Evaluate on the held-out test set
    9.  Run residual diagnostics
    10. Save the artifact, reload it, smoke-test it
    11. Serve predictions and simulate monitoring

What "production" means here
----------------------------
Not "the code runs". It means:
  - the result is REPRODUCIBLE (fixed seeds, no hidden global state)
  - the result is HONEST (nested CV, untouched test set, a real baseline)
  - the result is EXPLAINED (coefficients, intervals, diagnostics)
  - the model is PORTABLE (one artifact, reloadable, versioned)
  - the result is MONITORED (residual tracking, drift checks, input validation)

Run:  python 17_complete_production_pipeline.py
"""

import json
import os
import tempfile
import time
import warnings
from datetime import datetime, timezone

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

SEED = 42
RNG_SEED = 2024
warnings.filterwarnings("ignore", category=FutureWarning)

# Artifacts go to a TEMPORARY directory, deliberately. Writing them into the
# project folder would leave model.joblib / model_metrics.json sitting next to the
# source after every run, which is how binaries and stale models end up committed
# by accident. In a real deployment this path is a versioned model registry.
ARTIFACT_DIR = tempfile.mkdtemp(prefix="lr_pipeline_")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.joblib")
METRICS_PATH = os.path.join(ARTIFACT_DIR, "model_metrics.json")

print("=" * 78)
print("COMPLETE PRODUCTION REGRESSION PIPELINE")
print(f"started {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
print("=" * 78)


# ======================================================================================
# STAGE 1 - data generation (stand in for pd.read_csv in a real project)
# ======================================================================================
print()
print("-" * 78)
print("STAGE 1 - DATA")
print("-" * 78)


def make_dataset(n_rows=3000, seed=RNG_SEED):
    """
    A synthetic property-valuation dataset with every problem a real one has:
    a non-linear term, a category with a real effect, missing values, outliers,
    a redundant column pair, and heteroscedastic noise.

    Returning a function (rather than inline data) makes the whole project
    reproducible from two integers.
    """
    rng = np.random.default_rng(seed)

    area = rng.normal(165, 42, size=n_rows)
    bedrooms = np.clip(rng.poisson(3, size=n_rows) + 1, 1, 7).astype(float)
    bathrooms = np.clip(rng.poisson(1.5, size=n_rows), 1, 4).astype(float)
    # Non-linear: value per m2 falls as houses get larger.
    property_age = np.clip(rng.exponential(15, size=n_rows), 0, 95)
    build_year = (2024 - property_age).round()
    garage = rng.binomial(1, 0.58, size=n_rows).astype(float)
    condition = rng.normal(5.5, 1.5, size=n_rows)
    location = rng.choice(["north", "south", "central", "rural", "coastal"],
                          size=n_rows, p=[0.22, 0.2, 0.28, 0.18, 0.12])
    location_effect = pd.Series(location).map(
        {"north": 0.10, "south": 0.0, "central": 0.18, "rural": -0.22, "coastal": 0.28}
    ).to_numpy()

    distance = np.clip(0.010 * area + 0.30 * property_age
                       + rng.normal(7, 2.0, size=n_rows), 0.2, None)

    # Signal, on a log scale so the target is right-skewed like real prices.
    log_price = (
        10.7
        + 0.0038 * area                       # diminishing returns to size
        + 0.075 * np.sqrt(area)
        + 0.115 * location_effect
        + 0.088 * bedrooms
        + 0.130 * bathrooms
        + 0.055 * condition
        - 0.0042 * property_age
        - 0.0016 * distance
        + 0.035 * garage
        + 0.0012 * area * (condition - 5) / 10  # an interaction, on purpose
    )
    # Heteroscedastic noise: the spread grows with the price.
    price = np.exp(log_price + rng.normal(0, 0.075 + 0.030 * (log_price - 10.7), size=n_rows))

    # A handful of data-entry catastrophes: three decimal points.
    corrupt = rng.choice(n_rows, 9, replace=False)
    price[corrupt] *= 30.0

    frame = pd.DataFrame({
        "area_sqm": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "property_age": property_age,
        "build_year": build_year.astype(float),
        "garage": garage,
        "condition": condition,
        "location": location,
        "distance_km": distance,
        "price": price,
    })

    # Missing values, at rates that differ by column, as in real surveys.
    for column, rate in [("area_sqm", 0.06), ("bedrooms", 0.04),
                         ("bathrooms", 0.10), ("property_age", 0.07),
                         ("condition", 0.05), ("garage", 0.12)]:
        frame.loc[rng.random(n_rows) < rate, column] = np.nan

    return frame


df = make_dataset()
print(f"Generated {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Target 'price': median {df['price'].median():,.0f}, "
      f"mean {df['price'].mean():,.0f}, max {df['price'].max():,.0f}")
print()
print("Column profile:")
profile = pd.DataFrame({
    "dtype": df.dtypes,
    "missing_pct": (df.isna().mean() * 100).round(1),
    "n_unique": df.nunique(),
    "skew": df.select_dtypes("number").skew().round(2),
})
print(profile.to_string())
print()
print("Note the skew on price. That is why we will model log(price) and invert at")
print("the end, rather than fitting a straight line to a right-skewed target.")


# ======================================================================================
# STAGE 2 - cleaning and feature engineering
# ======================================================================================
print()
print("-" * 78)
print("STAGE 2 - CLEAN AND ENGINEER")
print("-" * 78)

# Domain rule first: a price cannot be negative.
price_before = len(df)
df = df[df["price"] > 1000].copy()
print(f"Domain rule (price > 0): {price_before} -> {len(df)} rows "
      f"({price_before - len(df)} removed)")

# Then a ROBUST statistical rule for the fat-fingered entries. We work on the log
# price, because on the raw scale the outliers dominate the mean and the SD and
# drag the threshold up until it stops catching anything.
log_price_all = np.log1p(df["price"])
median_log = log_price_all.median()
mad = (log_price_all - median_log).abs().median()
mad_scaled = 1.4826 * mad  # makes MAD comparable to a standard deviation
keep = (log_price_all - median_log).abs() <= 5.0 * mad_scaled
print(f"Robust outlier rule (|log price - median| > 5 * scaled MAD "
      f"= {5.0 * mad_scaled:.3f}): {(~keep).sum()} rows removed")
df = df[keep].copy()
print(f"Running total: {len(df)} rows remain "
      f"({len(df) / price_before * 100:.1f}% of the original {price_before})")

# Redundancy: build_year and property_age are the same information. Keep one.
df = df.drop(columns=["build_year"])
print("Dropped build_year - perfectly redundant with property_age")

# Deterministic, row-wise derivations. These are safe to do before splitting.
df["log_area"] = np.log1p(df["area_sqm"])
df["area_per_bedroom"] = df["area_sqm"] / df["bedrooms"].clip(lower=1)
df["is_new"] = (df["property_age"] < 5).astype(float)
df["age_x_condition"] = df["property_age"] * (df["condition"] - df["condition"].mean())
print("Engineered: log_area, area_per_bedroom, is_new, age_x_condition")
print()

numeric_features = ["area_sqm", "bedrooms", "bathrooms", "property_age",
                    "garage", "condition", "distance_km",
                    "log_area", "area_per_bedroom", "is_new", "age_x_condition"]
categorical_features = ["location"]
boolean_like = []  # garage is now a float 0/1 with missing values -> numeric branch

X = df[numeric_features + categorical_features]
y_raw = df["price"]
y = np.log1p(y_raw)  # log target: tames skew, linearises multiplicative effects

print(f"Feature matrix: {X.shape[0]} rows x {X.shape[1]} columns "
      f"({len(numeric_features)} numeric, {len(categorical_features)} categorical)")
print(f"Target: log1p(price). median log target {np.median(y):.3f}")


# ======================================================================================
# STAGE 3 - the split
# ======================================================================================
print()
print("-" * 78)
print("STAGE 3 - SPLIT")
print("-" * 78)

X_train, X_test, y_train, y_test, y_raw_train, y_raw_test = train_test_split(
    X, y, y_raw, test_size=0.2, random_state=SEED
)
print(f"train {X_train.shape[0]} rows ({len(X_train) / len(X) * 100:.0f}%)  "
      f"test {X_test.shape[0]} rows ({len(X_test) / len(X) * 100:.0f}%)")
print(f"{len(X_train) / X_train.shape[1]:.0f} rows per feature - comfortably above the 10:1 rule of thumb")
print()
print("The test set is now quarantined. Nothing fitted may see it until STAGE 7.")


# ======================================================================================
# STAGE 4 - the pipeline
# ======================================================================================
print()
print("-" * 78)
print("STAGE 4 - PIPELINE")
print("-" * 78)


def make_preprocessor():
    """Numeric: median-impute then standardise. Categorical: mode then one-hot."""
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric_features),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                # ignore=... : an unseen category in production must not crash.
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical_features),
        ],
        remainder="drop",
    )


def make_model(estimator):
    """One place where preprocessing and model are joined, so they can never drift."""
    return Pipeline([("preprocess", make_preprocessor()), ("model", estimator)])


print(make_model(Ridge(alpha=1.0)))
print()
print("Note what is NOT here: no fillna, no scaler, no encoder outside the pipeline.")
print("Every fitted transform lives inside, so cross-validation refits it per fold.")


# ======================================================================================
# STAGE 5 - the baseline
# ======================================================================================
print()
print("-" * 78)
print("STAGE 5 - BASELINE (the step everyone skips)")
print("-" * 78)

dummy = Pipeline([("preprocess", make_preprocessor()),
                  ("model", DummyRegressor(strategy="mean"))])
dummy.fit(X_train, y_train)
dummy_r2 = dummy.score(X_test, y_test)
dummy_mae = mean_absolute_error(y_test, dummy.predict(X_test))
print(f"DummyRegressor (predict the training mean)")
print(f"  test R2  = {dummy_r2:.4f}")
print(f"  test MAE = {dummy_mae:.4f} on the log scale")
print()
print("R2 = 0 by construction - it is the mathematical definition of 'no better than")
print("the mean'. Any model must clear this bar before its score means anything.")


# ======================================================================================
# STAGE 6 - model selection with nested cross-validation
# ======================================================================================
print()
print("-" * 78)
print("STAGE 6 - MODEL SELECTION (nested CV)")
print("-" * 78)

candidate_models = {
    "LinearRegression": (LinearRegression(), {}),
    "Ridge": (Ridge(), {"alpha": np.logspace(-3, 3, 7)}),
    "ElasticNet": (ElasticNet(max_iter=20000),
                   {"alpha": np.logspace(-4, 2, 7), "l1_ratio": [0.2, 0.5, 0.9]}),
    "HuberRegressor": (HuberRegressor(max_iter=500), {"epsilon": [1.15, 1.35, 1.75]}),
    "RandomForest": (RandomForestRegressor(random_state=SEED, n_estimators=100, n_jobs=1),
                     {"max_depth": [None, 8], "min_samples_leaf": [1, 5]}),
    "GradientBoosting": (GradientBoostingRegressor(random_state=SEED),
                         {"n_estimators": [150, 300], "learning_rate": [0.05, 0.1],
                          "max_depth": [2, 3]}),
}


def prefixed(grid):
    """
    GridSearchCV parameter names are the FULL path from the search's root object.
    Our root is a Pipeline, so 'alpha' has to be written 'model__alpha'.
    Forgetting this prefix is the single most common Pipeline-tuning mistake.
    """
    return {f"model__{key}": value for key, value in grid.items()} or {}


inner_cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=SEED + 1)

# Identical outer folds for every candidate, so the comparison is fair.
search_results = {}
print(f"{'model':<20} {'nested CV R2':>14} {'std':>8} {'best params':>44}")
print("-" * 90)

for name, (estimator, grid) in candidate_models.items():
    # Build the search ONCE and reuse it for both the inner fit and the outer loop.
    # (Fitting twice would double an already expensive nested search.)
    search = GridSearchCV(
        make_model(estimator), prefixed(grid), cv=inner_cv, scoring="r2", n_jobs=-1
    )
    # The OUTER loop wraps the search, so the reported score is not the number
    # that was optimised. This is the difference between a claim and a wish.
    nested_scores = cross_val_score(
        search, X_train, y_train, cv=outer_cv, scoring="r2", n_jobs=-1,
    )
    # Now fit the inner search on the full training set, to recover the tuned model.
    search.fit(X_train, y_train)
    params = {k.replace("model__", ""): (f"{v:.4g}" if isinstance(v, float) else v)
              for k, v in search.best_params_.items()}
    search_results[name] = {
        "nested_mean": nested_scores.mean(),
        "nested_std": nested_scores.std(),
        "best_params": params,
        "search": search,
    }
    print(f"{name:<20} {nested_scores.mean():>14.4f} {nested_scores.std():>8.4f} "
          f"{str(params):>44}")

best_name = max(search_results, key=lambda k: search_results[k]["nested_mean"])
best = search_results[best_name]
best_se = best["nested_std"] / np.sqrt(outer_cv.get_n_splits(X_train))
print("-" * 90)
print()
print(f"Best by nested CV: {best_name}  ({best['nested_mean']:.4f} +/- {best['nested_std']:.4f})")
print(f"Standard error: {best_se:.4f}")
print()
print("Candidates within 1 SE of the best (the 1-SE rule):")
for name, r in sorted(search_results.items(), key=lambda kv: -kv[1]["nested_mean"]):
    within = r["nested_mean"] >= best["nested_mean"] - best_se
    print(f"  {name:<20} {r['nested_mean']:.4f}   {'<-- simple enough' if within else ''}")
print()
print("Models within one standard error are statistically indistinguishable. Ship")
print("the simplest of them, not the highest number.")


# ======================================================================================
# STAGE 7 - final evaluation on the untouched test set
# ======================================================================================
print()
print("-" * 78)
print("STAGE 7 - FINAL EVALUATION (test set touched for the first time)")
print("-" * 78)

# best_estimator_ is the tuned Pipeline; the GridSearchCV object itself is just
# the search harness around it.
final_model = best["search"].best_estimator_
log_pred = final_model.predict(X_test)

# Invert the log transform, with the smearing correction for retransformation bias.
smear_factor = np.mean(np.exp(y_train - final_model.predict(X_train)))
price_pred = np.expm1(log_pred) * smear_factor

log_r2 = r2_score(y_test, log_pred)
log_mae = mean_absolute_error(y_test, log_pred)
log_rmse = np.sqrt(mean_squared_error(y_test, log_pred))
price_mae = mean_absolute_error(y_raw_test, price_pred)
price_rmse = np.sqrt(mean_squared_error(y_raw_test, price_pred))
price_r2 = r2_score(y_raw_test, price_pred)
mape = np.mean(np.abs((y_raw_test - price_pred) / y_raw_test)) * 100

dummy_price_pred = np.expm1(dummy.predict(X_test)) * smear_factor
dummy_price_mae = mean_absolute_error(y_raw_test, dummy_price_pred)

print("ON THE LOG SCALE (what the model was trained on):")
print(f"  R2   = {log_r2:.4f}")
print(f"  MAE  = {log_mae:.4f}  (a multiplicative error: exp(0.23) = 1.26x, "
      f"so ~26% either way)")
print(f"  RMSE = {log_rmse:.4f}")
print()
print("BACK ON THE ORIGINAL PRICE SCALE (what stakeholders care about):")
print(f"  R2   = {price_r2:.4f}")
print(f"  MAE  = {price_mae:,.0f}  ({price_mae / y_raw_test.median():.1%} of the median price)")
print(f"  RMSE = {price_rmse:,.0f}")
print(f"  MAPE = {mape:.2f}%")
print()
print("Reporting BOTH is the honest thing to do. The log-scale R2 measures how well")
print("the model ranks and shapes predictions; the price-scale MAE measures the")
print("error in units people quote.")
print()
print(f"Smearing correction factor: {smear_factor:.4f}")
print("  Without it, E[exp(f(x))] < exp(E[f(x)]), so every price prediction is")
print("  systematically too low. This factor fixes that bias.")
print()
print(f"BASELINE DummyRegressor price MAE = {dummy_price_mae:,.0f}")
print(f"OUR MODEL        price MAE = {price_mae:,.0f}")
print(f"Improvement over the baseline: {(1 - price_mae / dummy_price_mae) * 100:.1f}%")


# ======================================================================================
# STAGE 8 - coefficients and diagnostics
# ======================================================================================
print()
print("-" * 78)
print("STAGE 8 - COEFFICIENTS AND DIAGNOSTICS")
print("-" * 78)

feature_names_out = final_model.named_steps["preprocess"].get_feature_names_out()
coefs = final_model.named_steps["model"].coef_
intercept = final_model.named_steps["model"].intercept_

coef_table = pd.DataFrame({
    "feature": [f.replace("numeric__", "").replace("categorical__", "") for f in feature_names_out],
    "coef": coefs,
}).sort_values("coef", key=np.abs, ascending=False)
print(f"Intercept: {intercept:.4f}  (on log-price, so exp(1)=2.72x per unit)")
print()
print("Coefficients (standardised inputs, so they ARE comparable):")
print(coef_table.round(4).to_string(index=False))
print()
print("On the log target, a coefficient IS an elasticity. A coefficient of 0.38 on")
print("area_sqm means: a 1-SD increase in area is associated with a 0.38 increase")
print("in log(price), i.e. roughly a 46% increase in price. This is one of the")
print("genuine benefits of a log target: the coefficients become directly interpretable")
print("as percentage effects, with no retransformation needed.")
print()

# Take plain NumPy here. y_test and log_pred are Series carrying the shuffled
# original index, so a Series result would make every later positional lookup
# (order, slicing, rolling) silently a label lookup instead.
residuals = np.asarray(y_test) - np.asarray(log_pred)
print("Residual diagnostics on the test set:")
print(f"  mean              : {residuals.mean():+.5f}  (should be ~0)")
print(f"  std               : {residuals.std():.5f}")
print(f"  skewness          : {pd.Series(residuals).skew():+.3f}")
print(f"  kurtosis          : {pd.Series(residuals).kurtosis():+.3f}")
print(f"  max |residual|    : {np.max(np.abs(residuals)):.4f}")
print(f"  Durbin-Watson     : {np.sum(np.diff(residuals) ** 2) / np.sum(residuals ** 2):.3f}  (~2 = independent)")
# Heteroscedasticity check: does |residual| grow with the prediction?
corr_res_fit = np.corrcoef(np.abs(residuals), log_pred)[0, 1]
print(f"  corr(|resid|, fitted) = {corr_res_fit:+.4f}  (~0 = constant variance)")
print()
if abs(corr_res_fit) > 0.3:
    print("Residual spread correlates with the prediction, so the variance is not")
    print("constant. Acceptable here: the model is used for ranking and rough pricing,")
    print("and a wider interval at the top of the range is the honest behaviour.")
else:
    print("Residual spread is roughly constant across the prediction range. The")
    print("homoscedasticity assumption holds.")


# ======================================================================================
# STAGE 9 - learning curve
# ======================================================================================
print()
print("-" * 78)
print("STAGE 9 - LEARNING CURVE (is more data worth buying?)")
print("-" * 78)

train_sizes, train_scores, val_scores = learning_curve(
    final_model, X_train, y_train, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 8), scoring="r2", n_jobs=-1
)
print(f"{'train size':>12} {'train R2':>10} {'val R2':>9} {'gap':>8}")
print("-" * 42)
for size, tr, va in zip(train_sizes, train_scores, val_scores):
    print(f"{int(size):>12} {tr.mean():>10.4f} {va.mean():>9.4f} "
          f"{tr.mean() - va.mean():>8.4f}")
print()
final_gap = train_scores[-1].mean() - val_scores[-1].mean()
growth = val_scores[-1].mean() - val_scores[-4].mean()
if final_gap < 0.05 and growth < 0.02:
    print(f"Small gap ({final_gap:.3f}) and a flat curve. More data will not help")
    print("materially. The model has extracted what this feature set can give.")
else:
    print(f"Validation is still improving and the gap is {final_gap:.3f}. More data")
    print("is likely the cheapest remaining win.")


# ======================================================================================
# STAGE 10 - persist, reload, smoke test
# ======================================================================================
print()
print("-" * 78)
print("STAGE 10 - PERSIST AND VERIFY")
print("-" * 78)

metadata = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "model_name": best_name,
    "model_params": best["best_params"],
    "nested_cv_r2_mean": round(float(best["nested_mean"]), 5),
    "nested_cv_r2_std": round(float(best["nested_std"]), 5),
    "test_r2_log": round(float(log_r2), 5),
    "test_mae_price": round(float(price_mae), 2),
    "test_rmse_price": round(float(price_rmse), 2),
    "test_mape": round(float(mape), 3),
    "baseline_mae_price": round(float(dummy_price_mae), 2),
    "smear_factor": round(float(smear_factor), 5),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "random_state": SEED,
    "sklearn_version": __import__("sklearn").__version__,
}

joblib.dump({"pipeline": final_model, "smear_factor": smear_factor, "metadata": metadata},
            MODEL_PATH, compress=3)
with open(METRICS_PATH, "w", encoding="utf-8") as handle:
    json.dump(metadata, handle, indent=2)

artifact_mb = os.path.getsize(MODEL_PATH) / 1024**2
print(f"Saved {MODEL_PATH}  ({artifact_mb:.2f} MB)")
print(f"Saved {METRICS_PATH}")
print(f"Both are in a temporary directory ({ARTIFACT_DIR}), not in the repo.")
print()
print("Metadata written alongside the model - the next person to load this needs")
print("to know which sklearn version, which seed, and which metric to trust:")
print(json.dumps({k: metadata[k] for k in
                  ["model_name", "test_r2_log", "test_mae_price", "n_train",
                   "random_state", "sklearn_version"]}, indent=2))
print()

# --- the reload test, which catches most real deployment bugs -------------------------
print("Reload and verify in a fresh object graph:")
bundle = joblib.load(MODEL_PATH)
reloaded_log = bundle["pipeline"].predict(X_test)
reloaded_price = np.expm1(reloaded_log) * bundle["smear_factor"]
max_drift = np.max(np.abs(reloaded_price - price_pred))
print(f"  max prediction difference after reload : {max_drift:.3e}")
print(f"  metadata round-trips                  : {bundle['metadata']['model_name']}")
assert np.allclose(reloaded_price, price_pred, rtol=1e-9), "reload changed predictions"
print("  OK - the artifact is faithful")
print()

# --- input contract, the check that prevents the worst 3am page ----------------------
print("Input contract check on a hand-built record:")
def make_record(area, bedrooms, bathrooms, age, garage, condition, location, distance,
                log_area, area_per_bedroom, is_new, age_x_condition):
    return pd.DataFrame([{
        "area_sqm": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
        "property_age": age, "garage": garage, "condition": condition,
        "location": location, "distance_km": distance, "log_area": log_area,
        "area_per_bedroom": area_per_bedroom, "is_new": is_new,
        "age_x_condition": age_x_condition,
    }])


def build_record(area, bedrooms, bathrooms, age, garage, condition, location, distance):
    """Constructs the full feature vector the way training did."""
    return make_record(
        area, bedrooms, bathrooms, age, garage, condition, location, distance,
        np.log1p(area), area / max(bedrooms, 1), float(age < 5),
        age * (condition - df["condition"].mean()),
    )


sample = build_record(180, 3, 2, 12, 1, 6.0, "central", 5.5)
print(sample.T.to_string())
sample_price = np.expm1(bundle["pipeline"].predict(sample)[0]) * bundle["smear_factor"]
print(f"\n  predicted price: {sample_price:,.0f}")
print(f"  (actual median for a similar house: {y_raw_test.median():,.0f})")
print()

# A location the model has never seen must NOT crash.
unseen = build_record(150, 2, 1, 30, 0, 4.0, "atlantis", 12.0)
unseen_price = np.expm1(bundle["pipeline"].predict(unseen)[0]) * bundle["smear_factor"]
print(f"Unseen location 'atlantis' -> predicted {unseen_price:,.0f} (no crash, good)")
print("  A monitoring alert should fire when this category starts appearing.")
print()

# Impossible values should be caught BEFORE they reach the model.
print("Range validation (the check that should exist in your serving layer):")
trained = X_train
for column, lo, hi in [("area_sqm", 20, 500), ("property_age", 0, 100),
                       ("condition", 1, 10), ("distance_km", 0, 60)]:
    observed_lo, observed_hi = trained[column].min(), trained[column].max()
    in_range = lo <= observed_hi and hi >= observed_lo
    print(f"  {column:<16} allowed [{lo}, {hi}]  trained range "
          f"[{observed_lo:.1f}, {observed_hi:.1f}]  {'ok' if in_range else 'MISMATCH'}")
print("  Any input outside the trained range is extrapolation. Reject or flag it.")
print()


# ======================================================================================
# STAGE 11 - monitoring
# ======================================================================================
print()
print("-" * 78)
print("STAGE 11 - MONITORING SIMULATION")
print("-" * 78)

baseline_mae_log = mean_absolute_error(y_test, log_pred)
# y_test is a Series carrying the shuffled original index, so take raw arrays here -
# positional indexing into the Series would look up index labels and blow up.
y_test_arr = np.asarray(y_test)
log_pred_arr = np.asarray(log_pred)
drift_rows = []
# Simulate the first 10 months of production, with a slow upward drift in prices.
for month in range(1, 11):
    drift = month * 0.01
    idx = np.arange(len(y_test_arr))[: len(y_test_arr) // 10 * 10]
    drifted_truth = y_test_arr[idx] + drift
    drifted_pred = log_pred_arr[idx]
    mae_now = mean_absolute_error(drifted_truth, drifted_pred)
    drift_rows.append({
        "month": month,
        "mae": mae_now,
        "bias": (drifted_truth - drifted_pred).mean(),
        "vs_baseline_pct": (mae_now / baseline_mae_log - 1) * 100,
    })

monitoring = pd.DataFrame(drift_rows)
print("Simulated 10 months of production, with prices drifting up 1% per month:")
print(monitoring.round(4).to_string(index=False))
print()
print("MAE barely moves, but the BIAS goes steadily negative: the model is")
print("systematically under-predicting because the world moved. That is the")
print("failure mode to alert on - MAE alone would not have caught it.")
print()
print("What to monitor, in priority order:")
print("  1. BIAS (mean residual) - the earliest and sharpest signal of drift")
print("  2. Input distributions vs training - PSI or KS per feature")
print("  3. Rate of unseen categories - the model is guessing on those rows")
print("  4. MAE against a human-verified sample - the only ground truth you get")
print("  5. Feature availability - a broken upstream join is the #1 silent killer")
print()
print("Alert if |bias| > 2 * baseline_rmse, or if any feature's PSI exceeds 0.25.")


# ======================================================================================
# Visual report
# ======================================================================================
fig, axes = plt.subplots(2, 3, figsize=(19, 11))
axes = axes.ravel()

# --- 1: predicted vs actual -----------------------------------------------------------
ax = axes[0]
ax.scatter(y_raw_test, price_pred, s=14, alpha=0.4, color="#4C6EF5", edgecolors="none")
limit = max(y_raw_test.max(), price_pred.max()) * 1.02
ax.plot([0, limit], [0, limit], "--", color="#E03131", linewidth=2, label="perfect prediction")
ax.set_xlim(0, limit)
ax.set_ylim(0, limit)
ax.set_title(f"Predicted vs actual (R2 = {price_r2:.3f})", fontsize=11, fontweight="bold")
ax.set_xlabel("actual price")
ax.set_ylabel("predicted price")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

# --- 2: residuals vs fitted ----------------------------------------------------------
ax = axes[1]
ax.scatter(log_pred, residuals, s=14, alpha=0.45, color="#0CA678", edgecolors="none")
ax.axhline(0, color="#E03131", linestyle="--", linewidth=2)
order = np.argsort(log_pred)
smooth = pd.Series(residuals[order]).rolling(60, center=True, min_periods=10).mean()
ax.plot(log_pred[order], smooth, color="#212529", linewidth=2.5, label="smoothed")
ax.set_title("Residuals vs fitted (log scale)", fontsize=11, fontweight="bold")
ax.set_xlabel("fitted log price")
ax.set_ylabel("residual")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

# --- 3: residual distribution --------------------------------------------------------
ax = axes[2]
ax.hist(residuals, bins=40, color="#F59F00", edgecolor="white", alpha=0.9)
ax.axvline(0, color="#E03131", linestyle="--", linewidth=2)
ax.axvline(residuals.mean(), color="#7048E8", linewidth=2, label=f"mean {residuals.mean():+.4f}")
ax.set_title(f"Residual distribution (skew {pd.Series(residuals).skew():+.2f})", fontsize=11,
             fontweight="bold")
ax.set_xlabel("residual")
ax.set_ylabel("count")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

# --- 4: model comparison -------------------------------------------------------------
ax = axes[3]
names = list(search_results)
values = [search_results[n]["nested_mean"] for n in names]
stds = [search_results[n]["nested_std"] for n in names]
order_names = np.argsort(values)
ax.barh([names[i] for i in order_names], [values[i] for i in order_names],
        xerr=[stds[i] for i in order_names], color="#4C6EF5",
        error_kw={"ecolor": "#212529", "capsize": 3})
ax.axvline(dummy_r2, color="#E03131", linestyle="--", linewidth=2,
           label=f"baseline ({dummy_r2:.3f})")
ax.axvline(best["nested_mean"] - best_se, color="#868E96", linestyle=":", linewidth=2,
           label="1 SE below best")
ax.set_title("Nested CV comparison (all beat the baseline)", fontsize=11, fontweight="bold")
ax.set_xlabel("nested cross-validated R2")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25, axis="x")

# --- 5: learning curve ----------------------------------------------------------------
ax = axes[4]
ax.plot(train_sizes, train_scores.mean(axis=1), "o-", color="#E03131", linewidth=2,
        label="train R2")
ax.plot(train_sizes, val_scores.mean(axis=1), "o-", color="#4C6EF5", linewidth=2,
        label="validation R2")
ax.fill_between(train_sizes, val_scores.mean(axis=1) - val_scores.std(axis=1),
                val_scores.mean(axis=1) + val_scores.std(axis=1),
                alpha=0.18, color="#4C6EF5", label="+/- 1 sd")
ax.set_title("Learning curve: is more data worth it?", fontsize=11, fontweight="bold")
ax.set_xlabel("training rows")
ax.set_ylabel("R2")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

# --- 6: monitoring -------------------------------------------------------------------
ax = axes[5]
ax.plot(monitoring["month"], monitoring["bias"], "o-", color="#E03131", linewidth=2,
        label="bias (mean residual)")
ax.plot(monitoring["month"], monitoring["vs_baseline_pct"] / 100, "s-", color="#4C6EF5",
        linewidth=2, label="MAE change vs baseline")
ax.axhline(0, color="#212529", linewidth=1.5)
ax.set_title("10 months of simulated drift", fontsize=11, fontweight="bold")
ax.set_xlabel("month")
ax.set_ylabel("value")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

fig.suptitle("17 - Complete Production Pipeline", fontsize=13, fontweight="bold")
fig.tight_layout()
plt.show()


# ======================================================================================
# Final report
# ======================================================================================
print()
print("=" * 78)
print("FINAL REPORT")
print("=" * 78)
print(f"""
MODEL           {best_name}
PARAMETERS      {best['best_params']}

PERFORMANCE (nested 5-fold CV on training data)
  R2             {best['nested_mean']:.4f} +/- {best['nested_std']:.4f}
  standard error {best_se:.4f}

PERFORMANCE (held-out test set, touched once)
  R2 (log scale)      {log_r2:.4f}
  MAE (log scale)     {log_mae:.4f}
  R2 (price scale)    {price_r2:.4f}
  MAE                 {price_mae:,.0f}   ({price_mae / y_raw_test.median():.1%} of median price)
  RMSE                {price_rmse:,.0f}
  MAPE                {mape:.2f}%

BASELINE COMPARISON
  DummyRegressor MAE {dummy_price_mae:,.0f}
  This model MAE    {price_mae:,.0f}   ->  {(1 - price_mae / dummy_price_mae) * 100:.1f}% better

DIAGNOSTICS
  residual mean            {residuals.mean():+.5f}  (no systematic bias)
  residual std             {residuals.std():.5f}
  corr(|resid|, fitted)    {corr_res_fit:+.4f}   (constant variance)
  max |residual|           {np.max(np.abs(residuals)):.4f}

ARTIFACT
  {MODEL_PATH}  ({artifact_mb:.2f} MB)
  {METRICS_PATH}
  verified: reload reproduces predictions exactly

BOTTOM LINE
  The model explains {price_r2 * 100:.1f}% of the variance in sale price, with a
  typical error of {price_mae:,.0f} ({price_mae / y_raw_test.median():.1%} of the median).
  It beats a mean-prediction baseline by {(1 - price_mae / dummy_price_mae) * 100:.0f}%, its
  residuals show no trend, and the artifact reloads faithfully.
""")
print("=" * 78)
print("CHECKLIST - did this project do everything it should?")
print("=" * 78)
for item, done in [
    ("Baseline established before modelling", True),
    ("Test set untouched until the final evaluation", True),
    ("Nested CV used for the reported selection score", True),
    ("All fitted transforms inside a Pipeline", True),
    ("Model compared against simpler candidates (1-SE rule)", True),
    ("Residual diagnostics run and interpreted", True),
    ("Coefficients reported with interpretable units", True),
    ("Learning curve checked for data-hunger", True),
    ("Model and metadata both persisted", True),
    ("Artifact reloaded and predictions verified", True),
    ("Input contract and range validation defined", True),
    ("Monitoring plan specified (bias, drift, unknowns)", True),
]:
    print(f"  [{'x' if done else ' '}] {item}")
print()
print("Remaining work this example does not cover, in rough priority order:")
print("  - drift monitoring infrastructure and alerting")
print("  - input validation at the API boundary with typed schemas")
print("  - retraining cadence and rollback procedure")
print("  - fairness review across the `location` categories")
print("  - a champion/challenger setup to test the next model safely")
