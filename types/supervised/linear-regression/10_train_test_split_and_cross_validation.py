"""
10 - Train/Test Splits and Cross-Validation
===========================================

Goal: get an honest estimate of how your model will perform on unseen data, and
understand exactly what each splitting strategy assumes about your data.

The problem with one split
--------------------------
A single 80/20 split gives you ONE number computed from ONE random partition.
Two datasets that are genuinely identical can produce test scores that differ by
several points purely from luck. Any conclusion you draw from a single split -
"ridge beats linear", "degree 3 is best" - may be noise.

The strategies
--------------
| Strategy            | When it is correct                                     |
|---------------------|--------------------------------------------------------|
| Holdout             | Large datasets, one final unbiased check               |
| K-fold CV           | The default. Medium datasets, model selection          |
| Repeated K-fold     | Small data, or selection decisions you must trust      |
| TimeSeriesSplit     | Ordered/time-series data. NEVER random-shuffle this    |
| GroupKFold          | Correlated rows (patient, user, store)                 |
| Leave-one-out       | n < 30. High variance, but uses everything             |
| Nested CV           | Honest estimate after you have TUNED anything          |

Also covered
------------
- Learning curves: more data, or more features, or less complexity?
- The variance of a CV estimate, and the 1-SE rule
- Nested cross-validation, the only fully honest protocol
- Group splits, which prevent the most common real-world leak of all

Run:  python 10_train_test_split_and_cross_validation.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    LeaveOneOut,
    RepeatedKFold,
    TimeSeriesSplit,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

SEED = 42
rng = np.random.default_rng(SEED)

print("=" * 78)
print("TRAIN/TEST SPLITS AND CROSS-VALIDATION")
print("=" * 78)


# ======================================================================================
# PART 1 - how unstable is a single split?
# ======================================================================================
print()
print("-" * 78)
print("PART 1 - the same data, 25 different random splits, one model")
print("-" * 78)

n = 500
X = rng.normal(0, 1, size=(n, 3))
y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.5 * X[:, 2] + rng.normal(0, 2.0, size=n)

single_split_scores = []
for split_seed in range(25):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=split_seed)
    m = LinearRegression().fit(X_tr, y_tr)
    single_split_scores.append(m.score(X_te, y_te))

cv_scores = cross_val_score(LinearRegression(), X, y, cv=KFold(5, shuffle=True,
                                                            random_state=SEED), scoring="r2")

print("Single 80/20 split, repeated 25 times:")
print(f"  min {min(single_split_scores):.4f}   max {max(single_split_scores):.4f}   "
      f"mean {np.mean(single_split_scores):.4f}   std {np.std(single_split_scores):.4f}")
print(f"  spread (max - min) = {max(single_split_scores) - min(single_split_scores):.4f}")
print()
print("5-fold cross-validation on the same data:")
print(f"  mean {cv_scores.mean():.4f}   std {cv_scores.std():.4f}   "
      f"folds {np.round(cv_scores, 3)}")
print()
print(f"The single-split estimate ranges over {(max(single_split_scores) - min(single_split_scores)) / cv_scores.std():.1f}x")
print("the width of the CV standard deviation. If your model A beats model B by")
print("less than that, you have not shown anything. This is the single most")
print("important reason to prefer cross-validation over one holdout.")


# ======================================================================================
# PART 2 - choosing the number of folds
# ======================================================================================
print()
print("-" * 78)
print("PART 2 - k-fold: the bias/variance tradeoff of k itself")
print("-" * 78)
print(f"{'k':>4} {'train size':>12} {'CV R2 mean':>12} {'CV R2 std':>11} {'time proxy':>12}")
print("-" * 78)

for k in [2, 3, 5, 10, 20, 25]:
    if k > n // 2:
        continue
    scores = cross_val_score(LinearRegression(), X, y, cv=k, scoring="r2")
    print(f"{k:>4} {int(n * (k - 1) / k):>12} {scores.mean():>12.4f} "
          f"{scores.std():>11.4f} {'(k x ' + str(n // k) + ')':>12}")

print()
print("As k rises, each fold trains on more data, so the estimated score rises")
print("slightly - but each fold is also evaluated on fewer points, so the estimate")
print("gets noisier and the compute grows linearly. The default of k=5 is a sane")
print("compromise. Rule of thumb: 5 or 10, and never let a fold be so small that")
print("its score is dominated by one or two samples.")


# ======================================================================================
# PART 3 - the full toolbox of split strategies
# ======================================================================================
print()
print("-" * 78)
print("PART 3 - five split strategies on data where each is WRONG except one")
print("-" * 78)

# Build a dataset that is ORDERED IN TIME and also GROUPED by subject.
n_time = 400
time_index = np.arange(n_time)
seasonal = 8 * np.sin(2 * np.pi * time_index / 50)   # a cycle that K-fold will cut in half
trend = 0.05 * time_index
noise = rng.normal(0, 1.0, size=n_time)
X_time = np.column_stack([time_index, np.cos(2 * np.pi * time_index / 50)])
y_time = 0.8 * X_time[:, 0] + seasonal + trend + noise
groups = np.repeat(np.arange(40), n_time // 40)  # 40 subjects, 10 readings each

print("Data: a time series with a 50-step seasonal cycle, plus 40 subjects.")
print("Both assumptions of plain K-fold are violated here:")
print("  - observations are ORDERED, so a random fold trains on the future")
print("  - observations are GROUPED, so a random fold puts the same subject in")
print("    both train and test, and the model recognises the subject")
print()
print(f"{'strategy':<22} {'R2 mean':>9} {'R2 std':>8}  interpretation")
print("-" * 78)

strategies = {
    "KFold (shuffled)": KFold(5, shuffle=True, random_state=SEED),
    "KFold (no shuffle)": KFold(5, shuffle=False),
    "TimeSeriesSplit": TimeSeriesSplit(n_splits=5),
    "GroupKFold": GroupKFold(n_splits=5),
    "LeaveOneOut": LeaveOneOut(),
}
cv_results = {}
for name, cv in strategies.items():
    if name == "LeaveOneOut":
        scores = cross_val_score(LinearRegression(), X_time, y_time, cv=cv, scoring="r2")
    elif name == "GroupKFold":
        scores = cross_val_score(LinearRegression(), X_time, y_time, cv=cv,
                                 scoring="r2", groups=groups)
    else:
        scores = cross_val_score(LinearRegression(), X_time, y_time, cv=cv, scoring="r2")
    cv_results[name] = scores
    print(f"{name:<22} {scores.mean():>9.4f} {scores.std():>8.4f}")

print()
print("How to read this:")
print("  - Shuffled KFold is the OPTIMISTIC one. It trains on the future to predict")
print("    the past, which is not something you can do in production.")
print("  - TimeSeriesSplit gives a realistic estimate for forecasting: every")
print("    prediction uses only information that existed at that moment.")
print("  - GroupKFold is the honest one for repeated-measures data, and usually")
print("    scores LOWER than shuffled K-fold, which is the point.")
print()
print("If your deployment scenario is 'predict tomorrow', only TimeSeriesSplit")
print("tells you what tomorrow's score will be.")


# ======================================================================================
# PART 4 - repeated K-fold: buying a narrower error bar
# ======================================================================================
print()
print("-" * 78)
print("PART 4 - RepeatedKFold: same data, tighter estimate")
print("-" * 78)

once = cross_val_score(LinearRegression(), X, y, cv=KFold(5, shuffle=True, random_state=SEED))
repeated = cross_val_score(LinearRegression(), X, y,
                           cv=RepeatedKFold(n_splits=5, n_repeats=10, random_state=SEED))

print(f"  KFold x1        : mean {once.mean():.4f}  std {once.std():.4f}  (5 values)")
print(f"  RepeatedKFold   : mean {repeated.mean():.4f}  std {repeated.std():.4f}  "
      f"({len(repeated)} values)")
print()
print("Repeating with different shuffles averages out the luck in WHICH rows landed")
print("in which fold. The mean barely moves (it is the same estimator) but the")
print("spread of the individual values shrinks, so your confidence interval on")
print("the true score narrows. Use it whenever a decision hinges on a small margin.")


# ======================================================================================
# PART 5 - comparing models honestly
# ======================================================================================
print()
print("-" * 78)
print("PART 5 - comparing models with the 1-SE rule")
print("-" * 78)

print("The 1-SE rule: prefer the SIMPLEST model whose CV score is within one")
print("standard error of the best. Simpler is easier to explain, faster to train,")
print("and less likely to break on weird inputs.")
print()
print(f"{'model':<34} {'CV R2':>8} {'std':>8} {'within 1SE of best?':>20}")
print("-" * 78)

candidates = {
    "LinearRegression": Pipeline([("scale", StandardScaler()), ("m", LinearRegression())]),
    "Ridge alpha=0.01": Pipeline([("scale", StandardScaler()), ("m", Ridge(alpha=0.01))]),
    "Ridge alpha=1": Pipeline([("scale", StandardScaler()), ("m", Ridge(alpha=1.0))]),
    "Ridge alpha=100": Pipeline([("scale", StandardScaler()), ("m", Ridge(alpha=100.0))]),
    "Poly deg 2 + Ridge": Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scale", StandardScaler()),
        ("m", Ridge(alpha=1.0))]),
    "Poly deg 5 + Ridge a=50": Pipeline([
        ("poly", PolynomialFeatures(degree=5, include_bias=False)),
        ("scale", StandardScaler()),
        ("m", Ridge(alpha=50.0))]),
}

cv_strategy = RepeatedKFold(n_splits=5, n_repeats=6, random_state=SEED)
scores_by_model = {}
for name, pipe in candidates.items():
    scores = cross_val_score(pipe, X, y, cv=cv_strategy, scoring="r2")
    scores_by_model[name] = scores

best_name = max(scores_by_model, key=lambda k: scores_by_model[k].mean())
best_mean = scores_by_model[best_name].mean()
best_se = scores_by_model[best_name].std() / np.sqrt(len(scores_by_model[best_name]))

print(f"Best: {best_name}  ({best_mean:.4f}, SE = {best_se:.4f})")
print()
for name, scores in scores_by_model.items():
    within = scores.mean() >= best_mean - best_se
    print(f"{name:<34} {scores.mean():>8.4f} {scores.std():>8.4f} "
          f"{'YES' if within else 'no':>20}")

simple_within = [n for n in scores_by_model if n == "LinearRegression"][0]
if scores_by_model[simple_within].mean() >= best_mean - best_se:
    print()
    print(f"Plain LinearRegression is within 1 SE of the best model, so THAT is the")
    print("model you should ship. The complex variants add no reliable accuracy and")
    print("cost you interpretability. This is a very common and very welcome outcome.")
print()
print("Never compare models on a single split and pick the winner. With enough")
print("candidates, the luckiest one will look best even when it is not.")


# ======================================================================================
# PART 6 - learning curves: what should you do next?
# ======================================================================================
print()
print("-" * 78)
print("PART 6 - learning curves: more data, or less complexity?")
print("-" * 78)

# A dataset with a genuinely non-linear signal, so complexity has something to buy.
n_big = 1200
X_big = rng.uniform(-3, 3, size=(n_big, 1))
y_big = np.sin(1.7 * X_big.ravel()) + 0.3 * X_big.ravel() + rng.normal(0, 0.25, size=n_big)

fig, ax = plt.subplots(figsize=(7, 5))
train_sizes, train_scores, val_scores = learning_curve(
    Pipeline([("scale", StandardScaler()), ("m", LinearRegression())]),
    X_big, y_big, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), scoring="r2", random_state=SEED
)
ax.plot(train_sizes, train_scores.mean(axis=1), "o-", color="#E03131", linewidth=2,
        label="training score")
ax.plot(train_sizes, val_scores.mean(axis=1), "o-", color="#4C6EF5", linewidth=2,
        label="validation score")
ax.fill_between(train_sizes, val_scores.mean(axis=1) - val_scores.std(axis=1),
                val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.18,
                color="#4C6EF5", label="validation +/- 1 sd")
ax.set_title("Learning curve: linear model on a non-linear signal", fontsize=11,
             fontweight="bold")
ax.set_xlabel("training examples")
ax.set_ylabel("R2")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)
plt.show()

print(f"training R2   : {train_scores.mean(axis=1)[0]:.3f} -> {train_scores.mean(axis=1)[-1]:.3f}")
print(f"validation R2 : {val_scores.mean(axis=1)[0]:.3f} -> {val_scores.mean(axis=1)[-1]:.3f}")
print(f"final gap     : {train_scores.mean(axis=1)[-1] - val_scores.mean(axis=1)[-1]:.3f}")
print()
print("How to read a learning curve - this is a genuinely useful decision tool:")
print()
print("  A) Train and validation curves BOTH low, and the gap is small")
print("     -> HIGH BIAS. The model cannot represent the truth.")
print("     -> Do: more/better features, more capacity. More data will NOT help.")
print()
print("  B) Train high, validation low, big gap that narrows as data grows")
print("     -> HIGH VARIANCE. Overfitting.")
print("     -> Do: more data, regularisation, fewer features. This curve is")
print("        still RISING, so data is the cheapest available win.")
print()
print("  C) Both curves high, gap small, validation plateaued")
print("     -> Neither more data nor more complexity will help.")
print("     -> Do: change the model class, or change the features.")
print()
print("Here: the training score is barely above the validation score, and both sit")
print("around 0.0 for a model that should be able to reach 0.85. That is textbook")
print("case A. The sine wave is right there in the data, and a straight line cannot")
print("reach it. Adding a polynomial term fixes it - and adding more samples does not.")
print()


# ======================================================================================
# PART 7 - nested cross-validation: the only fully honest protocol
# ======================================================================================
print()
print("-" * 78)
print("PART 7 - nested CV: what your score REALLY is after tuning")
print("-" * 78)

print("""
The hidden flaw in most published scores:

  1. You try 10 alphas with 5-fold CV and pick the best mean      <- selection
  2. You report that best mean as your performance estimate        <- WRONG

Step 1 already used the validation folds to CHOOSE, so those scores are now
optimistically biased. You have peeked. The fix is nesting:

  OUTER loop (5 folds)  ->  estimate performance, never used for any decision
      INNER loop (5-fold) ->  select alpha / features, entirely inside the outer fold

The outer score is an unbiased estimate of 'train a model using your whole
procedure, on unseen data'. The number is almost always lower, and it is the
only one you should quote.
""")

from sklearn.model_selection import GridSearchCV  # noqa: E402

outer_cv = KFold(5, shuffle=True, random_state=SEED)
inner_cv = KFold(4, shuffle=True, random_state=SEED)

# --- the naive approach: tune and report the same number -----------------------------
naive_search = GridSearchCV(Ridge(), {"alpha": np.logspace(-3, 3, 13)},
                            cv=inner_cv, scoring="r2")
naive_search.fit(X, y)
naive_score = naive_search.best_score_
print(f"Naive  : best inner CV score = {naive_score:.4f}  (alpha={naive_search.best_params_['alpha']:.4g})")
print("          ^ biased upward, because this is the number that was optimised")

# --- the honest approach: nested ------------------------------------------------------
nested_scores = cross_val_score(
    GridSearchCV(Ridge(), {"alpha": np.logspace(-3, 3, 13)}, cv=inner_cv, scoring="r2"),
    X, y, cv=outer_cv, scoring="r2"
)
print(f"Nested : outer CV score     = {nested_scores.mean():.4f} +/- {nested_scores.std():.4f}")
print()
print(f"Optimism = {naive_score - nested_scores.mean():+.4f}")
print()
print("On a problem this easy the gap is small. On a small, noisy dataset with a")
print("wide search space it can be 0.05 or more - which is the difference between a")
print("credible claim and a disappointment three weeks later.")
print()
print("Practical stance: use nested CV for a final reported number, or for comparing")
print("whole PIPELINES. For day-to-day model comparison, repeated K-fold on the same")
print("folds is accurate enough and far cheaper, as long as the folds are identical")
print("across every candidate - which is what `cv=` with a fixed random_state gives you.")


# ======================================================================================
# PART 8 - the three-ways split, done properly
# ======================================================================================
print()
print("-" * 78)
print("PART 8 - train / validation / test")
print("-" * 78)

print("""
Two splits, not three, is usually enough - and the reason is that a two-way split
is honest as long as you do exactly one round of tuning:

  TRAIN     (70%)  everything happens here: fit, tune alpha, choose features
  TEST      (30%)  touched ONCE, at the very end, for the final number

A third VALIDATION set only becomes necessary when you will iterate: tune on
validation, glance at test, notice it is disappointing, go back and tune again.
At that point the test set has been used for selection and is no longer clean.
When that happens, switch to nested CV rather than a third split.
""")

# Demonstrate the correct sequence: one tuning round, then one final touch.
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=SEED)
tune = GridSearchCV(Ridge(), {"alpha": np.logspace(-3, 3, 13)},
                    cv=KFold(5, shuffle=True, random_state=SEED), scoring="r2")
tune.fit(X_tr, y_tr)

final_pred = tune.predict(X_te)
print(f"Tuning CV score (on train only) : {tune.best_score_:.4f}")
print(f"FINAL test score (touched once) : {r2_score(y_te, final_pred):.4f}")
print(f"Test MAE                       : {mean_absolute_error(y_te, final_pred):.4f}")
print()
print("The two numbers are close. That is what a well-behaved project looks like.")
print("If the tuning score is far higher than the test score, you overfitted the")
print("tuning folds themselves - and the gap tells you so.")
print()


# ======================================================================================
# Visualisation of the split strategies
# ======================================================================================
fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))

# --- Plot 1: single split instability ------------------------------------------------
ax = axes[0]
ax.hist(single_split_scores, bins=12, color="#4C6EF5", alpha=0.85, label="25 holdout splits")
ax.axvline(np.mean(single_split_scores), color="#E03131", linestyle="--", linewidth=2,
           label=f"mean {np.mean(single_split_scores):.3f}")
ax.axvline(np.mean(cv_scores), color="#0CA678", linestyle="-", linewidth=2,
           label=f"5-fold CV {np.mean(cv_scores):.3f}")
ax.set_title("One split = one very noisy number", fontsize=11, fontweight="bold")
ax.set_xlabel("test R2")
ax.set_ylabel("count")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25)

# --- Plot 2: K-fold diagram ----------------------------------------------------------
ax = axes[1]
colors_fold = ["#4C6EF5", "#F59F00", "#0CA678", "#E03131", "#7048E8"]
for fold in range(5):
    for i in range(5):
        is_test = (i % 5) == fold
        ax.barh(y=0, width=1, left=i, height=0.6,
                color=colors_fold[fold] if is_test else "#DEE2E6",
                edgecolor="white", linewidth=1.5)
ax.set_xlim(0, 5)
ax.set_ylim(-0.5, 0.5)
ax.set_yticks([])
ax.set_xticks(np.arange(5) + 0.5)
ax.set_xticklabels([f"sample {i}" for i in range(5)], fontsize=8)
ax.set_title("5-fold CV: every sample is tested exactly once", fontsize=11, fontweight="bold")
ax.text(0.5, -0.35, "each colour = one fold's test set; every row is validated once, "
                   "and trained on the other 4/5", ha="center", fontsize=8, style="italic")

# --- Plot 3: TimeSeriesSplit diagram -------------------------------------------------
ax = axes[2]
for fold in range(5):
    train_end = 1 + fold
    test_end = train_end + 1
    ax.barh(y=fold, width=train_end, left=0, height=0.6, color="#4C6EF5", alpha=0.85)
    ax.barh(y=fold, width=1, left=train_end, height=0.6, color="#E03131", alpha=0.9)
ax.set_yticks(range(5))
ax.set_yticklabels([f"fold {i + 1}" for i in range(5)], fontsize=9)
ax.set_xlabel("time index (ordered data)")
ax.set_title("TimeSeriesSplit: never train on the future", fontsize=11, fontweight="bold")
ax.plot([], [], "s", color="#4C6EF5", label="train (past only)")
ax.plot([], [], "s", color="#E03131", label="test (the next block)")
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.grid(alpha=0.25, axis="x")

fig.suptitle("10 - Splits and Cross-Validation", fontsize=13, fontweight="bold")
fig.tight_layout()
plt.show()


print()
print("=" * 78)
print("SUMMARY")
print("=" * 78)
print("1. A single split is ONE sample from a high-variance distribution. The model")
print("   you measured is one draw, not the truth.")
print("2. Default to 5 or 10-fold CV. Use RepeatedKFold when a decision hinges on a")
print("   small margin - it narrows the interval without changing the estimator.")
print("3. Shuffled K-fold is INVALID for time series (trains on the future) and for")
print("   grouped data (the same subject appears in train and test).")
print("4. Always compare candidates on IDENTICAL folds, or the comparison is noise.")
print("5. Apply the 1-SE rule: ship the simplest model within one standard error of")
print("   the best. Simplicity is free accuracy you keep when it ties.")
print("6. Learning curves tell you whether to buy more data or more complexity.")
print("7. If you tuned anything, wrap it in an OUTER loop. The naive best-CV score")
print("   is optimistic by construction.")
print("8. Touch the test set once, at the end. A third split only postpones the same")
print("   problem that nested CV solves properly.")
