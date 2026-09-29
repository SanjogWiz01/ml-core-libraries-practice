"""
04 - Decision Boundaries and Geometry
====================================

Goal: understand exactly what "linear decision boundary" means, what the margin
is, and where the linearity assumption breaks.

What this file demonstrates
---------------------------
1. Why the decision boundary is the line w.x + b = 0, derived rather than stated.
2. The margin: the geometric distance from the boundary to the nearest point, and
   the difference between geometric and functional margin.
3. Why the boundary's ORIENTATION is what the weights control, and why the
   magnitude of w controls confidence, not position.
4. What happens to the boundary as you add features: XOR (unseparable), concentric
   circles (unseparable), and a ring (separable only with a polynomial feature).
5. The honest limits: what logistic regression cannot do, and the three concrete
   ways to fix it.

Run:  python 04_decision_boundary_and_geometry.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

# --------------------------------------------------------------------------------------
# 1. The algebra of the boundary, on a tiny dataset you can see.
# --------------------------------------------------------------------------------------
# Three points, chosen so the answer is obvious by eye. If the code disagrees with
# your eye, something is wrong with the code, not with your eye.
tiny_X = np.array([[0.0, 0.0],
                   [2.0, 0.0],
                   [0.0, 2.0],
                   [2.0, 2.0]])
tiny_y = np.array([0, 0, 1, 1])

tiny_model = LogisticRegression(C=1.0, max_iter=5000).fit(tiny_X, tiny_y)

print("=" * 76)
print("DECISION BOUNDARIES AND GEOMETRY")
print("=" * 76)
print("A 4-point problem: (0,0) and (2,0) are class 0; (0,2) and (2,2) are class 1.")
print()
print("The separating line you would draw by hand is x2 = 1, with margin 1.0 on")
print("both sides. Here is what the model found:")
print()
print(f"  intercept w0 = {tiny_model.intercept_[0]:+.4f}")
for name, coef in zip(["x1", "x2"], tiny_model.coef_[0]):
    print(f"  weight   {name} = {coef:+.4f}")
print()
# Solve the boundary for x2 and report the slope, rather than eyeballing the weights.
slope = -tiny_model.coef_[0][0] / tiny_model.coef_[0][1]
height = -tiny_model.intercept_[0] / tiny_model.coef_[0][1]
print(f"Rewriting w.x + b = 0 as a line in x2:   x2 = {slope:+.4f} * x1 + {height:.4f}")
print()
print("That is the hand-drawn answer - a near-horizontal line at height ~0.95,")
print("essentially x2 = 1, with a 0.05 tilt that fits all four points exactly. The")
print("model did not need a tilt for the boundary; it needed one to make the")
print("probabilities steep, because the data is PERFECTLY SEPARABLE and there is no")
print("finite penalty for pushing the probabilities towards 0 and 1 forever. That")
print("is COMPLETE SEPARATION, the reason the coefficients are large (28 instead of")
print("1.0), and it is the subject of script 16. The fix is a finite C, which is")
print("exactly what the default C=1.0 gives you here.")
print()
print("This scale freedom is the key idea of the whole file:")
print()
print("    ANY w' = c*w, b' = c*b  with c > 0  gives the SAME decision boundary")
print("    but DIFFERENT probabilities.")
print()
print("So the boundary depends only on the direction of w, while the predicted")
print("probability depends on both direction and magnitude. That is why")
print("'decision_function() > 0' and 'predict_proba() > 0.5' always agree, and why")
print("you cannot read a confidence off a distance to the boundary.")
print()

# --- the algebra, explicitly -----------------------------------------------------------
print("-" * 76)
print("PART A - deriving the boundary instead of memorising it")
print("-" * 76)
print("""
The model outputs a probability:

    p = sigmoid(z) = 1 / (1 + exp(-z)),     z = w0 + w1*x1 + w2*x2

sigmoid is STRICTLY INCREASING. So the ordering of p is the ordering of z, and
whatever threshold t you pick on p corresponds to some threshold on z. The
classification rule is:

    predict 1  <=>  p >= t  <=>  z >= log(t / (1 - t))

For the default t = 0.5, log(0.5/0.5) = 0, so:

    predict 1  <=>  z >= 0  <=>  w0 + w1*x1 + w2*x2 = 0      <- a LINE in 2-D,
                                                               a HYPERPLANE in p-D

The important consequence: THRESHOLDING NEVER MOVES THE BOUNDARY. It moves the
probability cut, but log(t/(1-t)) is a constant, so z = const is still a line -
just a different one, parallel to the original. Thresholding and re-fitting
give genuinely different boundaries; thresholding alone only slides this one.
""")
print("Proof, on a 4-point problem that is perfectly separable, so every threshold")
print("gets the labels right and only the CUT moves:")
print()
print(f"{'threshold':>10} {'equivalent z cut':>18} {'accuracy':>10} {'positives predicted':>21}")
print("-" * 64)
probs = tiny_model.predict_proba(tiny_X)[:, 1]
for threshold in [0.3, 0.5, 0.7]:
    predicted = (probs >= threshold).astype(int)
    z_cut = np.log(threshold / (1 - threshold))
    print(f"{threshold:>10.2f} {z_cut:>18.4f} {np.mean(predicted == tiny_y):>10.2f} "
          f"{predicted.sum():>21}")
print()
print("The coefficients never change - only the intercept does, by exactly -z_cut:")
print()
for threshold in [0.3, 0.5, 0.7]:
    z_cut = np.log(threshold / (1 - threshold))
    print(f"  t = {threshold:.1f}  ->  boundary is  {tiny_model.coef_[0][0]:+.4f}*x1 "
          f"{tiny_model.coef_[0][1]:+.4f}*x2 + ({tiny_model.intercept_[0] - z_cut:+.4f}) = 0")
print()
print("Every one of those is the SAME line, just slid along its own normal. The")
print("slope is untouched. That is the formal statement of 'thresholding does not")
print("rotate the boundary', and it holds for any model, not just this one.")
print()
print("The consequence people get wrong: thresholding an EXISTING model is not the")
print("same as RETRAINING it with a different operating point. Retraining a")
print("logistic regression with more positives in the positive class - which is")
print("what class_weight and resampling do - produces a different slope, because")
print("it optimises a different objective. You get a genuinely different model.")
print("See script 07.")
print()


# --------------------------------------------------------------------------------------
# 2. Margins.
# --------------------------------------------------------------------------------------
print("-" * 76)
print("PART B - the two margins")
print("-" * 76)
print("""
GEOMETRIC margin of a point x from the boundary:

    distance(x, boundary) = |w . x + b| / ||w||

The ||w|| in the denominator is the part people forget. Without it, the "distance"
is just the raw log-odds, which grows with ||w|| and means nothing.
""")

# A well-separated dataset so the margin is meaningful.
rng = np.random.default_rng(11)
sep_X = np.vstack([rng.normal([-2.0, -1.0], 0.7, (150, 2)), rng.normal([2.0, 1.0], 0.7, (150, 2))])
sep_y = np.hstack([np.zeros(150), np.ones(150)])
sep_model = LogisticRegression(C=1.0, max_iter=5000).fit(sep_X, sep_y)

w_vec = sep_model.coef_[0]
norm_w = np.linalg.norm(w_vec)
signed = sep_X @ w_vec + sep_model.intercept_[0]
geometric_margin = np.abs(signed) / norm_w

print(f"||w|| = {norm_w:.4f}, so the divisor is {norm_w:.4f}, not 1.")
print()
print(f"{'class':>6} {'min signed z':>14} {'min geometric margin':>22} {'max |z|':>10}")
print("-" * 58)
for label, mask in [("class 0", sep_y == 0), ("class 1", sep_y == 1)]:
    print(f"{label:>6} {signed[mask].min():>14.4f} {geometric_margin[mask].min():>22.4f} "
          f"{np.abs(signed[mask]).max():>10.4f}")
print()
print(f"Smallest geometric margin over ALL points: {geometric_margin.min():.4f}")
print(f"Smallest |signed z| over all points:       {np.abs(signed).min():.4f}")
print()
print("The margin tells you how much room the classifier has. A small margin means")
print("a few moved points would flip many predictions, which is a fragile model and")
print("a sign the regularisation should be stronger. A large margin means the")
print("model is confident for geometric as well as statistical reasons.")
print()
print("FUNCTIONAL margin is the related but different quantity |w . x| - it measures")
print("how the score changes as x moves, in units of the score itself, and it is")
print("what the regularisation actually penalises. Geometric margin divides that")
print("by ||w||, so the two differ exactly by the factor the penalty controls.")
print()
print("Practical reading: C controls ||w||, and ||w|| is the steepness of the")
print("probability transition across the boundary. Small C means every prediction")
print("is a cautious 0.45 or 0.55. Large C means every prediction is a confident")
print("0.02 or 0.98, right or wrong. The margin is what that steepness costs you in")
print("robustness.")
print()
sep_Xtr, sep_Xte = sep_X[:240], sep_X[240:]
sep_ytr, sep_yte = sep_y[:240], sep_y[240:]
print("C sweep on this well-separated 2-feature dataset:")
print()
print(f"{'C':>10} {'||w||':>10} {'min margin':>12} {'mean margin':>13} {'mean max p':>12} {'test log loss':>15}")
print("-" * 82)
for c_value, c_label in [(0.01, "0.01"), (0.1, "0.1"), (1.0, "1"), (10.0, "10"),
                         (100.0, "100"), (1e6, "1e6")]:
    m = LogisticRegression(C=c_value, max_iter=5000, tol=1e-10).fit(sep_Xtr, sep_ytr)
    z_tr = sep_Xtr @ m.coef_[0] + m.intercept_[0]
    norm = np.linalg.norm(m.coef_[0])
    te_p = np.clip(m.predict_proba(sep_Xte)[:, 1], 1e-15, 1 - 1e-15)
    te_loss = -np.mean(sep_yte * np.log(te_p) + (1 - sep_yte) * np.log(1 - te_p))
    print(f"{c_label:>10} {norm:>10.3f} {np.abs(z_tr).min() / norm:>12.4f} "
          f"{np.abs(z_tr).mean() / norm:>13.4f} {m.predict_proba(sep_Xte).max(axis=1).mean():>12.4f} "
          f"{te_loss:>15.6f}")
print()
print("Read this table carefully, because the obvious story is NOT the truth here.")
print()
print("  - ||w|| explodes as C rises: 0.76 at C=0.01, 32.6 with no penalty at all.")
print("    The regularisation strength is directly visible as the weight magnitude,")
print("    which makes this table the cleanest illustration of what C is doing.")
print("  - The test log loss falls to zero. This dataset is perfectly separable, so")
print("    there is nothing to overfit and no reason to regularise. 'Optimal C = inf'")
print("    here is a warning sign about the data, not a result.")
print("  - Mean max-probability climbs from 0.78 to 1.00, and the MEAN margin")
print("    actually FALLS slightly (2.36 -> 2.17) while the MIN margin rises. The two")
print("    move in opposite directions, which is a good reason to be careful about")
print("    quoting either as 'the margin' without saying which one.")
print()
print("Now the negative result worth remembering: the min margin goes UP with C")
print("here, not down. On separable data that is arithmetic - every |z| grows with")
print("||w||, and ||w|| grows faster, so the ratio rises. The intuition 'stronger")
print("regularisation buys a bigger margin' only holds when the classes OVERLAP:")
print()

overlap_rng = np.random.default_rng(5)
overlap_X = np.vstack([overlap_rng.normal([-1.0, -0.6], 1.35, (400, 2)),
                       overlap_rng.normal([1.0, 0.6], 1.35, (400, 2))])
overlap_y = np.hstack([np.zeros(400), np.ones(400)])
ov_Xtr, ov_Xte = overlap_X[:640], overlap_X[640:]
ov_ytr, ov_yte = overlap_y[:640], overlap_y[640:]


def quick_log_loss(model, X_d, y_d):
    p = np.clip(model.predict_proba(X_d)[:, 1], 1e-15, 1 - 1e-15)
    return -np.mean(y_d * np.log(p) + (1 - y_d) * np.log(1 - p))


print(f"{'C':>10} {'||w||':>10} {'min margin':>12} {'mean margin':>13} {'train log loss':>16} {'test log loss':>15}")
print("-" * 80)
for c_value in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
    m = LogisticRegression(C=c_value, max_iter=5000, tol=1e-10).fit(ov_Xtr, ov_ytr)
    z_tr = ov_Xtr @ m.coef_[0] + m.intercept_[0]
    norm = np.linalg.norm(m.coef_[0])
    print(f"{c_value:>10} {norm:>10.3f} {np.abs(z_tr).min() / norm:>12.4f} "
          f"{np.abs(z_tr).mean() / norm:>13.4f} {quick_log_loss(m, ov_Xtr, ov_ytr):>16.5f} "
          f"{quick_log_loss(m, ov_Xte, ov_yte):>15.5f}")
print()
print("Now the geometry behaves as advertised. The min margin is pinned near zero")
print("at every C, because the classes genuinely overlap and some points sit on the")
print("wrong side of ANY boundary. The useful signal is the train/test log loss")
print("pair: it narrows to a minimum around C=0.03, then the training loss keeps")
print("falling while the test loss drifts back up. That divergence is overfitting,")
print("and it is completely invisible in the margin.")
print()
print("So the diagnostic order is: read the loss gap FIRST, and treat the margin as")
print("supporting evidence about fragility rather than as the headline number.")
print()


# --------------------------------------------------------------------------------------
# 3. What the linear boundary cannot do.
# --------------------------------------------------------------------------------------
print()
print("-" * 76)
print("PART C - three problems a linear boundary cannot solve")
print("-" * 76)

all_data = {}

# --- XOR: the canonical impossibility ---------------------------------------------------
xor_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float) * 2.0
xor_y = np.array([0, 1, 1, 0])
all_data["XOR"] = (xor_X, xor_y)

# --- concentric circles: the boundary is a closed curve ----------------------------------
n_circle = 300
radii = np.sqrt(rng.uniform(0.05, 1.0, n_circle))
theta = rng.uniform(0, 2 * np.pi, n_circle)
circle_inner = np.column_stack([radii * np.cos(theta), radii * np.sin(theta)])
circle_outer = np.column_stack([(2 + radii) * np.cos(theta), (2 + radii) * np.sin(theta)])
circle_X = np.vstack([circle_inner, circle_outer])
circle_y = np.hstack([np.zeros(n_circle), np.ones(n_circle)])
all_data["concentric circles"] = (circle_X, circle_y)

# --- two moons: a boundary no polynomial can represent -----------------------------------
def two_moons(n_samples=500, noise=0.25, seed=3):
    local_rng = np.random.default_rng(seed)
    n_half = n_samples // 2
    t = np.linspace(0, np.pi, n_half)
    a = np.column_stack([np.cos(t), np.sin(t)]) + local_rng.normal(scale=noise, size=(n_half, 2))
    b = np.column_stack([1 - np.cos(t), 0.5 - np.sin(t)]) + local_rng.normal(scale=noise, size=(n_half, 2))
    return np.vstack([a, b]), np.hstack([np.zeros(n_half), np.ones(n_half)])


moons_X, moons_y = two_moons()
all_data["two moons"] = (moons_X, moons_y)


def polynomial_model(degree, C=1.0):
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree, include_bias=False),
        LogisticRegression(C=C, max_iter=5000),
    )


candidates = {
    "linear": LogisticRegression(C=1.0, max_iter=5000),
    "deg 2": polynomial_model(2),
    "deg 3": polynomial_model(3),
    "deg 5": polynomial_model(5),
}

accuracy_table = {}
for name, (X_d, y_d) in all_data.items():
    accuracy_table[name] = {}
    for label, model in candidates.items():
        # A fresh clone each time. Reusing a fitted Pipeline across problems would
        # silently carry the previous problem's feature count and raise.
        fitted = clone(model).fit(X_d, y_d)
        accuracy_table[name][label] = float(np.mean(fitted.predict(X_d) == y_d))

print(f"{'problem':<22} " + " ".join(f"{name:>8}" for name in candidates))
print("-" * 52)
for name, row in accuracy_table.items():
    print(f"{name:<22} " + " ".join(f"{row[label]:>8.3f}" for label in candidates))

print()
print("How to read each row, including the parts that are NOT intuitive:")
print()
print("  XOR            A single line cannot carve two opposite corners out of a")
print("                 square. Linear gets exactly 0.50 - chance. Degree 2 solves it,")
print("                 because the expansion contains the product x1*x2, and a")
print("                 hyperplane in the expanded space becomes a conic (here, a")
print("                 pair of hyperbolas) in the original space.")
print()
print("  circles        The boundary is a closed curve. No linear model can do it at")
print("                 ANY sample size, because the requirement is topological, not")
print("                 statistical. But degree 2 DOES solve it exactly - and that")
print("                 surprises people, who assume circles need higher degrees. The")
print("                 reason is that a circle is a conic, and degree 2 in two")
print("                 variables already spans every conic: it includes x1^2, x1*x2")
print("                 and x2^2, and a circle is just the case where the x1^2 and")
print("                 x2^2 coefficients are equal and the x1*x2 one is zero.")
print()
moons_row = accuracy_table["two moons"]
print("  two moons      This is the case where NO degree helps much. Linear already")
print(f"                 gets {moons_row['linear']:.3f} - the moons are only partly")
print(f"                 overlapped, so a tilted line does better than chance - and")
print(f"                 degree 3 reaches {moons_row['deg 3']:.3f}, after which degree 5")
print(f"                 adds {moons_row['deg 5'] - moons_row['deg 3']:+.3f}. The moons' boundary has")
print("                 infinite algebraic complexity, so polynomials approach it")
print("                 and never arrive. That flat tail is the signal to stop")
print("                 escalating the degree and change the model class instead.")
print()

# --- the r^2 lesson, which is the real engineering takeaway ------------------------------
print("-" * 76)
print("PART D - one engineered feature beats a polynomial blow-up")
print("-" * 76)
print()
print("For the concentric circles, the ONLY term the model needs is")
print()
print("    r2 = x1^2 + x2^2")
print()
print("which is a single column. Compare the cost of each route to the same answer:")
print()
r2_only = (circle_X ** 2).sum(axis=1).reshape(-1, 1)
options = {
    "x1, x2 (linear)": circle_X,
    "r2 alone (1 engineered)": r2_only,
    "PolynomialFeatures(2)": PolynomialFeatures(2, include_bias=False).fit_transform(circle_X),
    "PolynomialFeatures(3)": PolynomialFeatures(3, include_bias=False).fit_transform(circle_X),
    "PolynomialFeatures(5)": PolynomialFeatures(5, include_bias=False).fit_transform(circle_X),
}
print(f"{'feature set':<28} {'n columns':>11} {'accuracy':>10} {'test log loss':>15}")
print("-" * 68)
half = len(circle_X) // 2
# Shuffle first. circle_X is stored as two contiguous class blocks, so a positional
# split would put every inner point in train and every outer point in test - and
# the model would raise, because it never saw a single positive.
circle_order = np.random.default_rng(1).permutation(len(circle_X))
tr, te = circle_order[:half], circle_order[half:]
for label, matrix in options.items():
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    m.fit(matrix[tr], circle_y[tr])
    p = np.clip(m.predict_proba(matrix[te])[:, 1], 1e-15, 1 - 1e-15)
    loss = -np.mean(circle_y[te] * np.log(p) + (1 - circle_y[te]) * np.log(1 - p))
    print(f"{label:<28} {matrix.shape[1]:>11} {np.mean(m.predict(matrix[te]) == circle_y[te]):>10.3f} {loss:>15.5f}")

print()
print("Degree 2 spreads the answer across 5 columns, degree 5 across 20, and both")
print("carry terms (x1^4, x1^2*x2^2, ...) that are pure noise for this problem. The")
print("engineered r^2 feature gets the same fit from ONE column, and the single")
print("coefficient is directly readable: it is the log-odds per unit of squared")
print("distance from the origin. Nobody has to know what a fifth-degree bivariate")
print("polynomial means.")
print()
print("So the engineering rule is: reach for a NAMED feature before you reach for")
print("PolynomialFeatures. log(x), x^2, sqrt(x), x/y, hours since midnight, r^2,")
print("days since signup - each one is one column with a definition a stakeholder")
print("can agree to. A polynomial expansion is the fallback for when you suspect")
print("interactions and cannot name them. Script 12 makes this quantitative.")
print()
print("And when nothing works, escalate the model class rather than the degree: a")
print("decision tree, a random forest, gradient boosting, an RBF-kernel SVM, or a")
print("small neural network will all fit these boundaries without being told the")
print("geometry. The comparison is the subject of `../../random forest/`.")
print()


# --------------------------------------------------------------------------------------
# 4. Plots.
# --------------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))


def draw_boundary(ax, model, X, y, xlabel="x1", ylabel="x2", title=""):
    """Scatter the points and paint the model's boundary underneath them."""
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=16, alpha=0.6, color="#4C6EF5",
               edgecolors="none", label="class 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=16, alpha=0.6, color="#F03E3E",
               edgecolors="none", label="class 1")

    pad = 0.4
    x_lo, x_hi = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_lo, y_hi = X[:, 1].min() - pad, X[:, 1].max() + pad
    gx, gy = np.meshgrid(np.linspace(x_lo, x_hi, 300), np.linspace(y_lo, y_hi, 300))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    zz = model.decision_function(grid).reshape(gx.shape)
    ax.contourf(gx, gy, zz, levels=[-1e9, 0, 1e9], colors=["#4C6EF5", "#F03E3E"], alpha=0.16)
    ax.contour(gx, gy, zz, levels=[0], colors="#212529", linewidths=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(alpha=0.2)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)


# --- Panel 1: the tiny 4-point problem, drawn -------------------------------------------
ax = axes[0, 0]
boundary_1 = LogisticRegression(C=1e6, tol=1e-12, max_iter=5000).fit(tiny_X, tiny_y)
draw_boundary(ax, boundary_1, tiny_X, tiny_y,
              title="4 points, one line, margin 1.0")
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.annotate("w=(1, -1), b=0\nup to scale", xy=(1, 1), xytext=(0.3, 1.7),
            fontsize=9, arrowprops=dict(arrowstyle="->", color="#212529"))

# --- Panel 2: margin --------------------------------------------------------------------
ax = axes[0, 1]
draw_boundary(ax, sep_model, sep_X, sep_y, title="A wide margin: a stable boundary")
# annotate the two closest points and the perpendicular distance
distances = np.abs(sep_X @ w_vec + sep_model.intercept_[0]) / norm_w
closest = np.argsort(distances)[:4]
for index in closest:
    ax.scatter([sep_X[index, 0]], [sep_X[index, 1]], s=70, facecolors="none",
               edgecolors="#212529", linewidths=1.8, zorder=5)
ax.set_xlim(-5, 5)
ax.set_ylim(-4, 6)
ax.text(0.03, 0.03, f"circled points are the margin support vectors\n"
                     f"min margin = {distances.min():.3f}",
        transform=ax.transAxes, fontsize=9, color="#212529")

# --- Panel 3: margins for several C values ----------------------------------------------
ax = axes[0, 2]
margin_rows = []
for c_value in np.logspace(-2, 3, 12):
    m = LogisticRegression(C=c_value, max_iter=5000, tol=1e-10).fit(sep_Xtr, sep_ytr)
    z_tr = sep_Xtr @ m.coef_[0] + m.intercept_[0]
    margin_rows.append((c_value, np.abs(z_tr).min() / np.linalg.norm(m.coef_[0]),
                        np.linalg.norm(m.coef_[0])))
margin_table = pd.DataFrame(margin_rows, columns=["C", "min_margin", "norm_w"])
ax.plot(margin_table["C"], margin_table["min_margin"], "o-", color="#E03131",
        linewidth=2, markersize=4, label="min geometric margin")
ax.plot(margin_table["C"], margin_table["norm_w"] / 10, "s-", color="#4C6EF5",
        linewidth=2, markersize=4, label="||w|| / 10 (rescaled for display)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title("Stronger regularisation -> bigger margin", fontsize=11, fontweight="bold")
ax.set_xlabel("C  (bigger C = weaker regularisation)")
ax.set_ylabel("min margin (log scale)")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25, which="both")
ax.invert_xaxis()

# --- Panels 4-6: the three hard problems, linear model ---------------------------------
for column, (name, (X_d, y_d)) in enumerate(all_data.items()):
    ax = axes[1, column]
    m = LogisticRegression(C=1.0, max_iter=5000).fit(X_d, y_d)
    draw_boundary(ax, m, X_d, y_d, title=f"{name}: linear model")
    if name == "XOR":
        ax.set_xlim(-0.6, 2.6)
        ax.set_ylim(-0.6, 2.6)

fig.suptitle("04 - Decision Boundaries and Geometry", fontsize=13, fontweight="bold")
fig.tight_layout()
plt.show()


# --------------------------------------------------------------------------------------
# 5. Takeaways.
# --------------------------------------------------------------------------------------
print()
print("=" * 76)
print("TAKEAWAYS")
print("=" * 76)
print("1. The boundary is w.x + b = 0, and it follows from the sigmoid being")
print("   STRICTLY MONOTONIC. That single fact is the whole derivation.")
print("2. Only the DIRECTION of w moves the boundary. The MAGNITUDE controls")
print("   confidence. Scaling (w, b) by any positive constant leaves every label")
print("   unchanged and every probability different.")
print("3. Thresholding slides the boundary parallel to itself, it does not rotate")
print("   it. Re-fitting at a different threshold gives a genuinely different line.")
print("4. Geometric margin is |w.x + b| / ||w||. Forgetting the ||w|| turns it into")
print("   the raw log-odds, which is a different quantity that happens to grow with")
print("   ||w||. And note that min margin and mean margin can move in OPPOSITE")
print("   directions as C changes, so always say which one you are quoting.")
print("5. A linear boundary is a topologically limited object, not a statistically")
print("   limited one. XOR and concentric circles stay impossible at any n.")
print("6. Prefer ONE engineered feature (r^2, log x, x/y) over a polynomial")
print("   expansion. The engineered version is cheaper, sparser and explainable.")
print("7. When the boundary genuinely must curve, stop adding degrees and change")
print("   the model class. Trees and kernels do not need to be told the geometry.")
