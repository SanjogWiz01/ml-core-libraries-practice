# Supervised Learning

Supervised learning is machine learning where the model learns from **labelled**
data: every training example comes with the "correct answer" attached.

The dataset is split into input features `X` and target labels `y`. The model
learns a mapping `f: X -> y` and is then able to predict the label for inputs it
has never seen.

## The Two Sub-Problem Types

| Type | Target `y` | Goal | Example algorithms |
|------|-------------|------|--------------------|
| **Regression** | continuous number | predict a quantity | LinearRegression, Ridge, Lasso, SVR |
| **Classification** | discrete category | predict a class | LogisticRegression, DecisionTree, SVC, k-NN |

## The Canonical Workflow

```
1. Load data          -> X (features), y (labels)
2. Split              -> train / validation / test   (never test on train data)
3. Preprocess         -> scale, encode, impute, engineer features
4. Fit (train)        -> model.fit(X_train, y_train)      <- learns parameters
5. Predict            -> model.predict(X_test)           <- applies learned parameters
6. Evaluate           -> R2, RMSE, MAE  /  accuracy, F1, AUC
7. Tune + refit       -> cross-validate, adjust settings, repeat
8. Deploy             -> serialise the fitted model, serve predictions
```

## Bias-Variance Tradeoff

Supervised algorithms differ mainly in *how* they trade off the two error
sources:

- **High bias, low variance** - simple models (LinearRegression, LogisticRegression).
  Underfits, but predictions are stable and easy to interpret.
- **Low bias, high variance** - flexible models (deep nets, unpruned trees).
  Fits training data closely, but may generalise poorly.
- **Ensembles** (RandomForest, GradientBoosting) and **regularisation**
  (Ridge, Lasso) exist specifically to grab the best of both.

## Algorithm Map

- LinearRegression, Ridge, Lasso, ElasticNet  -> `linear-regression/`
- PolynomialFeatures expansion                 -> `linear-regression/`
- DecisionTree, RandomForest, GradientBoosting -> `../../random forest/`
- k-NN, SVM, NaiveBayes, NeuralNets           -> planned

## Current Contents

| Folder | Status |
|--------|--------|
| [`linear-regression/`](linear-regression/) | complete - 3 guides + 17 runnable scripts |
| `logistic-regression/` | planned |
| `decision-trees/` | planned |
| `svm/` | planned |

## How To Run The Code

```bash
pip install numpy pandas scikit-learn matplotlib
cd types/supervised/linear-regression
python 01_linear_regression_from_scratch.py
```

## Related

- [`../unsupervised/`](../unsupervised/)
- [`../reinforcement-learning/`](../reinforcement-learning/)
