"""Iris Classification — multi-model comparison on a classic dataset."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)


def compare_models(X_train, X_test, y_train, y_test, class_names):
    models = {
        'LogisticRegression': Pipeline([('s', StandardScaler()),
                                        ('m', LogisticRegression(max_iter=1000, random_state=42))]),
        'KNN(k=7)':           Pipeline([('s', StandardScaler()),
                                        ('m', KNeighborsClassifier(n_neighbors=7))]),
        'DecisionTree':       Pipeline([('m', DecisionTreeClassifier(max_depth=4, random_state=42))]),
        'RandomForest':       Pipeline([('m', RandomForestClassifier(n_estimators=100, random_state=42))]),
        'GradientBoosting':   Pipeline([('m', GradientBoostingClassifier(n_estimators=100, random_state=42))]),
        'SVM(rbf)':           Pipeline([('s', StandardScaler()),
                                        ('m', SVC(kernel='rbf', C=10, gamma='scale', probability=True))]),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\n{'Model':20s}  {'CV Acc':>8}  {'CV±':>7}  {'Test Acc':>9}  {'Test F1':>8}")
    print(f"{'-'*20}  {'-'*8}  {'-'*7}  {'-'*9}  {'-'*8}")

    results = {}
    for name, pipe in models.items():
        cv = cross_val_score(pipe, X_train, y_train, cv=skf, scoring='accuracy', n_jobs=-1)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average='macro')
        print(f"{name:20s}  {cv.mean():8.4f}  {cv.std():7.4f}  {acc:9.4f}  {f1:8.4f}")
        results[name] = {'pipe': pipe, 'cv_mean': cv.mean(), 'test_acc': acc, 'y_pred': y_pred}

    return results


def plot_comparison(results):
    names  = list(results.keys())
    cv_acc = [results[n]['cv_mean'] for n in names]
    te_acc = [results[n]['test_acc'] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, cv_acc, width, label='CV Accuracy', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, te_acc, width, label='Test Accuracy', color='firebrick', alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha='right')
    ax.set_ylim(0.85, 1.02)
    ax.set_ylabel("Accuracy"); ax.set_title("Iris: Model Comparison")
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    plt.tight_layout(); plt.show()


def plot_confusion_matrices(results, class_names, y_test):
    n = len(results)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
    axes = axes.ravel()

    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res['y_pred'])
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks(range(3)); ax.set_xticklabels(class_names, rotation=30)
        ax.set_yticks(range(3)); ax.set_yticklabels(class_names)
        ax.set_title(name)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha='center', va='center',
                        color='white' if cm[i, j] > cm.max()/2 else 'black')

    for ax in axes[n:]:
        ax.axis('off')
    plt.suptitle("Confusion Matrices — Iris (all models)")
    plt.tight_layout(); plt.show()


def learning_curve_demo(X, y):
    """Show how training set size affects accuracy for best models."""
    from sklearn.model_selection import learning_curve

    print("\n--- Learning Curve (RandomForest) ---")
    pipe = Pipeline([('m', RandomForestClassifier(n_estimators=100, random_state=42))])
    train_sizes, train_scores, val_scores = learning_curve(
        pipe, X, y, cv=5, scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )

    plt.figure(figsize=(8, 4))
    plt.plot(train_sizes, train_scores.mean(1), 'b-', label='Train')
    plt.fill_between(train_sizes, train_scores.mean(1) - train_scores.std(1),
                     train_scores.mean(1) + train_scores.std(1), alpha=0.1, color='b')
    plt.plot(train_sizes, val_scores.mean(1), 'r-', label='Validation')
    plt.fill_between(train_sizes, val_scores.mean(1) - val_scores.std(1),
                     val_scores.mean(1) + val_scores.std(1), alpha=0.1, color='r')
    plt.xlabel("Training examples"); plt.ylabel("Accuracy")
    plt.title("Learning Curve — RandomForest on Iris")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()


def main():
    print("=" * 55)
    print("IRIS CLASSIFICATION — Multi-Model Comparison")
    print("=" * 55)

    data = load_iris()
    X, y = data.data, data.target
    class_names = data.target_names

    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {list(class_names)} (50 samples each)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = compare_models(X_train, X_test, y_train, y_test, class_names)

    best_name = max(results, key=lambda k: results[k]['cv_mean'])
    print(f"\nBest model: {best_name}")
    print(f"\nDetailed report ({best_name}):")
    print(classification_report(y_test, results[best_name]['y_pred'],
                                  target_names=class_names))

    plot_comparison(results)
    plot_confusion_matrices(results, class_names, y_test)
    learning_curve_demo(X, y)


if __name__ == "__main__":
    main()
