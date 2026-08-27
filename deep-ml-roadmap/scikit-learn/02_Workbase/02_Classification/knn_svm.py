"""KNN and SVM — instance-based and margin-based classifiers."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score


def find_best_k(X_train, y_train, k_range):
    """Return CV accuracy for each K value."""
    cv_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
        cv_scores.append(scores.mean())
    return cv_scores


def main():
    print("=" * 55)
    print("KNN & SVM — Wine Dataset (3-class classification)")
    print("=" * 55)

    data = load_wine()
    X, y = data.data, data.target

    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features, {len(data.target_names)} classes")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Both KNN and SVM are sensitive to scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ============================================================
    # K-Nearest Neighbors
    # ============================================================
    print("\n=== K-Nearest Neighbors ===")

    k_range = range(1, 31)
    cv_scores = find_best_k(X_train_s, y_train, k_range)

    best_k = k_range.start + np.argmax(cv_scores)
    print(f"Best K (CV): {best_k}  →  CV accuracy: {max(cv_scores):.4f}")

    knn = KNeighborsClassifier(n_neighbors=best_k, weights='uniform')
    knn.fit(X_train_s, y_train)
    y_pred_knn = knn.predict(X_test_s)

    print(f"Test accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_knn, target_names=data.target_names))

    # Plot K vs accuracy
    plt.figure(figsize=(8, 4))
    plt.plot(k_range, cv_scores, 'bo-')
    plt.axvline(best_k, color='r', linestyle='--', label=f'Best K={best_k}')
    plt.xlabel("K (n_neighbors)")
    plt.ylabel("CV Accuracy")
    plt.title("KNN: K vs Cross-Validation Accuracy")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.show()

    # ============================================================
    # Support Vector Machine
    # ============================================================
    print("\n=== Support Vector Machine ===")

    kernels = ['linear', 'rbf', 'poly']
    for kernel in kernels:
        svm = SVC(kernel=kernel, C=1.0, gamma='scale', random_state=42)
        cv_acc = cross_val_score(svm, X_train_s, y_train, cv=5, scoring='accuracy').mean()
        svm.fit(X_train_s, y_train)
        test_acc = accuracy_score(y_test, svm.predict(X_test_s))
        print(f"  kernel={kernel:8s}: CV acc={cv_acc:.4f}, test acc={test_acc:.4f}")

    # Best: RBF with tuned C
    print("\n--- SVM (RBF): effect of C ---")
    for C in [0.01, 0.1, 1, 10, 100]:
        svm = SVC(kernel='rbf', C=C, gamma='scale', random_state=42)
        svm.fit(X_train_s, y_train)
        acc = accuracy_score(y_test, svm.predict(X_test_s))
        n_sv = svm.n_support_.sum()
        print(f"  C={C:5}: test_acc={acc:.4f}, support_vectors={n_sv}")

    # Final SVM report
    best_svm = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
    best_svm.fit(X_train_s, y_train)
    print(f"\nBest SVM test accuracy: {accuracy_score(y_test, best_svm.predict(X_test_s)):.4f}")
    print(classification_report(y_test, best_svm.predict(X_test_s), target_names=data.target_names))

    # ============================================================
    # Comparison
    # ============================================================
    print("\n=== KNN vs SVM Summary ===")
    for name, model in [("KNN (best K)", knn), ("SVM (RBF, C=10)", best_svm)]:
        acc = accuracy_score(y_test, model.predict(X_test_s))
        print(f"  {name:20s}: {acc:.4f}")


if __name__ == "__main__":
    main()
