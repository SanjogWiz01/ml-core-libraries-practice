import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import joblib

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_data():
    """Load Nepali assets data"""
    data_path = os.path.join(SCRIPT_DIR, "..", "data", "nepali_assets.json")
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        # Fallback: check if data is in the ML Model directory
        fallback_path = os.path.join(SCRIPT_DIR, "data", "nepali_assets.json")
        if os.path.exists(fallback_path):
            with open(fallback_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            raise FileNotFoundError("nepali_assets.json not found")
    return data

def prepare_data(assets):
    """Prepare features and labels for ML training"""
    X = []
    y = []
    for asset in assets:
        # Combine name and base description as features
        text = f"{asset['name']} {asset['base_desc']}"
        X.append(text)
        y.append(asset["category"])
    return np.array(X), np.array(y)

def train_and_evaluate_models(X, y):
    """Train and evaluate multiple ML models"""
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.25, random_state=42)
    
    # Define models to evaluate
    models = {
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "SVC-linear": SVC(kernel='linear', probability=True, random_state=42),
    }
    
    results = {}
    
    for name, model in models.items():
        # Create pipeline with TF-IDF
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf', model)
        ])
        
        # Cross-validation (use min of 3 or number of samples - 1)
        cv_splits = min(3, len(X_train), max(1, len(np.unique(y_train))))
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv_splits)
        
        # Train on full training set
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        results[name] = {
            "pipeline": pipeline,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "test_accuracy": acc,
            "label_encoder": le
        }
        
        print(f"\n{name}:")
        print(f"  Cross-Val Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"  Test Accuracy: {acc:.4f}")
        print(f"  Classification Report:")
        try:
            print(classification_report(y_test, y_pred, target_names=le.classes_))
        except ValueError:
            print(classification_report(y_test, y_pred, labels=le.classes_, zero_division=0))
    
    return results

def train_best_model():
    """Train the best performing model and save artifacts"""
    assets = load_data()
    X, y = prepare_data(assets)
    
    print(f"Total samples: {len(X)}")
    print(f"Categories: {np.unique(y)}")
    
    # Train and evaluate models
    results = train_and_evaluate_models(X, y)
    
    # Select best model based on test accuracy
    best_model_name = max(results.keys(), key=lambda k: results[k]["test_accuracy"])
    best_result = results[best_model_name]
    
    print(f"\nBest model: {best_model_name} with test accuracy: {best_result['test_accuracy']:.4f}")
    
    # Save the best model and artifacts
    models_dir = os.path.join(SCRIPT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(best_result["pipeline"], os.path.join(models_dir, "nepali_asset_model.pkl"))
    joblib.dump(best_result["label_encoder"], os.path.join(models_dir, "label_encoder.pkl"))
    
    # Save categories
    categories = best_result["label_encoder"].classes_.tolist()
    with open(os.path.join(models_dir, "categories.json"), "w", encoding="utf-8") as f:
        json.dump(categories, f, ensure_ascii=False, indent=2)
    
    print(f"\nModel saved to {models_dir}/")
    print(f"Categories: {categories}")
    
    return best_result["pipeline"], best_result["label_encoder"]

if __name__ == "__main__":
    model, le = train_best_model()