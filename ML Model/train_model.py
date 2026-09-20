import json
import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_data():
    """Load Nepali assets data"""
    with open("data/nepali_assets.json", "r", encoding="utf-8") as f:
        data = json.load(f)
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

def train_model():
    """Train ML model for Nepali asset description prediction"""
    assets = load_data()
    X, y = prepare_data(assets)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Get unique class count
    n_classes = len(np.unique(y_encoded))
    print(f"Number of classes: {n_classes}")
    print(f"Class names: {le.classes_}")
    print(f"Total samples: {len(X)}")
    
    # Use all data for training since dataset is small
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_vec = vectorizer.fit_transform(X)
    
    # Train Naive Bayes classifier on all data
    model = MultinomialNB()
    model.fit(X_vec, y_encoded)
    
    # Save model and artifacts
    models_dir = os.path.join(SCRIPT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, os.path.join(models_dir, "nepali_asset_model.pkl"))
    joblib.dump(vectorizer, os.path.join(models_dir, "vectorizer.pkl"))
    joblib.dump(le, os.path.join(models_dir, "label_encoder.pkl"))
    
    # Save category mapping
    categories = le.classes_.tolist()
    with open(os.path.join(models_dir, "categories.json"), "w", encoding="utf-8") as f:
        json.dump(categories, f, ensure_ascii=False, indent=2)
    
    print(f"\nModel trained and saved!")
    print(f"Categories: {categories}")
    print(f"Using all {len(X)} samples for training")
    
    return model, vectorizer, le

if __name__ == "__main__":
    model, vectorizer, le = train_model()