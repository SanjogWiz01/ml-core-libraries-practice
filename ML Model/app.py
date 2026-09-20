import json
import os
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# Load model artifacts
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "nepali_asset_model.pkl")
VECTORIZER_PATH = os.path.join(SCRIPT_DIR, "models", "vectorizer.pkl")
LE_PATH = os.path.join(SCRIPT_DIR, "models", "label_encoder.pkl")
CATEGORIES_PATH = os.path.join(SCRIPT_DIR, "models", "categories.json")

model = None
vectorizer = None
label_encoder = None
categories = []

def load_artifacts():
    global model, vectorizer, label_encoder, categories
    if not os.path.exists(MODEL_PATH):
        return False
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    label_encoder = joblib.load(LE_PATH)
    if os.path.exists(CATEGORIES_PATH):
        with open(CATEGORIES_PATH, "r", encoding="utf-8") as f:
            categories = json.load(f)
    return True

load_artifacts()

@app.route("/")
def index():
    return render_template("index.html", categories=categories)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "input_text" not in data:
        return jsonify({"error": "No input text provided"}), 400
    
    input_text = data["input_text"]
    if not input_text.strip():
        return jsonify({"error": "Input text is empty"}), 400
    
    # Vectorize and predict
    try:
        X_vec = vectorizer.transform([input_text])
        prediction = model.predict(X_vec)[0]
        confidence = np.max(model.predict_proba(X_vec)[0])
        category = label_encoder.inverse_transform([prediction])[0]
        
        return jsonify({
            "category": category,
            "confidence": float(confidence),
            "input": input_text
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)