from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os
import numpy as np

app = Flask(__name__)
CORS(app)

# Load models safely
models = {}
model_files = {
    "Logistic Regression": "models/log_reg.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "Random Forest": "models/random_forest.pkl",
    "Support Vector Machine": "models/svm.pkl"
}

for name, path in model_files.items():
    try:
        models[name] = joblib.load(path)
    except Exception as e:
        print(f"Failed to load {name}: {e}")

# Load scaler & encoders
try:
    scaler = joblib.load("models/scaler.pkl")
except Exception as e:
    print(f"Failed to load scaler: {e}")
    scaler = None

try:
    encoders = joblib.load("models/encoders.pkl")
except Exception as e:
    print(f"Failed to load encoders: {e}")
    encoders = {}

def preprocess_input(data):
    df = pd.DataFrame([data])

    # Apply label encoding safely
    for col, le in encoders.items():
        if col in df.columns:
            try:
                # Replace unseen categories with -1
                df[col] = df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
            except Exception as e:
                print(f"Encoding failed for column {col}: {e}")
                df[col] = -1

    # Align with training features
    if scaler and hasattr(scaler, "feature_names_in_"):
        df = df.reindex(columns=scaler.feature_names_in_, fill_value=0)

    # Scale features safely
    if scaler:
        try:
            return scaler.transform(df)
        except Exception as e:
            print(f"Scaling failed: {e}")
            return df.values
    else:
        return df.values

@app.route("/", methods=["GET"])
def home():
    return app.send_static_file("index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "Backend is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        X = preprocess_input(data)

        results = []
        for name, model in models.items():
            try:
                # Some models (like SVM) may not have predict_proba
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X)[0][1]
                    pred = int(proba >= 0.5)
                    confidence = round(proba if pred == 1 else 1 - proba, 3)
                else:
                    # fallback for models without predict_proba
                    pred = int(model.predict(X)[0])
                    confidence = None
                    proba = None

                results.append({
                    "model": name,
                    "prediction": pred,
                    "confidence": confidence,
                    "churn_probability": proba
                })
            except Exception as e:
                results.append({
                    "model": name,
                    "error": f"Prediction failed: {e}"
                })

        return jsonify(results)

    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
