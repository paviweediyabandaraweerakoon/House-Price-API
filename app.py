from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model on startup
try:
    model = joblib.load("model.pkl")
    feature_names = joblib.load("feature_names.pkl")
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None
    feature_names = None

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "House Price Prediction API (Flask) is running",
        "model_loaded": model is not None
    })

@app.route("/predict", methods=["POST"])
def predict_house_price():
    """Predict house price for single input"""
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        data = request.get_json()

        # Validate input
        required_fields = ["sqft", "bedrooms", "bathrooms", "age"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Convert to DataFrame (fixes sklearn warning about feature names)
        features = pd.DataFrame([{
            "sqft": float(data["sqft"]),
            "bedrooms": int(data["bedrooms"]),
            "bathrooms": float(data["bathrooms"]),
            "age": int(data["age"])
        }])

        # Prediction
        prediction = model.predict(features)[0]
        prediction = round(float(prediction), 2)

        # Confidence logic
        if 50000 <= prediction <= 800000:
            confidence = "High"
        elif 30000 <= prediction <= 1000000:
            confidence = "Medium"
        else:
            confidence = "Low"

        logger.info(f"Prediction made: ${prediction:,.2f}")

        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "input_features": data
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 400

@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    """Predict prices for multiple houses"""
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        data_list = request.get_json()
        if not isinstance(data_list, list):
            return jsonify({"error": "Input should be a list of house features"}), 400

        predictions = []
        for data in data_list:
            # Validate each entry
            for field in ["sqft", "bedrooms", "bathrooms", "age"]:
                if field not in data:
                    return jsonify({"error": f"Missing field in one of the entries: {field}"}), 400

            features = pd.DataFrame([{
                "sqft": float(data["sqft"]),
                "bedrooms": int(data["bedrooms"]),
                "bathrooms": float(data["bathrooms"]),
                "age": int(data["age"])
            }])

            prediction = model.predict(features)[0]
            prediction = round(float(prediction), 2)

            predictions.append({
                "input": data,
                "predicted_price": prediction
            })

        return jsonify({"predictions": predictions})

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({"error": f"Batch prediction failed: {str(e)}"}), 400


if __name__ == "__main__":
    # Run on port 5000 to avoid conflicts
    app.run(debug=True, port=5000)
