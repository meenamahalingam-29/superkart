import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
superkart_api = Flask("SuperKart Sales Forecast API")
CORS(superkart_api)

# Load the trained model pipeline (includes encoder + model)
# The model file is copied into the app root by Dockerfile.
model = joblib.load("superkart_rf_sales_forecast_v1_0.joblib")

FEATURES = [
    "Product_Id",
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_Type",
    "Product_MRP",
    "Store_Id",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Store_Age"
]

@superkart_api.get("/")
def home():
    return "Welcome to the SuperKart Sales Forecast API!"

@superkart_api.post("/v1/predict")
def predict_single():
    payload = request.get_json()

    # Create one-row dataframe in the exact feature order
    sample = {feat: payload[feat] for feat in FEATURES}
    input_df = pd.DataFrame([sample])

    pred = float(model.predict(input_df)[0])

    return jsonify({"Predicted_Product_Store_Sales_Total": round(pred, 2)})

@superkart_api.post("/v1/predictbatch")
def predict_batch():
    # Expect a CSV file with the same feature columns
    file = request.files["file"]
    input_df = pd.read_csv(file)

    # Ensure required columns exist
    missing = [c for c in FEATURES if c not in input_df.columns]
    if missing:
        return jsonify({"error": f"Missing required columns: {missing}"}), 400

    preds = model.predict(input_df[FEATURES]).tolist()
    preds = [round(float(x), 2) for x in preds]

    return jsonify({"predictions": preds})

if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860, debug=False)