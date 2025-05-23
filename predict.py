import os
import numpy as np
import pandas as pd
import pickle
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Define paths
BASE_DIR = os.path.dirname(__file__)
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'model', 'stacked_model.keras')
SCALER_SAVE_PATH = os.path.join(BASE_DIR, 'model', 'scaler.pkl')
CSV_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'vehicular_data.csv')

# Define the feature columns (ensure these match your training data)
feature_columns = [
    "Engine rpm", 
    "Lub oil pressure", 
    "Fuel pressure", 
    "Coolant pressure", 
    "lub oil temp", 
    "Coolant temp"
]

# Create or update scaler.pkl using vehicular_data.csv at startup
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    X = df[feature_columns].values
    scaler = StandardScaler()
    scaler.fit(X)
    # Save the fitted scaler to scaler.pkl
    with open(SCALER_SAVE_PATH, 'wb') as f:
        pickle.dump(scaler, f)
else:
    raise FileNotFoundError(f"CSV file not found at {CSV_PATH}")

# Load the trained model and the scaler from the saved file
model = load_model(MODEL_SAVE_PATH)
with open(SCALER_SAVE_PATH, 'rb') as f:
    scaler = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract data from the form and convert to float
            engine_rpm = float(request.form['Engine_rpm'])
            lub_oil_pressure = float(request.form['Lub_oil_pressure'])
            fuel_pressure = float(request.form['Fuel_pressure'])
            coolant_pressure = float(request.form['Coolant_pressure'])
            lub_oil_temp = float(request.form['lub_oil_temp'])
            coolant_temp = float(request.form['Coolant_temp'])
        except ValueError as e:
            return f"Invalid input: {e}"

        # Build the feature array (order must match training data)
        features = [engine_rpm, lub_oil_pressure, fuel_pressure, coolant_pressure, lub_oil_temp, coolant_temp]
        X = np.array([features])

        # Transform the input data using the pre-fitted scaler
        X_scaled = scaler.transform(X)

        # Generate prediction: probability and binary outcome
        y_pred_prob = model.predict(X_scaled).ravel()[0]
        y_pred = int(y_pred_prob > 0.5)

        # Build a dictionary of input values to display in the result page
        input_values = {
            "Engine rpm": engine_rpm,
            "Lub oil pressure": lub_oil_pressure,
            "Fuel pressure": fuel_pressure,
            "Coolant pressure": coolant_pressure,
            "lub oil temp": lub_oil_temp,
            "Coolant temp": coolant_temp
        }

        # Create a detailed suggestion based on the prediction
        if y_pred == 1:
            suggestion = (
                "Engine Failure Predicted:\n"
                f"  - Engine rpm: {engine_rpm}\n"
                f"  - Lub oil pressure: {lub_oil_pressure}\n"
                f"  - Fuel pressure: {fuel_pressure}\n"
                f"  - Coolant pressure: {coolant_pressure}\n"
                f"  - Lub oil temp: {lub_oil_temp}\n"
                f"  - Coolant temp: {coolant_temp}\n\n"
                "The sensor readings indicate that one or more parameters may be outside the normal operating range. "
                "This could be a sign of potential mechanical issues. It is recommended to inspect the engine system, "
                "check for leaks, verify fluid levels, and consult a mechanic for a thorough diagnosis."
            )
        else:
            suggestion = (
                "Engine Operating Normally:\n"
                f"  - Engine rpm: {engine_rpm}\n"
                f"  - Lub oil pressure: {lub_oil_pressure}\n"
                f"  - Fuel pressure: {fuel_pressure}\n"
                f"  - Coolant pressure: {coolant_pressure}\n"
                f"  - Lub oil temp: {lub_oil_temp}\n"
                f"  - Coolant temp: {coolant_temp}\n\n"
                "The sensor readings are within expected ranges, suggesting that the engine is functioning normally. "
                "It is still recommended to continue regular maintenance and periodic monitoring to ensure ongoing performance."
            )

        # Render the result page with predictions and detailed suggestion
        return render_template('result.html',
                               result="Engine Failure" if y_pred == 1 else "Engine Normal",
                               input_values=input_values,
                               probability=round(y_pred_prob, 4),
                               suggestion=suggestion)
    # Render the input form for GET requests
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
