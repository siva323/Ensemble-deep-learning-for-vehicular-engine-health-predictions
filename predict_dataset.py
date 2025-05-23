import os
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Define the model save path (adjust as needed)
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), 'model', 'stacked_model.keras')

def predict_csv():
    # Define the path to your CSV file
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'raw', 'vehicular_data.csv')
    
    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    
    # Specify the feature columns used for prediction
    feature_columns = [
        "Engine rpm", 
        "Lub oil pressure", 
        "Fuel pressure", 
        "Coolant pressure", 
        "lub oil temp", 
        "Coolant temp"
    ]
    
    # Extract features from the DataFrame
    X = df[feature_columns].values
    
    # Scale features using StandardScaler (for production, load a saved scaler)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Load the trained model
    model = load_model(MODEL_SAVE_PATH)
    
    # Generate predictions: probability and binary outcome
    y_pred_prob = model.predict(X_scaled).ravel()
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Add the predictions as new columns in the DataFrame
    df["Predicted Engine Failure"] = y_pred
    df["Prediction Probability"] = y_pred_prob
    
    # Save the updated DataFrame to a new CSV file
    output_csv_path = os.path.join(os.path.dirname(__file__), 'vehicular_data_with_predictions.csv')
    df.to_csv(output_csv_path, index=False)
    print("Predictions saved to:", output_csv_path)

if __name__ == '__main__':
    predict_csv()
