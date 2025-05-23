import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from imblearn.over_sampling import SMOTE
# Define data directories and file paths
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'vehicular_data.csv')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")
    else:
        print(f"Directory already exists: {path}")

# Ensure the processed data directory exists
ensure_directory(PROCESSED_DATA_DIR)

def load_data():
    """
    Load raw data from CSV.
    """
    print("Loading data from:", RAW_DATA_PATH)
    try:
        df = pd.read_csv(RAW_DATA_PATH)
        print("Data loaded successfully. Shape:", df.shape)
    except Exception as e:
        print("Error loading data:", e)
        raise e
    return df

def preprocess_data(df):
    """
    Preprocess data:
    - Fill missing values.
    - Separate features and target.
    - Scale features.
    - Save processed features and target as CSV files.
    """
    print("Preprocessing data...")
    # Fill missing values with column means
    df.fillna(df.mean(), inplace=True)
    
    # Define target column and feature columns
    target_column = "Engine Condition"
    feature_columns = ["Engine rpm", "Lub oil pressure", "Fuel pressure", 
                       "Coolant pressure", "lub oil temp", "Coolant temp"]
    
    # Validate presence of required columns
    missing_features = set(feature_columns + [target_column]) - set(df.columns)
    if missing_features:
        raise KeyError(f"Columns {missing_features} not found in the data.")
    
    # Separate features and target
    X = df[feature_columns]
    y = df[target_column]
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Handle class imbalance with SMOTE
   
    
    # Convert scaled features and target to DataFrames
    X_df = pd.DataFrame(X_scaled, columns=feature_columns)
    y_df = pd.DataFrame(y, columns=[target_column])
    
    # Define file paths
    features_path = os.path.join(PROCESSED_DATA_DIR, 'features.csv')
    target_path = os.path.join(PROCESSED_DATA_DIR, 'target.csv')
    
    print(f"Saving processed features to: {features_path}")
    print(f"Saving processed target to: {target_path}")
    
    try:
        X_df.to_csv(features_path, index=False)
        y_df.to_csv(target_path, index=False)
        print("Files saved successfully.")
    except Exception as e:
        print("Error saving files:", e)
        raise e
    
    return X_scaled, y

def get_train_test_split(test_size=0.2, random_state=42):
    """
    Load, preprocess, and split data into train and test sets.
    """
    df = load_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    print("Data split into training and testing sets.")
    print("Training set shape:", X_train.shape, "Testing set shape:", X_test.shape)
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    df = load_data()
    X, y = preprocess_data(df)
    print("Data loaded and preprocessed.")
    print("Feature shape:", X.shape)
    print("Target shape:", y.shape)
