import os
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, url_for, send_from_directory
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    mean_squared_error,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    log_loss,
    roc_curve
)
from data_preprocessing import get_train_test_split  # Ensure this module is available

app = Flask(__name__)

# Define the path to the saved model (adjust as needed)
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), 'model', 'stacked_model.keras')

# -------------------------------
# Performance Metrics Functions
# -------------------------------
def compute_metrics(y_true, y_pred, y_pred_prob):
    """
    Compute performance metrics for binary classification.
    """
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_prob)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_prob))
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    logloss = log_loss(y_true, y_pred_prob)
    
    metrics = {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "roc_auc": roc_auc,
        "rmse": rmse,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "balanced_accuracy": balanced_acc,
        "log_loss": logloss
    }
    
    return metrics

# -------------------------------
# Endpoint to Serve Metrics Files
# -------------------------------
@app.route('/metrics/<filename>')
def metrics_files(filename):
    """
    Serve files from the data/metrics directory.
    """
    metrics_dir = os.path.join(os.path.dirname(__file__), 'data', 'metrics')
    return send_from_directory(metrics_dir, filename)

# -------------------------------
# Flask Application Route for Metrics View
# -------------------------------
@app.route('/metrics')
def metrics_view():
    """
    Loads test data, makes predictions with the trained model, computes performance metrics,
    generates a ROC curve plot, and renders all information along with the training loss plot.
    """
    # Load test data using the get_train_test_split function
    _, X_test, _, y_test = get_train_test_split()
    
    # Load the trained model
    model = load_model(MODEL_SAVE_PATH)
    
    # Make predictions on the test set
    y_pred_prob = model.predict(X_test).ravel()
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Compute performance metrics
    metrics = compute_metrics(y_test, y_pred, y_pred_prob)
    
    # Ensure the metrics directory exists before saving images
    metrics_dir = os.path.join(os.path.dirname(__file__), 'data', 'metrics')
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)
    
    # Generate ROC Curve plot
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve (AUC = %0.4f)" % metrics["roc_auc"])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    roc_curve_path = os.path.join(metrics_dir, 'roc_curve.png')
    plt.savefig(roc_curve_path)
    plt.close()
    
    # Generate URLs for the plots using the metrics_files endpoint
    training_loss_url = url_for('metrics_files', filename='training_loss.png')
    roc_curve_url = url_for('metrics_files', filename='roc_curve.png')
    
    # Render the metrics template with metrics and plot URLs
    return render_template("metrics.html", metrics=metrics, roc_curve_url=roc_curve_url, training_loss_url=training_loss_url)

@app.route('/')
def index():
    return "Welcome! Visit /metrics to view performance metrics."

if __name__ == '__main__':
    app.run(debug=True)
