import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# Load preprocessed data
X = pd.read_csv('data/processed/features.csv')
y = pd.read_csv('data/processed/target.csv').values.ravel()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base models
models_1 = [
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
    ('svm', make_pipeline(StandardScaler(), SVC(probability=True, C=2.0, kernel='rbf'))),
    ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)),
    ('dt', DecisionTreeClassifier(max_depth=10, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=7))
]

models_2 = [
    ('lr', LogisticRegression(C=1.5, max_iter=200)),
    ('svm', make_pipeline(StandardScaler(), SVC(probability=True, C=2.0, kernel='rbf'))),
    ('lda', LinearDiscriminantAnalysis()),
    ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)),
    ('ada', AdaBoostClassifier(n_estimators=200, learning_rate=0.05, random_state=42))
]

models_3 = [
    ('lr', LogisticRegression(C=1.5, max_iter=200)),
    ('knn', KNeighborsClassifier(n_neighbors=7)),
    ('svm', make_pipeline(StandardScaler(), SVC(probability=True, C=2.0, kernel='rbf'))),
    ('lda', LinearDiscriminantAnalysis()),
    ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)),
    ('ada', AdaBoostClassifier(n_estimators=200, learning_rate=0.05, random_state=42)),
    ('dt', DecisionTreeClassifier(max_depth=10, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
    ('gnb', GaussianNB())
]

# Function to get stacked predictions
def get_stacked_predictions(models, X_train, y_train, X_test):
    predictions_train = []
    predictions_test = []
    
    for name, model in models:
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        train_pred = model.predict_proba(X_train)[:, 1].reshape(-1, 1)
        test_pred = model.predict_proba(X_test)[:, 1].reshape(-1, 1)
        
        predictions_train.append(train_pred)
        predictions_test.append(test_pred)
    
    return np.column_stack(predictions_train), np.column_stack(predictions_test)

# Get predictions from each stacked model
train_pred_1, test_pred_1 = get_stacked_predictions(models_1, X_train, y_train, X_test)
train_pred_2, test_pred_2 = get_stacked_predictions(models_2, X_train, y_train, X_test)
train_pred_3, test_pred_3 = get_stacked_predictions(models_3, X_train, y_train, X_test)

# Concatenate stacked predictions
X_train_meta = np.column_stack([train_pred_1, train_pred_2, train_pred_3])
X_test_meta = np.column_stack([test_pred_1, test_pred_2, test_pred_3])

# Define Deep Neural Network as meta-model
def build_meta_model(input_shape):
    model = Sequential([
        Dense(256, input_shape=(input_shape,), kernel_initializer='he_normal'),
        LeakyReLU(),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, kernel_initializer='he_normal'),
        LeakyReLU(),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, kernel_initializer='he_normal'),
        LeakyReLU(),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Build and train meta-model
meta_model = build_meta_model(X_train_meta.shape[1])
history = meta_model.fit(X_train_meta, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=1)

# Meta-model predictions
y_pred_prob = meta_model.predict(X_test_meta).ravel()
y_pred_meta = (y_pred_prob > 0.6).astype(int)

# Evaluation
print("Final Stacked Model Accuracy:", accuracy_score(y_test, y_pred_meta))
print(classification_report(y_test, y_pred_meta))

# Additional metrics
auc = roc_auc_score(y_test, y_pred_prob)
cm = confusion_matrix(y_test, y_pred_meta)
print("ROC-AUC:", auc)
print("Confusion Matrix:\n", cm)

# Plot training history (loss curves)
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Define the path for the loss plot
loss_plot_path = os.path.join(os.path.dirname(__file__), 'data', 'metrics', 'training_loss.png')
os.makedirs(os.path.dirname(loss_plot_path), exist_ok=True)
plt.savefig(loss_plot_path)
print("Training loss plot saved to:", loss_plot_path)
plt.close()

# Generate ROC Curve plot
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")

roc_plot_path = os.path.join(os.path.dirname(__file__), 'data', 'metrics', 'roc_curve.png')
os.makedirs(os.path.dirname(roc_plot_path), exist_ok=True)
plt.savefig(roc_plot_path)
print("ROC curve plot saved to:", roc_plot_path)
plt.close()
