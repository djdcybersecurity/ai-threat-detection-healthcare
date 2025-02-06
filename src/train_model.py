import pandas as pd  # Data manipulation
import numpy as np  # Numerical operations
import joblib  # Model saving and loading
from sklearn.ensemble import RandomForestClassifier  # Machine learning model
from sklearn.metrics import accuracy_score, classification_report  # Model evaluation metrics
from preprocess import preprocess_pipeline  # Import preprocessing functions

def train_model(X_train, y_train, model_type="random_forest"):
    """
    Train a machine learning model on the given dataset.
    :param X_train: Training feature set
    :param y_train: Training target labels
    :param model_type: str, type of model to train (default: random_forest)
    :return: Trained model
    """
    if model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError("Unsupported model type")
    
    model.fit(X_train, y_train)  # Train the model
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model using accuracy and classification report.
    :param model: Trained model
    :param X_test: Test feature set
    :param y_test: Test target labels
    :return: Accuracy score and classification report
    """
    y_pred = model.predict(X_test)  # Make predictions
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
    report = classification_report(y_test, y_pred)  # Generate classification report
    return accuracy, report

def save_model(model, file_path="model.pkl"):
    """
    Save the trained model to a file.
    :param model: Trained model
    :param file_path: str, path to save the model
    """
    joblib.dump(model, file_path)

def load_model(file_path="model.pkl"):
    """
    Load a trained model from a file.
    :param file_path: str, path to the saved model
    :return: Loaded model
    """
    return joblib.load(file_path)

if __name__ == "__main__":
    # Define dataset path and target variable
    file_path = "data/dataset.csv"
    target_column = "label"
    
    # Preprocess the data
    X_train, X_test, y_train, y_test, encoders, scaler = preprocess_pipeline(file_path, target_column)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    accuracy, report = evaluate_model(model, X_test, y_test)
    print(f"Model Accuracy: {accuracy}\n")
    print(f"Classification Report:\n{report}")
    
    # Save the trained model
    save_model(model, "trained_model.pkl")
