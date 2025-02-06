import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical computing
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Feature scaling and encoding
from sklearn.model_selection import train_test_split  # Splitting dataset into training and testing

def load_data(file_path):
    """
    Load dataset from a CSV file.
    :param file_path: str, path to the dataset file
    :return: Pandas DataFrame
    """
    return pd.read_csv(file_path)

def handle_missing_values(df):
    """
    Handle missing values in the dataset by filling or dropping.
    :param df: Pandas DataFrame
    :return: DataFrame with handled missing values
    """
    df.fillna(df.median(), inplace=True)  # Fill missing values with median
    return df

def encode_categorical_columns(df):
    """
    Convert categorical features to numerical values using Label Encoding.
    :param df: Pandas DataFrame
    :return: DataFrame with encoded categorical columns
    """
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Store encoders for future use
    return df, label_encoders

def scale_features(df):
    """
    Normalize numerical features using StandardScaler.
    :param df: Pandas DataFrame
    :return: Scaled DataFrame
    """
    scaler = StandardScaler()
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df, scaler

def split_data(df, target_column, test_size=0.2, random_state=42):
    """
    Split dataset into training and testing sets.
    :param df: Pandas DataFrame
    :param target_column: str, name of the target column
    :param test_size: float, proportion of data to use for testing
    :param random_state: int, random seed for reproducibility
    :return: X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_column])  # Features
    y = df[target_column]  # Target variable
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def preprocess_pipeline(file_path, target_column):
    """
    Full preprocessing pipeline: Load, clean, encode, scale, and split data.
    :param file_path: str, path to the dataset file
    :param target_column: str, name of the target variable
    :return: X_train, X_test, y_train, y_test, encoders, scaler
    """
    df = load_data(file_path)  # Load data
    df = handle_missing_values(df)  # Handle missing values
    df, encoders = encode_categorical_columns(df)  # Encode categorical data
    df, scaler = scale_features(df)  # Scale numerical features
    X_train, X_test, y_train, y_test = split_data(df, target_column)  # Split dataset
    return X_train, X_test, y_train, y_test, encoders, scaler

if __name__ == "__main__":
    # Example usage of the preprocessing pipeline
    file_path = "data/dataset.csv"  # Path to the dataset
    target_column = "label"  # Define target variable
    X_train, X_test, y_train, y_test, encoders, scaler = preprocess_pipeline(file_path, target_column)
    print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")
