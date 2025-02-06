"""
preprocess.py - Data Preprocessing Script

This script is responsible for loading raw data, handling missing values, normalizing or encoding features,
and preparing the dataset for training.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Loads dataset from a CSV file."""
    return pd.read_csv(file_path)

def clean_data(df):
    """Handles missing values and drops unnecessary columns."""
    df.dropna(inplace=True)  # Remove missing values
    return df

def preprocess_features(df, target_column):
    """Splits dataset into features (X) and target (y), then scales numeric features."""
    X = df.drop(columns=[target_column])
    y = df[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Splits the dataset into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

if __name__ == "__main__":
    file_path = "data/dataset.csv"  # Modify with actual path
    df = load_data(file_path)
    df = clean_data(df)
    X, y = preprocess_features(df, target_column="label")
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print("Data preprocessing completed successfully!")
