# src/data/data_loader.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path
from src.utils.config import *

def load_and_prepare_data():
    if not OUTPUT_CSV_DIR.exists():
        print(f"Error: The feature file was not found at {OUTPUT_CSV_DIR}")
        print("Please run 'src/data/feature_extractor.py' first.")
        return None
    
    df = pd.read_csv(OUTPUT_CSV_DIR)
    
    df = df.drop('filename', axis=1)
    
    X = df.drop('genre', axis=1).values
    y_labels = df['genre'].values
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_labels)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Data loaded, split, and scaled successfully.")
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, label_encoder

def load_data_with_validation():
    if not OUTPUT_CSV_DIR.exists():
        print(f"Error: Feature file not found at {OUTPUT_CSV_DIR}")
        return None
        
    df = pd.read_csv(OUTPUT_CSV_DIR).drop('filename', axis=1)
    X = df.drop('genre', axis=1).values
    y_labels = df['genre'].values
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_labels)
    
    # First split: separate a test set (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Second split: create train and validation sets from the remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    print("Data loaded with train, validation, and test sets.")
    
    loader = label_encoder
    
    return X_train, X_val, X_test, y_train, y_val, y_test, loader

if __name__ == '__main__':
    data = load_and_prepare_data()
    if data:
        print("\nData loader test successful!")