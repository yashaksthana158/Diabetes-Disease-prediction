"""
preprocessing.py
Handles data loading, zero-value imputation, scaling, and train/test splitting.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Features where 0 is physiologically impossible and indicates missing data
ZERO_AS_NULL = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

FEATURE_COLS = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]
TARGET_COL = 'Outcome'

RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_data(filepath: str) -> pd.DataFrame:
    """Load the raw CSV dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'.\n"
            "Download it from https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database "
            "and place it at data/diabetes.csv"
        )
    return pd.read_csv(filepath)


def impute_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """Replace biologically impossible zeros with median-imputed values."""
    df = df.copy()
    df[ZERO_AS_NULL] = df[ZERO_AS_NULL].replace(0, np.nan)
    df[ZERO_AS_NULL] = df[ZERO_AS_NULL].fillna(df[ZERO_AS_NULL].median())
    return df


def split_features_target(df: pd.DataFrame):
    """Separate feature matrix X from target vector y."""
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y


def get_train_test_split(X, y):
    """Stratified 80/20 train/test split."""
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


def scale_features(X_train, X_test, scaler_save_path: str = None):
    """
    Fit StandardScaler on training data and transform both splits.
    Optionally saves the fitted scaler for use during inference.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if scaler_save_path:
        os.makedirs(os.path.dirname(scaler_save_path), exist_ok=True)
        joblib.dump(scaler, scaler_save_path)
        print(f"Scaler saved to {scaler_save_path}")

    return X_train_scaled, X_test_scaled, scaler


def run_pipeline(data_path: str, scaler_save_path: str = "models/scaler.pkl"):
    """End-to-end preprocessing: load → impute → split → scale."""
    df = load_data(data_path)
    df = impute_zeros(df)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test, scaler_save_path)

    print(f"Dataset loaded: {df.shape[0]} rows, {len(FEATURE_COLS)} features")
    print(f"Train size: {X_train_sc.shape[0]} | Test size: {X_test_sc.shape[0]}")
    print(f"Class balance (train) — 0: {(y_train == 0).sum()}, 1: {(y_train == 1).sum()}")

    return X_train_sc, X_test_sc, y_train, y_test, scaler


if __name__ == "__main__":
    run_pipeline("data/diabetes.csv")
