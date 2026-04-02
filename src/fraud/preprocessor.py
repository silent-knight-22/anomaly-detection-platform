"""
Fraud-specific preprocessing pipeline.
Handles scaling, SMOTE, and data preparation for fraud detection.
"""

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import sys
import os

# Allow imports from src/shared
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.shared.preprocessor import (
    DataPreprocessor, remove_duplicates,
    check_nulls, get_class_distribution
)


def load_fraud_data(filepath: str) -> pd.DataFrame:
    """Load raw fraud CSV file."""
    print(f"Loading fraud data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"Shape: {df.shape}")
    return df


def preprocess_fraud(df: pd.DataFrame):
    """
    Full preprocessing pipeline for fraud detection.
    Returns X_train, X_test, y_train, y_test (after SMOTE on train)
    """

    print("\n=== FRAUD PREPROCESSING PIPELINE ===\n")

    # Step 1: Basic checks
    print("Step 1: Data Quality Checks")
    check_nulls(df)
    df = remove_duplicates(df)
    get_class_distribution(df, 'Class')

    # Step 2: Drop Time column (not useful as raw seconds)
    # Keep Amount — it's a real feature
    df = df.drop(columns=['Time'])
    print("\nStep 2: Dropped 'Time' column")

    # Step 3: Scale 'Amount' — V1-V28 are already PCA scaled
    print("\nStep 3: Scaling 'Amount' column")
    preprocessor = DataPreprocessor(domain='fraud')
    df = preprocessor.scale_features(df, cols_to_scale=['Amount'], fit=True)

    # Step 4: Train/test split
    print("\nStep 4: Train/Test Split")
    X_train, X_test, y_train, y_test = preprocessor.split_data(
        df, target_col='Class', test_size=0.2
    )

    # Step 5: Apply SMOTE only on training data
    print("\nStep 5: Applying SMOTE to training data")
    print(f"Before SMOTE — Fraud: {y_train.sum():,} | Non-fraud: {(y_train==0).sum():,}")

    smote = SMOTE(random_state=42, sampling_strategy=0.1)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print(f"After SMOTE  — Fraud: {y_train_res.sum():,} | Non-fraud: {(y_train_res==0).sum():,}")
    print(f"New train size: {len(X_train_res):,}")

    # Step 6: Save scaler
    preprocessor.save_scaler('models/fraud')

    print("\n✅ Preprocessing complete.")
    return X_train_res, X_test, y_train_res, y_test, preprocessor