"""
Shared Preprocessing Module
Used by both fraud and intrusion detection pipelines.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os


class DataPreprocessor:
    """
    Handles all preprocessing steps shared across domains.
    - Scaling numerical features
    - Train/test splitting
    - Saving and loading scalers
    """

    def __init__(self, domain: str):
        """
        domain: 'fraud' or 'intrusion'
        """
        self.domain = domain
        self.scaler = StandardScaler()
        self.is_fitted = False

    def scale_features(self, df: pd.DataFrame,
                       cols_to_scale: list,
                       fit: bool = True) -> pd.DataFrame:
        """
        Scale specified columns using StandardScaler.
        fit=True during training, fit=False during inference.
        """
        df = df.copy()
        if fit:
            df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Scaler not fitted yet. Run with fit=True first.")
            df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
        return df

    def split_data(self, df: pd.DataFrame,
                   target_col: str,
                   test_size: float = 0.2,
                   random_state: int = 42):
        """
        Split dataframe into train/test sets.
        Returns X_train, X_test, y_train, y_test
        """
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y  # preserves class ratio in both splits
        )

        print(f"Train size: {X_train.shape[0]:,} | Test size: {X_test.shape[0]:,}")
        print(f"Train fraud rate: {y_train.mean():.4f} | Test fraud rate: {y_test.mean():.4f}")

        return X_train, X_test, y_train, y_test

    def save_scaler(self, save_dir: str):
        """Save fitted scaler to disk for later use in API."""
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f'{self.domain}_scaler.pkl')
        joblib.dump(self.scaler, path)
        print(f"Scaler saved to: {path}")

    def load_scaler(self, save_dir: str):
        """Load a previously saved scaler."""
        path = os.path.join(save_dir, f'{self.domain}_scaler.pkl')
        self.scaler = joblib.load(path)
        self.is_fitted = True
        print(f"Scaler loaded from: {path}")


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows and report how many were removed."""
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    removed = before - after
    if removed > 0:
        print(f"Removed {removed:,} duplicate rows ({removed/before*100:.2f}%)")
    else:
        print("No duplicate rows found.")
    return df


def check_nulls(df: pd.DataFrame) -> None:
    """Print null value counts for each column."""
    nulls = df.isnull().sum()
    if nulls.sum() == 0:
        print("No null values found. Dataset is clean.")
    else:
        print("Null values found:")
        print(nulls[nulls > 0])


def get_class_distribution(df: pd.DataFrame, target_col: str) -> None:
    """Print class distribution summary."""
    counts = df[target_col].value_counts()
    pcts = df[target_col].value_counts(normalize=True) * 100
    print("Class Distribution:")
    for cls in counts.index:
        print(f"  Class {cls}: {counts[cls]:,} ({pcts[cls]:.3f}%)")
    print(f"  Imbalance ratio: {counts.max()//counts.min()}:1")
