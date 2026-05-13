"""
Fraud-specific preprocessing pipeline.

Supports both the original Kaggle credit-card CSV and the IEEE-CIS fraud
dataset now stored under data/raw/fraud.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.shared.preprocessor import (  # noqa: E402
    check_nulls,
    get_class_distribution,
    remove_duplicates,
)


FRAUD_DATA_DIR = Path("data/raw/fraud")
MODEL_DIR = Path("models/fraud")
PREPROCESSOR_PATH = MODEL_DIR / "fraud_preprocessor.pkl"

LEGACY_TARGET = "Class"
IEEE_TARGET = "isFraud"
TARGET_COLUMNS = (IEEE_TARGET, LEGACY_TARGET)


def _read_csv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    print(f"Loading fraud data from: {path}")
    df = pd.read_csv(path, nrows=nrows, low_memory=False)
    print(f"Shape: {df.shape}")
    return df


def _find_legacy_creditcard_file(data_dir: Path) -> Path | None:
    for file in data_dir.glob("*.csv"):
        if file.name.lower() == "creditcard.csv":
            return file
    return None


def load_fraud_data(
    filepath: str | os.PathLike[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """
    Load fraud data.

    If the IEEE-CIS files are present, train_transaction.csv is left-joined
    with train_identity.csv on TransactionID. If a direct CSV path is supplied,
    that file is loaded as-is.
    """
    path = Path(filepath) if filepath else FRAUD_DATA_DIR

    if path.is_file():
        return _read_csv(path, nrows=nrows)

    data_dir = path
    transaction_path = data_dir / "train_transaction.csv"
    identity_path = data_dir / "train_identity.csv"

    if transaction_path.exists():
        transaction_df = _read_csv(transaction_path, nrows=nrows)
        if identity_path.exists():
            identity_df = _read_csv(identity_path, nrows=nrows)
            df = transaction_df.merge(identity_df, on="TransactionID", how="left")
            print(f"Merged IEEE transaction + identity shape: {df.shape}")
            return df
        return transaction_df

    legacy_file = _find_legacy_creditcard_file(data_dir)
    if legacy_file:
        return _read_csv(legacy_file, nrows=nrows)

    raise FileNotFoundError(
        "No supported fraud dataset found. Expected IEEE files "
        "train_transaction.csv/train_identity.csv or creditcard.csv."
    )


def detect_target_column(df: pd.DataFrame) -> str:
    for target_col in TARGET_COLUMNS:
        if target_col in df.columns:
            return target_col
    raise ValueError(
        f"Could not find fraud target column. Expected one of: {TARGET_COLUMNS}"
    )


def _drop_non_predictive_columns(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    drop_cols = [target_col]
    if target_col == LEGACY_TARGET and "Time" in df.columns:
        drop_cols.append("Time")
    if "TransactionID" in df.columns:
        drop_cols.append("TransactionID")
    return df.drop(columns=drop_cols, errors="ignore")


def build_fraud_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, dict[str, Any]]:
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=10,
                    sparse_output=True,
                ),
            ),
        ]
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False,
    )
    metadata = {
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "raw_feature_columns": X.columns.tolist(),
    }
    return preprocessor, metadata


def save_fraud_preprocessor(
    preprocessor: ColumnTransformer,
    metadata: dict[str, Any],
    save_dir: str | os.PathLike[str] = MODEL_DIR,
) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / "fraud_preprocessor.pkl"
    joblib.dump({"preprocessor": preprocessor, "metadata": metadata}, path)
    print(f"Fraud preprocessor saved to: {path}")


def load_fraud_preprocessor(
    model_dir: str | os.PathLike[str] = MODEL_DIR,
) -> dict[str, Any]:
    return joblib.load(Path(model_dir) / "fraud_preprocessor.pkl")


def get_transformed_feature_names(bundle: dict[str, Any]) -> list[str]:
    preprocessor = bundle["preprocessor"]
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        return bundle["metadata"].get("raw_feature_columns", [])


def preprocess_fraud(
    df: pd.DataFrame,
    test_size: float = 0.2,
    apply_smote: bool = False,
):
    """
    Full preprocessing pipeline for fraud detection.

    Returns transformed X_train, X_test, y_train, y_test, and the saved
    preprocessor bundle metadata. SMOTE is off by default for IEEE because the
    transformed feature matrix is large and XGBoost handles imbalance with
    scale_pos_weight.
    """
    print("\n=== FRAUD PREPROCESSING PIPELINE ===\n")

    target_col = detect_target_column(df)
    print(f"Detected target column: {target_col}")

    print("Step 1: Data Quality Checks")
    check_nulls(df)
    df = remove_duplicates(df)
    get_class_distribution(df, target_col)

    y = df[target_col].astype(int)
    X = _drop_non_predictive_columns(df, target_col)

    print("\nStep 2: Train/Test Split")
    stratify = y if y.nunique() > 1 else None
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )
    print(f"Train size: {X_train_raw.shape[0]:,} | Test size: {X_test_raw.shape[0]:,}")
    print(f"Train fraud rate: {y_train.mean():.4f} | Test fraud rate: {y_test.mean():.4f}")

    print("\nStep 3: Fitting fraud preprocessing pipeline")
    preprocessor, metadata = build_fraud_preprocessor(X_train_raw)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    metadata.update(
        {
            "target_column": target_col,
            "dataset_format": "ieee_cis" if target_col == IEEE_TARGET else "creditcard_legacy",
            "transformed_feature_count": X_train.shape[1],
        }
    )

    if apply_smote:
        print("\nStep 4: Applying SMOTE to training data")
        fraud_count = int(y_train.sum())
        non_fraud_count = int(len(y_train) - fraud_count)
        print(f"Before SMOTE - Fraud: {fraud_count:,} | Non-fraud: {non_fraud_count:,}")
        if fraud_count and fraud_count < non_fraud_count:
            smote = SMOTE(random_state=42, sampling_strategy=0.1)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"After SMOTE - Fraud: {int(y_train.sum()):,} | Non-fraud: {int((y_train == 0).sum()):,}")
        else:
            print("Data already balanced or has no positive class - skipping SMOTE")
    else:
        print("\nStep 4: Skipping SMOTE; class imbalance handled in model training")

    save_fraud_preprocessor(preprocessor, metadata, MODEL_DIR)

    if sparse.issparse(X_train):
        print(f"Transformed train matrix: {X_train.shape} sparse")
    else:
        print(f"Transformed train matrix: {X_train.shape} dense")

    print("\nPreprocessing complete.")
    return X_train, X_test, y_train, y_test, {"preprocessor": preprocessor, "metadata": metadata}


def transform_fraud_records(
    records: pd.DataFrame | dict[str, Any] | list[dict[str, Any]],
    model_dir: str | os.PathLike[str] = MODEL_DIR,
):
    """Transform raw IEEE/legacy fraud records with the fitted preprocessor."""
    bundle = load_fraud_preprocessor(model_dir)
    metadata = bundle["metadata"]
    preprocessor = bundle["preprocessor"]

    if isinstance(records, pd.DataFrame):
        df = records.copy()
    elif isinstance(records, dict):
        df = pd.DataFrame([records])
    else:
        df = pd.DataFrame(records)

    df = df.drop(columns=list(TARGET_COLUMNS), errors="ignore")
    df = df.drop(columns=["TransactionID"], errors="ignore")

    raw_feature_columns = metadata["raw_feature_columns"]
    for col in raw_feature_columns:
        if col not in df.columns:
            df[col] = np.nan
    df = df[raw_feature_columns]
    return preprocessor.transform(df)
