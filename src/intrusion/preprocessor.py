"""
Intrusion Detection Preprocessing Pipeline.
Handles NSL-KDD dataset cleaning, encoding, and preparation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.shared.preprocessor import (
    DataPreprocessor, remove_duplicates,
    check_nulls, get_class_distribution
)

# NSL-KDD column names (41 features + label + difficulty)
COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag',
    'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
    'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    'label', 'difficulty'
]

# Categorical columns that need encoding
CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']

# Attack type grouping (multi-class)
ATTACK_MAP = {
    'normal': 'normal',
    'back': 'dos', 'land': 'dos', 'neptune': 'dos',
    'pod': 'dos', 'smurf': 'dos', 'teardrop': 'dos',
    'mailbomb': 'dos', 'apache2': 'dos', 'processtable': 'dos',
    'udpstorm': 'dos',
    'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe',
    'satan': 'probe', 'mscan': 'probe', 'saint': 'probe',
    'ftp_write': 'r2l', 'guess_passwd': 'r2l', 'imap': 'r2l',
    'multihop': 'r2l', 'phf': 'r2l', 'spy': 'r2l',
    'warezclient': 'r2l', 'warezmaster': 'r2l', 'sendmail': 'r2l',
    'named': 'r2l', 'snmpgetattack': 'r2l', 'snmpguess': 'r2l',
    'xlock': 'r2l', 'xsnoop': 'r2l', 'httptunnel': 'r2l',
    'buffer_overflow': 'u2r', 'loadmodule': 'u2r',
    'perl': 'u2r', 'rootkit': 'u2r', 'ps': 'u2r',
    'sqlattack': 'u2r', 'xterm': 'u2r'
}


def load_intrusion_data(train_path: str, test_path: str):
    """Load NSL-KDD train and test files."""
    print("Loading NSL-KDD dataset...")

    df_train = pd.read_csv(train_path, header=None, names=COLUMNS)
    df_test = pd.read_csv(test_path, header=None, names=COLUMNS)

    print(f"Train shape: {df_train.shape}")
    print(f"Test shape:  {df_test.shape}")
    return df_train, df_test


def map_attack_types(df: pd.DataFrame) -> pd.DataFrame:
    """Map specific attack names to 5 attack categories."""
    df = df.copy()

    # Clean label (remove trailing dots if any)
    df['label'] = df['label'].str.strip().str.lower()
    df['label'] = df['label'].str.replace('.', '', regex=False)

    # Map to attack category
    df['attack_type'] = df['label'].map(ATTACK_MAP)

    # Handle unknown labels
    unknown = df['attack_type'].isna().sum()
    if unknown > 0:
        print(f"Warning: {unknown} unknown labels found, mapping to 'unknown'")
        df['attack_type'] = df['attack_type'].fillna('unknown')

    # Binary label: 0 = normal, 1 = attack
    df['is_attack'] = (df['attack_type'] != 'normal').astype(int)

    return df


def encode_categoricals(df_train: pd.DataFrame,
                         df_test: pd.DataFrame):
    """Label encode categorical columns."""
    encoders = {}

    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        # Fit on combined to handle unseen categories in test
        combined = pd.concat([df_train[col], df_test[col]])
        le.fit(combined)

        df_train[col] = le.transform(df_train[col])
        df_test[col] = le.transform(df_test[col])
        encoders[col] = le

    return df_train, df_test, encoders


def preprocess_intrusion(train_path: str, test_path: str):
    """
    Full preprocessing pipeline for intrusion detection.
    Returns X_train, X_test, y_train, y_test
    """
    print("\n=== INTRUSION DETECTION PREPROCESSING ===\n")

    # Load
    df_train, df_test = load_intrusion_data(train_path, test_path)

    # Map attack types
    print("\nStep 1: Mapping attack types...")
    df_train = map_attack_types(df_train)
    df_test = map_attack_types(df_test)

    print("\nTrain attack distribution:")
    print(df_train['attack_type'].value_counts())

    # Drop unused columns
    drop_cols = ['label', 'difficulty', 'attack_type']
    target = 'is_attack'

    # Encode categoricals
    print("\nStep 2: Encoding categorical features...")
    df_train, df_test, encoders = encode_categoricals(df_train, df_test)

    # Save encoders
    os.makedirs('models/intrusion', exist_ok=True)
    joblib.dump(encoders, 'models/intrusion/label_encoders.pkl')

    # Check nulls
    print("\nStep 3: Data quality check...")
    check_nulls(df_train)
    df_train = remove_duplicates(df_train)

    # Split features and target
    X_train = df_train.drop(columns=drop_cols + [target])
    y_train = df_train[target]

    X_test = df_test.drop(columns=drop_cols + [target])
    y_test = df_test[target]

    # Scale
    print("\nStep 4: Scaling features...")
    preprocessor = DataPreprocessor(domain='intrusion')
    scale_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    X_train_scaled = preprocessor.scale_features(
        pd.DataFrame(X_train, columns=X_train.columns),
        cols_to_scale=scale_cols, fit=True
    )
    X_test_scaled = preprocessor.scale_features(
        pd.DataFrame(X_test, columns=X_test.columns),
        cols_to_scale=scale_cols, fit=False
    )

    preprocessor.save_scaler('models/intrusion')

    print(f"\nFinal shapes:")
    print(f"X_train: {X_train_scaled.shape} | y_train: {y_train.shape}")
    print(f"X_test:  {X_test_scaled.shape}  | y_test:  {y_test.shape}")
    print(f"\nAttack rate - Train: {y_train.mean():.3f} | Test: {y_test.mean():.3f}")

    print("\n✅ Intrusion preprocessing complete.")
    return X_train_scaled, X_test_scaled, y_train, y_test, preprocessor