"""
==============================================================================
AGENT 1: Data & Preprocessing Agent
==============================================================================
Role   : Data Processing Expert
Purpose: Load, validate, preprocess, and split the Breast Cancer Wisconsin
         dataset so that downstream agents receive clean, scaled data.

INPUT  : None (loads directly from sklearn)
OUTPUT : dict with keys:
    - X_train        : np.ndarray  — Scaled training features
    - X_test         : np.ndarray  — Scaled test features
    - y_train        : np.ndarray  — Training labels (0=Malignant, 1=Benign)
    - y_test         : np.ndarray  — Test labels
    - feature_names  : list[str]   — 30 feature column names
    - scaler         : StandardScaler — Fitted scaler for reuse in deployment
    - raw_df         : pd.DataFrame  — Full unscaled DataFrame with diagnosis
    - label_map      : dict        — {0: "Malignant", 1: "Benign"}
==============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Label mapping (sklearn uses 0=Malignant, 1=Benign)
# ---------------------------------------------------------------------------
LABEL_MAP = {0: "Malignant", 1: "Benign"}
RANDOM_STATE = 42          # Fixed seed for reproducibility across all agents
TEST_SIZE = 0.20           # 80/20 split as specified


def run() -> dict:
    """
    Execute the full data-preprocessing pipeline.

    Returns
    -------
    dict
        A structured output containing train/test splits, fitted scaler,
        raw dataframe, and metadata — ready for Agent 2 and Agent 3.
    """

    # ------------------------------------------------------------------
    # STEP 1 — Load the dataset
    # ------------------------------------------------------------------
    # sklearn bundles the Breast Cancer Wisconsin (Diagnostic) dataset:
    #   569 samples, 30 numeric features, 2 classes.
    data = load_breast_cancer()
    print("=" * 70)
    print("AGENT 1: Data & Preprocessing")
    print("=" * 70)

    # ------------------------------------------------------------------
    # STEP 2 — Build a well-structured DataFrame
    # ------------------------------------------------------------------
    # Using data.feature_names gives us human-readable column headers
    # (e.g. 'mean radius', 'worst concavity') instead of generic x0…x29.
    df = pd.DataFrame(data.data, columns=data.feature_names)

    # Add numeric target column (0 or 1)
    df["target"] = data.target

    # Add human-readable diagnosis column for EDA
    df["diagnosis"] = df["target"].map(LABEL_MAP)

    print(f"\n✅ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")
    print(f"   Features: {len(data.feature_names)}")
    print(f"   Classes : {list(LABEL_MAP.values())}")

    # ------------------------------------------------------------------
    # STEP 3 — Data validation
    # ------------------------------------------------------------------
    print("\n--- Data Validation ---")

    # 3a. Missing values
    # WHY: Missing values would break StandardScaler and model training.
    missing = df.isnull().sum().sum()
    print(f"Missing values : {missing}")
    if missing > 0:
        print("⚠️  WARNING: Missing values detected! Filling with column median.")
        df.fillna(df.median(numeric_only=True), inplace=True)
    else:
        print("   ✓ No missing values — dataset is complete.")

    # 3b. Data types
    # WHY: All features must be numeric for ML algorithms.
    print(f"\nData types:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"   {dtype}: {count} columns")

    # 3c. Basic statistics
    print(f"\nDataset shape  : {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head().to_string())

    # 3d. Class distribution
    print(f"\nClass distribution:")
    class_counts = df["diagnosis"].value_counts()
    for label, count in class_counts.items():
        pct = count / len(df) * 100
        print(f"   {label}: {count} ({pct:.1f}%)")

    # ------------------------------------------------------------------
    # STEP 4 — Feature scaling (StandardScaler)
    # ------------------------------------------------------------------
    # WHY: StandardScaler (z-score normalization) is essential because:
    #   - SVM and KNN rely on distance calculations; features with larger
    #     ranges (e.g. 'mean area' ~ 100–2500) would dominate features
    #     with smaller ranges (e.g. 'mean smoothness' ~ 0.05–0.16).
    #   - Logistic Regression converges faster with scaled features.
    #   - StandardScaler transforms each feature to mean=0, std=1.
    X = df[data.feature_names].values   # Shape: (569, 30)
    y = df["target"].values             # Shape: (569,)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\n✅ Feature scaling applied (StandardScaler)")
    print(f"   Before scaling — mean: {X.mean():.4f}, std: {X.std():.4f}")
    print(f"   After scaling  — mean: {X_scaled.mean():.4f}, std: {X_scaled.std():.4f}")

    # ------------------------------------------------------------------
    # STEP 5 — Train/test split
    # ------------------------------------------------------------------
    # WHY: 80/20 is a standard split that balances having enough training
    #   data with a meaningful test set. random_state ensures that every
    #   run produces identical splits for reproducibility.
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y   # Preserve class proportions in both splits
    )

    print(f"\n✅ Train/test split (80/20, stratified, random_state={RANDOM_STATE})")
    print(f"   Training set : {X_train.shape[0]} samples")
    print(f"   Test set     : {X_test.shape[0]} samples")

    # ------------------------------------------------------------------
    # STEP 6 — Package output for downstream agents
    # ------------------------------------------------------------------
    output = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": list(data.feature_names),
        "scaler": scaler,
        "raw_df": df,
        "label_map": LABEL_MAP,
    }

    print(f"\n✅ Agent 1 complete. Output keys: {list(output.keys())}")
    print("=" * 70)
    return output


# ---------------------------------------------------------------------------
# Allow standalone execution for testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = run()
    print(f"\nStandalone test — output type: {type(result)}")
    print(f"X_train shape: {result['X_train'].shape}")
    print(f"X_test shape : {result['X_test'].shape}")
