import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# =========================================================
# DATASET CONFIGURATION
# =========================================================

FEATURE_COLUMNS = [
    "temperature_c",
    "vibration_mm_s",
    "pressure_kpa",
]

TARGET_COLUMNS = [
    "overheat_alert",
    "fault_alert",
]

ALL_REQUIRED_COLUMNS = FEATURE_COLUMNS + TARGET_COLUMNS

RANDOM_STATE = 42


# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset(file):
    """
    Load a CSV dataset and validate its columns and values.

    Accepts:
        - CSV file path
        - Streamlit uploaded file
        - Pandas DataFrame

    Returns:
        Validated DataFrame
    """

    if isinstance(file, pd.DataFrame):
        df = file.copy()

    else:
        df = pd.read_csv(file)

    # Remove unnecessary whitespace from column names
    df.columns = df.columns.str.strip()

    # Check required columns
    missing_columns = [
        column
        for column in ALL_REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Select required features and targets
    # machine_id is intentionally excluded
    df = df[ALL_REQUIRED_COLUMNS].copy()

    # Convert columns to numeric
    for column in ALL_REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Check missing values
    if df.isnull().any().any():

        missing_counts = df.isnull().sum()

        missing_info = missing_counts[
            missing_counts > 0
        ].to_dict()

        raise ValueError(
            f"Dataset contains missing or invalid values: "
            f"{missing_info}"
        )

    # Check infinite values
    if not np.isfinite(df.to_numpy()).all():

        raise ValueError(
            "Dataset contains infinite values."
        )

    # Validate binary targets
    for target in TARGET_COLUMNS:

        unique_values = set(
            df[target].unique()
        )

        if not unique_values.issubset({0, 1}):

            raise ValueError(
                f"Target '{target}' must contain only 0 and 1."
            )

    # Convert target columns to integer
    for target in TARGET_COLUMNS:

        df[target] = df[target].astype(int)

    # Check dataset size
    if len(df) < 20:

        raise ValueError(
            "Dataset must contain at least 20 observations "
            "for training and evaluation."
        )

    return df


# =========================================================
# DATASET INFORMATION
# =========================================================

def get_dataset_summary(df):
    """
    Generate basic information about the dataset.
    """

    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "features": FEATURE_COLUMNS,
        "targets": TARGET_COLUMNS,
        "missing_values": int(
            df.isnull().sum().sum()
        ),
    }

    return summary


# =========================================================
# TARGET DISTRIBUTION
# =========================================================

def get_target_distribution(df):
    """
    Return the frequency of each target class.

    Also calculates the number of observations
    where both alerts are active.
    """

    distribution = {}

    for target in TARGET_COLUMNS:

        counts = df[target].value_counts().sort_index()

        distribution[target] = {
            "0": int(counts.get(0, 0)),
            "1": int(counts.get(1, 0)),
        }

    both_alerts = int(
        (
            (df["overheat_alert"] == 1)
            &
            (df["fault_alert"] == 1)
        ).sum()
    )

    distribution["both_alerts_active"] = both_alerts

    return distribution


# =========================================================
# TRAIN / VALIDATION / TEST SPLIT
# =========================================================

def split_dataset(df, random_state=RANDOM_STATE):
    """
    Split dataset into:

        Training:   70%
        Validation: 15%
        Testing:    15%

    Uses reproducible random splitting.

    For 180 observations:
        Train = 126
        Validation = 27
        Test = 27
    """

    # Separate features and targets
    X = df[FEATURE_COLUMNS].copy()

    y = df[TARGET_COLUMNS].copy()

    total_samples = len(df)

    # Calculate exact split sizes
    test_size = int(round(total_samples * 0.15))

    validation_size = int(
        round(total_samples * 0.15)
    )

    # First split: training + validation / testing
    X_train_val, X_test, y_train_val, y_test = (
        train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )
    )

    # Second split: training / validation
    X_train, X_val, y_train, y_val = (
        train_test_split(
            X_train_val,
            y_train_val,
            test_size=validation_size,
            random_state=random_state,
            shuffle=True,
        )
    )

    return {
        "X_train": X_train.reset_index(drop=True),
        "X_val": X_val.reset_index(drop=True),
        "X_test": X_test.reset_index(drop=True),

        "y_train": y_train.reset_index(drop=True),
        "y_val": y_val.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
    }


# =========================================================
# FEATURE STANDARDIZATION
# =========================================================

def standardize_data(
    X_train,
    X_val,
    X_test,
):
    """
    Standardize features using:

        z = (x - mean) / standard_deviation

    IMPORTANT:
    Mean and standard deviation are calculated
    using training data only.

    Validation and test data use the same
    training statistics.
    """

    # Calculate statistics using training data
    mean = X_train.mean()

    std = X_train.std(ddof=0)

    # Avoid division by zero
    std = std.replace(0, 1)

    # Standardize datasets
    X_train_scaled = (
        X_train - mean
    ) / std

    X_val_scaled = (
        X_val - mean
    ) / std

    X_test_scaled = (
        X_test - mean
    ) / std

    return {
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,

        "mean": mean,
        "std": std,
    }


# =========================================================
# COMPLETE DATA PREPARATION
# =========================================================

def prepare_dataset(
    file,
    random_state=RANDOM_STATE,
):
    """
    Complete dataset preparation pipeline.

    Steps:
        1. Load CSV
        2. Validate data
        3. Split data
        4. Standardize features
        5. Return all prepared datasets
    """

    # Step 1: Load and validate
    df = load_dataset(file)

    # Step 2: Split
    splits = split_dataset(
        df,
        random_state=random_state,
    )

    # Step 3: Standardize
    scaled = standardize_data(
        splits["X_train"],
        splits["X_val"],
        splits["X_test"],
    )

    # Step 4: Combine results
    result = {
        "raw_data": df,

        **splits,

        **scaled,

        "summary": get_dataset_summary(df),

        "target_distribution": get_target_distribution(df),

        "random_state": random_state,
    }

    return result


# =========================================================
# MANUAL INPUT STANDARDIZATION
# =========================================================

def scale_manual_input(
    temperature_c,
    vibration_mm_s,
    pressure_kpa,
    mean,
    std,
):
    """
    Standardize manually entered sensor values
    using training-set statistics.
    """

    manual_data = pd.DataFrame(
        [[
            temperature_c,
            vibration_mm_s,
            pressure_kpa,
        ]],
        columns=FEATURE_COLUMNS,
    )

    scaled_data = (
        manual_data - mean
    ) / std

    return scaled_data.to_numpy(
        dtype=np.float32
    )