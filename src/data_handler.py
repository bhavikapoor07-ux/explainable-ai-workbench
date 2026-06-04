# =============================================================================
# data_handler.py
# Explainable AI Workbench — Data Ingestion & Understanding Module
# =============================================================================

import pandas as pd
import numpy as np
from pathlib import Path


# =============================================================================
# FUNCTION 1 — Load Data
# Accepts the uploaded file from Streamlit and returns a clean DataFrame
# =============================================================================

def load_data(uploaded_file):
    """
    Load a CSV file uploaded via Streamlit into a Pandas DataFrame.

    Parameters:
        uploaded_file : Streamlit UploadedFile object

    Returns:
        df            : pd.DataFrame on success
        error_message : str on failure (None if successful)
    """
    try:
        df = pd.read_csv(uploaded_file)

        # Basic sanity check — must have at least 2 columns and 10 rows
        if df.shape[0] < 10:
            return None, "Dataset too small. Please upload a file with at least 10 rows."
        if df.shape[1] < 2:
            return None, "Dataset must have at least 2 columns (features + target)."

        # Strip whitespace from column names
        df.columns = df.columns.str.strip()

        return df, None

    except Exception as e:
        return None, f"Failed to load file: {str(e)}"


# =============================================================================
# FUNCTION 2 — Load Demo Dataset
# Returns one of the built-in demo datasets
# =============================================================================

def load_demo_dataset(demo_type="regression"):
    """
    Load a built-in demo dataset for testing.

    Parameters:
        demo_type : "regression" or "classification"

    Returns:
        df        : pd.DataFrame
        name      : str — name of the demo dataset
    """
    if demo_type == "regression":
        # Student Performance Demo Dataset
        np.random.seed(42)
        n = 300
        study_hours = np.random.uniform(1, 10, n)
        attendance = np.random.uniform(50, 100, n)
        sleep_hours = np.random.uniform(4, 10, n)
        stress_level = np.random.uniform(1, 10, n)
        internet_usage = np.random.uniform(1, 8, n)

        # Target: exam score with realistic non-linear relationships
        score = (
            5 * np.sqrt(study_hours)
            + 0.3 * attendance
            + 0.5 * sleep_hours
            - 0.4 * stress_level
            - 0.2 * internet_usage
            + np.random.normal(0, 2, n)
        )
        score = np.clip(score, 0, 100)

        df = pd.DataFrame({
            "study_hours": np.round(study_hours, 2),
            "attendance": np.round(attendance, 2),
            "sleep_hours": np.round(sleep_hours, 2),
            "stress_level": np.round(stress_level, 2),
            "internet_usage": np.round(internet_usage, 2),
            "exam_score": np.round(score, 2)
        })
        return df, "Student Performance (Regression Demo)"

    elif demo_type == "classification":
        # Loan Approval Demo Dataset
        np.random.seed(42)
        n = 300
        income = np.random.uniform(20000, 120000, n)
        credit_score = np.random.uniform(300, 850, n)
        loan_amount = np.random.uniform(5000, 50000, n)
        employment_years = np.random.uniform(0, 20, n)
        debt_ratio = np.random.uniform(0.1, 0.9, n)

        # Target: loan approved (1) or rejected (0)
        score = (
            0.00003 * income
            + 0.005 * credit_score
            - 0.00001 * loan_amount
            + 0.05 * employment_years
            - 2 * debt_ratio
            - 1.5
        )
        prob = 1 / (1 + np.exp(-score))
        approved = (prob > 0.5).astype(int)

        df = pd.DataFrame({
            "income": np.round(income, 2),
            "credit_score": np.round(credit_score, 0).astype(int),
            "loan_amount": np.round(loan_amount, 2),
            "employment_years": np.round(employment_years, 1),
            "debt_ratio": np.round(debt_ratio, 3),
            "loan_approved": approved
        })
        return df, "Loan Approval (Classification Demo)"


# =============================================================================
# FUNCTION 3 — Validate Data
# Checks for data quality issues and returns a structured report
# =============================================================================

def validate_data(df):
    """
    Validate the DataFrame for common data quality issues.

    Parameters:
        df      : pd.DataFrame

    Returns:
        report  : dict with validation results and warnings
    """
    report = {
        "is_valid": True,
        "warnings": [],
        "info": []
    }

    # Check for missing values
    missing = df.isnull().sum()
    total_missing = missing.sum()
    if total_missing > 0:
        missing_cols = missing[missing > 0].to_dict()
        report["warnings"].append(
            f"Missing values detected in {len(missing_cols)} column(s): {missing_cols}. "
            f"These will be filled with median values automatically."
        )

    # Check for duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        report["warnings"].append(
            f"{duplicate_count} duplicate rows detected. These will be removed automatically."
        )

    # Check for very small dataset
    if df.shape[0] < 50:
        report["warnings"].append(
            f"Small dataset ({df.shape[0]} rows). Model performance may be limited."
        )

    # Check for non-numeric columns (that aren't the target)
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        to_encode = [c for c in non_numeric if df[c].nunique() <= 10]
        to_drop = [c for c in non_numeric if df[c].nunique() > 10]
        if to_encode:
            report["info"].append(
                f"Categorical columns with ≤10 unique values will be encoded automatically: {to_encode}."
            )
        if to_drop:
            report["info"].append(
                f"High-cardinality columns with >10 unique values will be excluded (too many categories to encode meaningfully): {to_drop}."
            )

    # Info about dataset size
    report["info"].append(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")

    return report


# =============================================================================
# FUNCTION 4 — Detect Problem Type
# Auto-detects Regression vs Classification from the target column
# =============================================================================

def detect_problem_type(df, target_column):
    """
    Automatically detect whether the problem is Regression or Classification
    based on the selected target column.

    Parameters:
        df            : pd.DataFrame
        target_column : str — name of the target column

    Returns:
        problem_type  : "regression" or "classification"
        reason        : str — human-readable explanation of why
        confidence    : "high" or "medium"
    """
    target = df[target_column]
    n_unique = target.nunique()
    dtype = target.dtype
    total_rows = len(target)

    # Rule 1: String/object dtype → always Classification
    if dtype == object or dtype.name == "category":
        return "classification", f"Target column contains text categories ({n_unique} unique values).", "high"

    # Rule 2: Boolean → Classification
    if dtype == bool:
        return "classification", "Target column is boolean (True/False).", "high"

    # Rule 3: Only 2 unique values → Binary Classification
    if n_unique == 2:
        unique_vals = sorted(target.unique())
        return "classification", f"Target column has exactly 2 unique values {unique_vals} — binary classification.", "high"

    # Rule 4: Few unique integers (≤ 10% of rows AND ≤ 20 unique) → Classification
    if n_unique <= 20 and (n_unique / total_rows) < 0.05:
        return "classification", f"Target column has {n_unique} unique integer values — treating as multi-class classification.", "high"

    # Rule 5: Float dtype with many unique values → Regression
    if dtype in [np.float32, np.float64] and n_unique > 20:
        return "regression", f"Target column is continuous float with {n_unique} unique values.", "high"

    # Rule 6: Integer with many unique values → Regression
    if n_unique > 20:
        return "regression", f"Target column has {n_unique} unique values ({round(n_unique/total_rows*100, 1)}% of rows) — treating as continuous regression target.", "high"

    # Rule 7: Ambiguous case — let user know
    return "classification", f"Target column has {n_unique} unique values. Defaulting to classification — you can override this if needed.", "medium"


# =============================================================================
# FUNCTION 5 — Get Dataset Summary
# Returns a comprehensive statistical summary of the dataset
# =============================================================================

def get_dataset_summary(df, target_column):
    """
    Generate a comprehensive summary of the dataset.

    Parameters:
        df            : pd.DataFrame
        target_column : str

    Returns:
        summary       : dict with all summary statistics
    """
    summary = {}

    # Basic shape
    summary["n_rows"] = df.shape[0]
    summary["n_cols"] = df.shape[1]
    summary["n_features"] = df.shape[1] - 1  # excluding target

    # Missing values
    missing = df.isnull().sum()
    summary["missing_values"] = missing[missing > 0].to_dict()
    summary["total_missing"] = int(missing.sum())

    # Column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    summary["numeric_columns"] = [c for c in numeric_cols if c != target_column]
    summary["categorical_columns"] = [c for c in categorical_cols if c != target_column]
    summary["target_column"] = target_column

    # Feature columns (all except target)
    summary["feature_columns"] = [c for c in df.columns if c != target_column]

    # Statistical summary for numeric columns
    stats = {}
    for col in numeric_cols:
        col_data = df[col].dropna()
        stats[col] = {
            "min": round(float(col_data.min()), 4),
            "q1": round(float(col_data.quantile(0.25)), 4),
            "median": round(float(col_data.median()), 4),
            "q3": round(float(col_data.quantile(0.75)), 4),
            "max": round(float(col_data.max()), 4),
            "std": round(float(col_data.std()), 4),
            "missing": int(df[col].isnull().sum())
        }
    summary["statistics"] = stats

    # Target distribution
    target_data = df[target_column].dropna()
    if target_data.dtype == object or target_data.nunique() <= 20:
        # Classification target — value counts
        summary["target_distribution"] = target_data.value_counts().to_dict()
        summary["target_type"] = "categorical"
    else:
        # Regression target — percentile summary
        summary["target_distribution"] = {
            "min": round(float(target_data.min()), 4),
            "q1": round(float(target_data.quantile(0.25)), 4),
            "median": round(float(target_data.median()), 4),
            "q3": round(float(target_data.quantile(0.75)), 4),
            "max": round(float(target_data.max()), 4),
            "mean": round(float(target_data.mean()), 4),
            "std": round(float(target_data.std()), 4)
        }
        summary["target_type"] = "continuous"

    return summary


# =============================================================================
# FUNCTION 6 — Clean Data
# Applies automatic cleaning before model training
# =============================================================================

def clean_data(df, target_column):
    """
    Apply automatic data cleaning steps.

    Parameters:
        df            : pd.DataFrame
        target_column : str

    Returns:
        df_clean      : cleaned pd.DataFrame
        cleaning_log  : list of actions taken
    """
    df_clean = df.copy()
    cleaning_log = []

    # Step 1: Remove duplicate rows
    before = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    after = len(df_clean)
    if before != after:
        cleaning_log.append(f"Removed {before - after} duplicate rows.")

    # Step 2: Fill missing values in numeric columns with median
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        missing_count = df_clean[col].isnull().sum()
        if missing_count > 0:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            cleaning_log.append(f"Filled {missing_count} missing values in '{col}' with median ({median_val:.2f}).")

    # Step 3: Drop rows where target is missing
    target_missing = df_clean[target_column].isnull().sum()
    if target_missing > 0:
        df_clean = df_clean.dropna(subset=[target_column])
        cleaning_log.append(f"Dropped {target_missing} rows with missing target values.")

    # Step 4: Encode low-cardinality categorical columns (≤10 unique values)
    # Drop high-cardinality categorical columns (>10 unique values)
    non_numeric_cols = df_clean.select_dtypes(exclude=[np.number]).columns.tolist()
    # Don't touch the target column here — handle it separately
    non_numeric_features = [c for c in non_numeric_cols if c != target_column]

    for col in non_numeric_features:
        n_unique = df_clean[col].nunique()
        if n_unique <= 10:
            # Label encode: assign integer 0,1,2... to each unique category
            # We use label encoding (not one-hot) to keep the DataFrame shape simple
            # and compatible with tree-based models like RF and XGBoost
            categories = sorted(df_clean[col].dropna().unique())
            mapping = {cat: idx for idx, cat in enumerate(categories)}
            df_clean[col] = df_clean[col].map(mapping)
            cleaning_log.append(
                f"Encoded categorical column '{col}' → {mapping}."
            )
        else:
            # Too many unique values — drop the column
            df_clean = df_clean.drop(columns=[col])
            cleaning_log.append(
                f"Dropped high-cardinality column '{col}' ({n_unique} unique values — too many to encode meaningfully)."
            )

    # Step 5: Handle categorical target column (for classification)
    if df_clean[target_column].dtype == object:
        categories = sorted(df_clean[target_column].dropna().unique())
        mapping = {cat: idx for idx, cat in enumerate(categories)}
        df_clean[target_column] = df_clean[target_column].map(mapping)
        cleaning_log.append(
            f"Encoded target column '{target_column}' → {mapping}."
        )

    if not cleaning_log:
        cleaning_log.append("No cleaning required. Dataset is already clean.")

    return df_clean, cleaning_log
