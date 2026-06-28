# =============================================================================
# model_arena.py
# Explainable AI Workbench — Model Arena Module
# Trains multiple ML models and evaluates them for comparison
# =============================================================================

import numpy as np
import pandas as pd
import time
import joblib
import os

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Cache directory for saving trained models
CACHE_DIR = "model_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


# =============================================================================
# FUNCTION 1 — Prepare Data
# Splits data into train/test sets and scales features
# =============================================================================

def prepare_data(df_clean, target_column, test_size=0.2, random_state=42):
    """
    Split the cleaned DataFrame into train and test sets.

    Parameters:
        df_clean      : cleaned pd.DataFrame
        target_column : str
        test_size     : float — proportion for test set (default 0.2 = 20%)
        random_state  : int — for reproducibility

    Returns:
        X_train, X_test, y_train, y_test : split arrays
        feature_names                    : list of feature column names
    """
    feature_columns = [c for c in df_clean.columns if c != target_column]
    X = df_clean[feature_columns].values
    y = df_clean[target_column].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )

    # Safety check — replace any remaining NaN/Inf with 0
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    return X_train, X_test, y_train, y_test, feature_columns


# =============================================================================
# FUNCTION 2 — Train All Models
# Trains all models for the detected problem type and returns results
# =============================================================================

def train_all_models(df_clean, target_column, problem_type, random_state=42):
    """
    Train all models for the given problem type and return comparison results.

    Parameters:
        df_clean      : cleaned pd.DataFrame
        target_column : str
        problem_type  : "regression" or "classification"
        random_state  : int

    Returns:
        results       : list of dicts — one per model with metrics
        trained_models: dict — model_name → fitted model object
        data_splits   : dict — X_train, X_test, y_train, y_test, feature_names
    """

    X_train, X_test, y_train, y_test, feature_names = prepare_data(
        df_clean, target_column, random_state=random_state
    )

    data_splits = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names
    }

    results = []
    trained_models = {}

    if problem_type == "regression":
        models = _get_regression_models(random_state)
    else:
        models = _get_classification_models(random_state)

    for model_name, model, complexity in models:

        # Check cache first — avoid retraining
        cache_path = os.path.join(CACHE_DIR, f"{model_name.replace(' ', '_')}.joblib")

        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = round(time.time() - start_time, 3)

        # Cache the trained model
        joblib.dump(model, cache_path)

        # Evaluate
        y_pred = model.predict(X_test)

        if problem_type == "regression":
            metrics = _evaluate_regression(y_test, y_pred)
        else:
            metrics = _evaluate_classification(y_test, y_pred, model, X_test)

        result = {
            "model_name": model_name,
            "complexity": complexity,
            "train_time": train_time,
            **metrics
        }

        results.append(result)
        trained_models[model_name] = model

    return results, trained_models, data_splits


# =============================================================================
# HELPER — Get Regression Models
# =============================================================================

def _get_regression_models(random_state):
    """
    Returns list of (name, model_object, complexity_label) for regression.
    """
    return [
        (
            "Linear Regression",
            LinearRegression(),
            "transparent"
        ),
        (
            "Random Forest",
            RandomForestRegressor(
                n_estimators=100,
                random_state=random_state,
                n_jobs=-1
            ),
            "black_box"
        ),
        (
            "XGBoost",
            xgb.XGBRegressor(
                n_estimators=100,
                random_state=random_state,
                verbosity=0,
                n_jobs=-1
            ),
            "black_box"
        ),
    ]


# =============================================================================
# HELPER — Get Classification Models
# =============================================================================

def _get_classification_models(random_state):
    """
    Returns list of (name, model_object, complexity_label) for classification.
    """
    return [
        (
            "Logistic Regression",
            LogisticRegression(
                max_iter=1000,
                random_state=random_state
            ),
            "transparent"
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                n_jobs=-1
            ),
            "black_box"
        ),
        (
            "XGBoost",
            xgb.XGBClassifier(
                n_estimators=100,
                random_state=random_state,
                verbosity=0,
                n_jobs=-1,
                eval_metric="logloss"
            ),
            "black_box"
        ),
    ]


# =============================================================================
# HELPER — Evaluate Regression
# =============================================================================

def _evaluate_regression(y_test, y_pred):
    """
    Compute regression metrics.
    Returns dict with r2, mae, rmse.
    """
    r2 = round(r2_score(y_test, y_pred), 4)
    mae = round(mean_absolute_error(y_test, y_pred), 4)
    rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)

    return {
        "R²": r2,
        "MAE": mae,
        "RMSE": rmse
    }


# =============================================================================
# HELPER — Evaluate Classification
# =============================================================================

def _evaluate_classification(y_test, y_pred, model, X_test):
    """
    Compute classification metrics.
    Returns dict with accuracy, precision, recall, f1.
    """
    # Determine averaging strategy
    n_classes = len(np.unique(y_test))
    avg = "binary" if n_classes == 2 else "weighted"

    accuracy  = round(accuracy_score(y_test, y_pred), 4)
    precision = round(precision_score(y_test, y_pred, average=avg, zero_division=0), 4)
    recall    = round(recall_score(y_test, y_pred, average=avg, zero_division=0), 4)
    f1        = round(f1_score(y_test, y_pred, average=avg, zero_division=0), 4)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }


# =============================================================================
# FUNCTION 3 — Get Comparison DataFrame
# Formats results into a clean DataFrame for display
# =============================================================================

def get_comparison_dataframe(results, problem_type):
    """
    Convert results list into a formatted DataFrame for Streamlit display.

    Parameters:
        results      : list of dicts from train_all_models()
        problem_type : "regression" or "classification"

    Returns:
        df_display   : pd.DataFrame formatted for st.dataframe()
    """
    rows = []
    for r in results:
        complexity_label = "⭐ Transparent" if r["complexity"] == "transparent" else "🔴 Black Box"

        if problem_type == "regression":
            row = {
                "Model": r["model_name"],
                "R²  ↑": r["R²"],
                "MAE  ↓": r["MAE"],
                "RMSE  ↓": r["RMSE"],
                "Train Time": f"{r['train_time']}s",
                "Type": complexity_label
            }
        else:
            row = {
                "Model": r["model_name"],
                "Accuracy  ↑": r["Accuracy"],
                "Precision  ↑": r["Precision"],
                "Recall  ↑": r["Recall"],
                "F1 Score  ↑": r["F1 Score"],
                "Train Time": f"{r['train_time']}s",
                "Type": complexity_label
            }
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# FUNCTION 4 — Get Interpretability Warning
# Returns the warning message for a selected model
# =============================================================================

def get_interpretability_warning(model_name, problem_type):
    """
    Returns interpretability warning text and type for the selected model.

    Returns:
        message : str
        level   : "info" (transparent) or "warning" (black box)
    """
    transparent_models = ["Linear Regression", "Logistic Regression"]

    if model_name in transparent_models:
        return (
            f"⭐ **{model_name}** is already mathematically transparent — "
            f"its internal logic is a readable equation. "
            f"PySR surrogate may not add significant new insight, "
            f"but you can still proceed to explore the symbolic form.",
            "info"
        )
    else:
        return (
            f"🔴 **{model_name}** is a black-box model. "
            f"PySR surrogate discovery will be most valuable here — "
            f"it will extract a readable mathematical approximation "
            f"of this model's complex decision logic.",
            "warning"
        )


# =============================================================================
# FUNCTION 5 — Get Best Model Name
# Returns the name of the best performing model based on primary metric
# =============================================================================

def get_best_model(results, problem_type):
    """
    Identify the best performing model based on primary metric.
    Regression  → highest R²
    Classification → highest F1 Score

    Returns:
        best_model_name : str
    """
    if problem_type == "regression":
        best = max(results, key=lambda x: x["R²"])
    else:
        best = max(results, key=lambda x: x["F1 Score"])

    return best["model_name"]
