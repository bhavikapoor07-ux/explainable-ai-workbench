# =============================================================================
# feature_importance.py
# Explainable AI Workbench — Feature Importance Engine
# Computes permutation importance, selects top features adaptively,
# and prepares slider anchor points for remaining features
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


# =============================================================================
# FUNCTION 1 — Compute Permutation Importance
# Uses sklearn's permutation importance on the selected trained model
# =============================================================================

def compute_permutation_importance(
    model,
    X_test,
    y_test,
    feature_names,
    problem_type,
    n_repeats=10,
    random_state=42
):
    """
    Compute permutation feature importance for the selected model.

    Why permutation importance?
    - Model-agnostic: works for ANY model (RF, XGBoost, Linear, etc.)
    - Uses real feature names: no PCA abstraction
    - Measures actual impact: shuffles each feature and measures
      how much the score drops

    Parameters:
        model         : trained model object
        X_test        : np.array — test features
        y_test        : np.array — test target
        feature_names : list of str — feature column names
        problem_type  : "regression" or "classification"
        n_repeats     : int — how many times to shuffle each feature
        random_state  : int

    Returns:
        importance_df : pd.DataFrame with columns:
                        [feature, importance, std, cumulative_importance, rank]
        sorted by importance descending
    """

    # Scoring metric depends on problem type
    scoring = "r2" if problem_type == "regression" else "accuracy"

    # Compute permutation importance
    perm_result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring=scoring,
        n_jobs=-1
    )

    # Build DataFrame
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": perm_result.importances_mean,
        "std": perm_result.importances_std
    })

    # Clip negative importances to 0
    # Negative means the feature actually hurts the model — treat as 0 importance
    importance_df["importance"] = importance_df["importance"].clip(lower=0)

    # Sort by importance descending
    importance_df = importance_df.sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)

    # Add rank
    importance_df["rank"] = importance_df.index + 1

    # Compute percentage importance
    total = importance_df["importance"].sum()
    if total > 0:
        importance_df["importance_pct"] = (
            importance_df["importance"] / total * 100
        ).round(2)
    else:
        importance_df["importance_pct"] = 0.0

    # Compute cumulative importance percentage
    importance_df["cumulative_pct"] = importance_df["importance_pct"].cumsum().round(2)

    return importance_df


# =============================================================================
# FUNCTION 2 — Select Top Features Adaptively
# Keeps features until cumulative importance reaches threshold
# =============================================================================

def select_top_features(importance_df, threshold=85.0):
    """
    Adaptively select top features until cumulative importance
    reaches the specified threshold.

    Example:
        Feature A: 40% → cumulative 40%
        Feature B: 25% → cumulative 65%
        Feature C: 15% → cumulative 80%
        Feature D: 10% → cumulative 90% ← stops here if threshold=85%
        Feature E: 10% → cumulative 100%

    Parameters:
        importance_df : pd.DataFrame from compute_permutation_importance()
        threshold     : float — cumulative importance % to reach (default 85%)

    Returns:
        top_features  : list of str — selected feature names
        cutoff_index  : int — how many features were selected
    """

    # Always select at least 2 features (need 2 for Desmos visualization)
    # and at most all features
    selected = []
    for _, row in importance_df.iterrows():
        selected.append(row["feature"])
        if row["cumulative_pct"] >= threshold:
            break

    # Enforce minimum of 2 features
    if len(selected) < 2:
        selected = importance_df["feature"].tolist()[:2]

    return selected, len(selected)


# =============================================================================
# FUNCTION 3 — Get Slider Anchor Points
# Computes Min, Q1, Median, Q3, Max for each remaining feature
# =============================================================================

def get_slider_anchors(df_clean, feature, target_column):
    """
    Compute the 5 statistical anchor points for a feature's slider.

    Anchor points: Min, Q1, Median, Q3, Max
    These are robust to outliers (no mean, no std).

    Parameters:
        df_clean      : pd.DataFrame — cleaned dataset
        feature       : str — feature column name
        target_column : str — to exclude from computation

    Returns:
        anchors       : dict with keys: min, q1, median, q3, max
        anchor_labels : list of str — display labels
        anchor_values : list of float — corresponding values (deduplicated)
    """

    col_data = df_clean[feature].dropna()

    # Detect if column is integer-like (e.g. Age, Year, Count)
    is_integer_like = (
        col_data.dtype in [int, np.int32, np.int64] or
        (col_data.dtype in [float, np.float32, np.float64] and
         (col_data == col_data.round(0)).all())
    )

    def fmt(v):
        return int(round(v)) if is_integer_like else round(float(v), 4)

    min_val    = fmt(col_data.min())
    q1_val     = fmt(col_data.quantile(0.25))
    median_val = fmt(col_data.median())
    q3_val     = fmt(col_data.quantile(0.75))
    max_val    = fmt(col_data.max())

    # Build ordered list and deduplicate while preserving order
    raw_pairs = [
        ("Min",    min_val),
        ("Q1",     q1_val),
        ("Median", median_val),
        ("Q3",     q3_val),
        ("Max",    max_val),
    ]

    # Remove duplicates (can happen with skewed distributions)
    seen_values = set()
    anchor_labels = []
    anchor_values = []
    for label, val in raw_pairs:
        if val not in seen_values:
            seen_values.add(val)
            anchor_labels.append(label)
            anchor_values.append(val)

    anchors = {
        "min": min_val,
        "q1": q1_val,
        "median": median_val,
        "q3": q3_val,
        "max": max_val
    }

    return anchors, anchor_labels, anchor_values


# =============================================================================
# FUNCTION 4 — Prepare Anchored Feature Values
# Returns default median values for all remaining (non-axis) features
# =============================================================================

def prepare_anchored_values(df_clean, remaining_features):
    """
    Prepare default median anchor values for all remaining features.
    These are the starting values for sliders.

    Parameters:
        df_clean          : pd.DataFrame
        remaining_features: list of str — features NOT chosen as axes

    Returns:
        anchored_values   : dict — {feature_name: median_value}
    """
    anchored_values = {}
    for feature in remaining_features:
        median_val = float(df_clean[feature].median())
        anchored_values[feature] = round(median_val, 4)

    return anchored_values


# =============================================================================
# FUNCTION 5 — Build Prediction Input
# Constructs a full feature vector for model prediction
# using axis values + anchored remaining values
# =============================================================================

def build_prediction_input(
    feature_names,
    axis_feature_1,
    axis_feature_2,
    axis_values_1,
    axis_values_2,
    anchored_values
):
    """
    Build a 2D array of inputs for model prediction by combining:
    - A grid of axis_feature_1 × axis_feature_2 values
    - Fixed median values for all other features

    This is used to generate the surface/curve data for Desmos.

    Parameters:
        feature_names  : list of str — all feature names in order
        axis_feature_1 : str — first selected axis feature
        axis_feature_2 : str — second selected axis feature
        axis_values_1  : np.array — range of values for feature 1
        axis_values_2  : np.array — range of values for feature 2
        anchored_values: dict — {feature: median_value} for remaining features

    Returns:
        X_grid         : np.array shape (n1 * n2, n_features)
        grid_1         : np.array — meshgrid for feature 1
        grid_2         : np.array — meshgrid for feature 2
    """

    grid_1, grid_2 = np.meshgrid(axis_values_1, axis_values_2)
    n_points = grid_1.size

    # Build full feature matrix
    X_grid = np.zeros((n_points, len(feature_names)))

    for i, feature in enumerate(feature_names):
        if feature == axis_feature_1:
            X_grid[:, i] = grid_1.ravel()
        elif feature == axis_feature_2:
            X_grid[:, i] = grid_2.ravel()
        else:
            X_grid[:, i] = anchored_values.get(feature, 0.0)

    return X_grid, grid_1, grid_2