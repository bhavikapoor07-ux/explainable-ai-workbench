# =============================================================================
# symbolic_regression.py
# Explainable AI Workbench — Symbolic Regression Layer
# Uses PySR to discover mathematical equations that approximate
# the black-box model's predictions
# =============================================================================

import numpy as np
import pandas as pd
from pysr import PySRRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# COMPUTE MODE CONFIGURATIONS
# Each mode defines PySR's search intensity
# =============================================================================

COMPUTE_MODES = {
    "⚡ Quick": {
        "niterations": 8,
        "populations": 15,
        "description": "Fast exploration — good for simple relationships",
        "estimated_time": "~15-30 seconds"
    },
    "⚖️ Balanced": {
        "niterations": 20,
        "populations": 20,
        "description": "Recommended — good balance of speed and accuracy",
        "estimated_time": "~45-90 seconds"
    },
    "🔬 Deep Search": {
        "niterations": 40,
        "populations": 25,
        "description": "Thorough search — best for complex relationships",
        "estimated_time": "~2-4 minutes"
    }
}


# =============================================================================
# COMPLEXITY PREFERENCE CONFIGURATIONS
# Controls PySR's parsimony parameter
# Higher parsimony = simpler equations
# =============================================================================

COMPLEXITY_MODES = {
    "Simplicity": {
        "parsimony": 0.01,
        "description": "Prefers short readable equations even if slightly less accurate"
    },
    "Balanced": {
        "parsimony": 0.001,
        "description": "Default — good equation complexity and accuracy"
    },
    "Fidelity": {
        "parsimony": 0.0001,
        "description": "Prefers accurate equations even if more complex"
    }
}


# =============================================================================
# FIDELITY RATING SCALE
# =============================================================================

def get_fidelity_rating(fidelity_score):
    """
    Returns rating label, color, and message based on fidelity score.

    Parameters:
        fidelity_score : float — 0 to 100

    Returns:
        rating  : str
        color   : str (hex)
        message : str
    """
    if fidelity_score >= 95:
        return "Excellent", "#4caf50", "The surrogate equation closely mirrors the original model."
    elif fidelity_score >= 85:
        return "Good", "#2196f3", "The surrogate is a strong approximation of the model."
    elif fidelity_score >= 70:
        return "Fair", "#ff9800", "The surrogate captures the main trend but misses some complexity."
    elif fidelity_score >= 50:
        return "Weak", "#ff5722", "The surrogate is a rough approximation. Try Balanced or Deep Search mode."
    else:
        return "Unreliable", "#f44336", "Poor approximation. Try Deep Search mode or select different features."


# =============================================================================
# FUNCTION 1 — Prepare PySR Training Data
# Builds X, y arrays for PySR from the selected axes + anchored values
# =============================================================================

def prepare_pysr_data(
    df_clean,
    target_column,
    model,
    feature_names,
    axis_feature_1,
    axis_feature_2,
    anchored_values,
    problem_type,
    n_samples=500
):
    """
    Prepare training data for PySR by:
    1. Building a sample grid of axis_feature_1 and axis_feature_2 values
    2. Fixing remaining features at their anchored (median) values
    3. Getting model predictions as the target for PySR

    Parameters:
        df_clean       : pd.DataFrame — cleaned dataset
        target_column  : str
        model          : trained model object
        feature_names  : list of str — all feature names
        axis_feature_1 : str — X axis feature
        axis_feature_2 : str — Y axis feature
        anchored_values: dict — {feature: fixed_value}
        problem_type   : "regression" or "classification"
        n_samples      : int — number of sample points per axis

    Returns:
        X_pysr : pd.DataFrame — input features for PySR (only axis features)
        y_pysr : np.array — model predictions (surrogate target)
    """

    # Sample values for each axis feature
    col1 = df_clean[axis_feature_1].dropna()
    col2 = df_clean[axis_feature_2].dropna()

    # Create evenly spaced sample points within the data range
    axis1_samples = np.linspace(col1.min(), col1.max(), n_samples)
    axis2_samples = np.linspace(col2.min(), col2.max(), n_samples)

    # Build full feature matrix for model prediction
    # Each row: axis1 value, axis2 value, all other features at anchored values
    X_full = np.zeros((n_samples, len(feature_names)))

    for i, feature in enumerate(feature_names):
        if feature == axis_feature_1:
            X_full[:, i] = axis1_samples
        elif feature == axis_feature_2:
            X_full[:, i] = axis2_samples
        else:
            X_full[:, i] = anchored_values.get(feature, 0.0)

    # Get model predictions
    if problem_type == "regression":
        y_pysr = model.predict(X_full)
    else:
        # For classification: use probability of positive class
        y_pysr = model.predict_proba(X_full)[:, 1]

    # PySR input: only the two axis features
    X_pysr = pd.DataFrame({
        axis_feature_1: axis1_samples,
        axis_feature_2: axis2_samples
    })

    return X_pysr, y_pysr


# =============================================================================
# FUNCTION 2 — Run PySR
# Core symbolic regression function
# =============================================================================

def run_pysr(
    X_pysr,
    y_pysr,
    compute_mode="⚖️ Balanced",
    complexity_mode="Balanced",
    random_state=42
):
    """
    Run PySR symbolic regression to discover a mathematical equation.

    Parameters:
        X_pysr         : pd.DataFrame — input features (axis features only)
        y_pysr         : np.array — model predictions to approximate
        compute_mode   : str — one of COMPUTE_MODES keys
        complexity_mode: str — one of COMPLEXITY_MODES keys
        random_state   : int

    Returns:
        model_pysr     : fitted PySRRegressor object
        best_equation  : str — best equation as string
        all_equations  : pd.DataFrame — all discovered equations
    """

    mode_config       = COMPUTE_MODES[compute_mode]
    complexity_config = COMPLEXITY_MODES[complexity_mode]

    model_pysr = PySRRegressor(
        niterations=mode_config["niterations"],
        populations=mode_config["populations"],
        parsimony=complexity_config["parsimony"],

        # Allowed mathematical operators
        binary_operators=["+", "-", "*", "/", "^"],
        unary_operators=["sqrt", "log", "exp", "abs", "sin", "cos"],

        # Output settings
        verbosity=0,
        progress=False,

        # Equation complexity limits
        maxsize=20,
        maxdepth=6,

        # Reproducibility
        random_state=random_state,
        deterministic=True,
        procs=0,  # single process for stability

        # Temp directory for Julia files
        tempdir="pysr_temp",
        delete_tempfiles=True
    )

    model_pysr.fit(X_pysr, y_pysr)

    # Get best equation
    best_equation = str(model_pysr.sympy())
    all_equations = model_pysr.equations_

    return model_pysr, best_equation, all_equations


# =============================================================================
# FUNCTION 3 — Compute Fidelity Score
# Measures how closely surrogate predictions match original model predictions
# =============================================================================

def compute_fidelity(
    model_pysr,
    model_original,
    X_pysr,
    y_model_predictions
):
    """
    Compute fidelity: how closely surrogate matches original model.

    Fidelity = R² between surrogate predictions and original model predictions.
    This is NOT accuracy against ground truth — it measures surrogate faithfulness.

    Parameters:
        model_pysr          : fitted PySRRegressor
        model_original      : original trained ML model (not used directly)
        X_pysr              : pd.DataFrame — axis features
        y_model_predictions : np.array — original model's predictions

    Returns:
        fidelity_score : float — 0 to 100
        y_surrogate    : np.array — surrogate predictions
        mae            : float — mean absolute error between surrogate and model
    """

    y_surrogate = model_pysr.predict(X_pysr)

    # Handle NaN/Inf in surrogate predictions
    valid_mask = np.isfinite(y_surrogate) & np.isfinite(y_model_predictions)

    if valid_mask.sum() < 2:
        return 0.0, y_surrogate, float("inf")

    r2 = r2_score(
        y_model_predictions[valid_mask],
        y_surrogate[valid_mask]
    )
    mae = mean_absolute_error(
        y_model_predictions[valid_mask],
        y_surrogate[valid_mask]
    )

    # Clip R² to [0, 1] range and convert to percentage
    fidelity_score = max(0.0, min(1.0, r2)) * 100

    return round(fidelity_score, 2), y_surrogate, round(mae, 4)


# =============================================================================
# FUNCTION 4 — Format Equation For Display
# Cleans up the sympy equation string for readable display
# =============================================================================

def format_equation(equation_str, axis_feature_1, axis_feature_2):
    """
    Format the raw sympy equation string for clean display.

    Parameters:
        equation_str   : str — raw sympy string from PySR
        axis_feature_1 : str — X axis feature name
        axis_feature_2 : str — Y axis feature name

    Returns:
        formatted      : str — cleaned equation
        desmos_ready   : str — equation formatted for Desmos (x, y variables)
    """

    formatted = equation_str

    # Create Desmos-compatible version using x and y as variable names
    desmos_ready = equation_str
    desmos_ready = desmos_ready.replace(axis_feature_1, "x")
    desmos_ready = desmos_ready.replace(axis_feature_2, "y")

    # Clean up common sympy formatting
    desmos_ready = desmos_ready.replace("**", "^")
    desmos_ready = desmos_ready.replace("sqrt", "\\sqrt")
    desmos_ready = desmos_ready.replace("log", "\\ln")
    desmos_ready = desmos_ready.replace("exp", "e^")

    return formatted, desmos_ready


# =============================================================================
# FUNCTION 5 — Get Equation Complexity Info
# Returns human-readable complexity info about the discovered equation
# =============================================================================

def get_equation_complexity(model_pysr):
    """
    Extract complexity information about the best equation.

    Returns:
        info : dict with complexity, n_nodes, equation_length
    """
    try:
        best = model_pysr.equations_.iloc[model_pysr.equations_["score"].idxmax()]
        return {
            "complexity": int(best.get("complexity", 0)),
            "loss": round(float(best.get("loss", 0)), 6),
            "score": round(float(best.get("score", 0)), 6)
        }
    except Exception:
        return {"complexity": 0, "loss": 0.0, "score": 0.0}