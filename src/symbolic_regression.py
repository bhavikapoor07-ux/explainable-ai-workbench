# =============================================================================
# symbolic_regression.py
# Explainable AI Workbench — Symbolic Regression Layer
# Uses PySR (research-grade symbolic regression) with background threading
# so Streamlit UI never freezes during equation discovery
# =============================================================================

import numpy as np
import pandas as pd
import threading
import time as _time
from pysr import PySRRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# MODULE-LEVEL THREAD COMMUNICATION
# These dicts persist across Streamlit reruns within the same process.
# Background thread writes here; Streamlit polls here.
# =============================================================================

_thread_status  = {}   # {cache_key: "idle"|"starting"|"running"|"complete"|"error"}
_thread_results = {}   # {cache_key: (model, best_eq, all_eqs)}
_thread_errors  = {}   # {cache_key: error_message_str}
_thread_start   = {}   # {cache_key: float timestamp}


# =============================================================================
# COMPUTE MODE CONFIGURATIONS
# timeout_in_seconds guarantees PySR always returns within the hard cap
# =============================================================================
# =============================================================
# ⚠️ DEPLOYMENT NOTE:
# Before pushing to Hugging Face Spaces, replace COMPUTE_MODES
# with the deployment version (higher niterations/populations)
# =============================================================

COMPUTE_MODES = {
    "⚡ Quick": {
        "niterations": 5,
        "populations": 8,
        "timeout_in_seconds": 60,
        "description": "Fast exploration — good for simple relationships",
        "estimated_time": "~30-60 seconds"
    },
    "⚖️ Balanced": {
        "niterations": 15,
        "populations": 15,
        "timeout_in_seconds": 150,
        "description": "Recommended — good balance of speed and accuracy",
        "estimated_time": "~1-2 minutes"
    },
    "🔬 Deep Search": {
        "niterations": 25,
        "populations": 20,
        "timeout_in_seconds": 270,
        "description": "Thorough search — best for complex relationships",
        "estimated_time": "~3-4 minutes"
    }
}


# =============================================================================
# COMPLEXITY PREFERENCE CONFIGURATIONS
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
    Returns (rating, color_hex, message) based on fidelity score (0-100).
    """
    if fidelity_score >= 95:
        return "Excellent", "#4caf50", "The surrogate equation closely mirrors the original model."
    elif fidelity_score >= 85:
        return "Good", "#2196f3", "The surrogate is a strong approximation of the model."
    elif fidelity_score >= 70:
        return "Fair", "#ff9800", "The surrogate captures the main trend but misses some complexity."
    elif fidelity_score >= 50:
        return "Weak", "#ff5722", "Rough approximation. Try Balanced or Deep Search mode."
    else:
        return "Unreliable", "#f44336", "Poor fit. Try Deep Search or select different features."


# =============================================================================
# FUNCTION 1 — Prepare PySR Training Data
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
    Build X and y for PySR by:
    1. Sampling axis features across their data range
    2. Fixing all other features at their anchored (median) values
    3. Querying the trained ML model for predictions — PySR approximates THESE,
       not the ground truth labels. This is what makes it a surrogate.
    """

    col1 = df_clean[axis_feature_1].dropna()
    col2 = df_clean[axis_feature_2].dropna()

    axis1_samples = np.linspace(float(col1.min()), float(col1.max()), n_samples)
    axis2_samples = np.linspace(float(col2.min()), float(col2.max()), n_samples)

    X_full = np.zeros((n_samples, len(feature_names)))

    for i, feature in enumerate(feature_names):
        if feature == axis_feature_1:
            X_full[:, i] = axis1_samples
        elif feature == axis_feature_2:
            X_full[:, i] = axis2_samples
        else:
            X_full[:, i] = float(anchored_values.get(feature, 0.0))

    if problem_type == "regression":
        y_pysr = model.predict(X_full)
    else:
        # For classification: surrogate approximates probability of positive class
        y_pysr = model.predict_proba(X_full)[:, 1]

    X_pysr = pd.DataFrame({
        axis_feature_1: axis1_samples,
        axis_feature_2: axis2_samples
    })

    return X_pysr, y_pysr


# =============================================================================
# FUNCTION 2 — Run PySR (called inside background thread)
# =============================================================================

def run_pysr(
    X_pysr,
    y_pysr,
    compute_mode="⚖️ Balanced",
    complexity_mode="Balanced",
    random_state=42
):
    """
    Run PySR symbolic regression. This is CPU-intensive and runs in a
    background thread — never call this directly from Streamlit's main thread.

    Returns:
        model_pysr    : fitted PySRRegressor
        best_equation : str — best equation as sympy string
        all_equations : pd.DataFrame — full Pareto front
    """

    mode_config       = COMPUTE_MODES[compute_mode]
    complexity_config = COMPLEXITY_MODES[complexity_mode]

    model_pysr = PySRRegressor(
        niterations        = mode_config["niterations"],
        populations        = mode_config["populations"],
        timeout_in_seconds = mode_config["timeout_in_seconds"],
        parsimony          = complexity_config["parsimony"],

        # Required for deterministic, stable runs on Windows
        parallelism        = "serial",
        random_state       = random_state,
        deterministic      = True,

        # Mathematical operators available to PySR
        binary_operators   = ["+", "-", "*", "/", "^"],
        unary_operators    = ["sqrt", "log", "exp", "abs", "sin", "cos"],

        # Equation complexity bounds
        maxsize            = 20,
        maxdepth           = 6,

        # Suppress all output
        verbosity          = 0,
        progress           = False,

        # Temp directory management
        tempdir            = "pysr_temp",
        delete_tempfiles   = True,
    )

    model_pysr.fit(X_pysr, y_pysr)

    best_equation = str(model_pysr.sympy())
    all_equations = model_pysr.equations_

    return model_pysr, best_equation, all_equations


# =============================================================================
# FUNCTION 3 — Start Background Thread
# =============================================================================

def start_pysr_thread(
    X_pysr,
    y_pysr,
    compute_mode,
    complexity_mode,
    cache_key,
    random_state=42
):
    soft_timeout = COMPUTE_MODES[compute_mode]["timeout_in_seconds"]
    hard_limit = soft_timeout + 30  # 30 second grace beyond PySR's own timeout

    def _worker():
        try:
            _thread_status[cache_key] = "running"
            model, best_eq, all_eqs = run_pysr(
                X_pysr, y_pysr,
                compute_mode, complexity_mode,
                random_state
            )
            # Only update if watchdog hasn't already killed it
            if _thread_status.get(cache_key) != "error":
                _thread_results[cache_key] = (model, best_eq, all_eqs)
                _thread_status[cache_key] = "complete"
        except Exception as exc:
            if _thread_status.get(cache_key) != "error":
                _thread_errors[cache_key] = str(exc)
                _thread_status[cache_key] = "error"

    def _watchdog():
        """
        Runs in parallel. If PySR's own timeout is ignored (Julia overruns),
        this watchdog forcefully marks the job as error after hard_limit seconds.
        """
        _time.sleep(hard_limit)
        if _thread_status.get(cache_key) == "running":
            _thread_errors[cache_key] = (
                f"Hard timeout reached ({hard_limit}s). "
                f"PySR exceeded the maximum allowed time. "
                f"Try Quick or Balanced mode for faster results."
            )
            _thread_status[cache_key] = "error"

    _thread_status[cache_key]  = "starting"
    _thread_start[cache_key]   = _time.time()
    _thread_results.pop(cache_key, None)
    _thread_errors.pop(cache_key, None)

    # Main worker thread
    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    # Watchdog thread — kills the job if it overruns
    w = threading.Thread(target=_watchdog, daemon=True)
    w.start()


# =============================================================================
# THREAD STATUS HELPERS
# =============================================================================

def get_thread_status(cache_key):
    """Return current status string for a cache key."""
    return _thread_status.get(cache_key, "idle")


def get_thread_result(cache_key):
    """Return (model, best_eq, all_eqs) tuple if complete, else None."""
    return _thread_results.get(cache_key)


def get_thread_error(cache_key):
    """Return error message string if failed, else None."""
    return _thread_errors.get(cache_key)


def get_elapsed_seconds(cache_key):
    """Return seconds elapsed since thread started."""
    start = _thread_start.get(cache_key)
    if start is None:
        return 0
    return int(_time.time() - start)


def clear_thread(cache_key):
    """Clean up all thread state for a cache key."""
    _thread_status.pop(cache_key, None)
    _thread_results.pop(cache_key, None)
    _thread_errors.pop(cache_key, None)
    _thread_start.pop(cache_key, None)


# =============================================================================
# FUNCTION 4 — Compute Fidelity Score
# =============================================================================

def compute_fidelity(model_pysr, model_original, X_pysr, y_model_predictions):
    """
    Fidelity = R²(surrogate_predictions, original_model_predictions).

    This measures how faithfully the symbolic surrogate reproduces the
    black-box model's behavior — NOT accuracy against ground truth labels.
    A fidelity of 97% means the surrogate mirrors the model very closely.
    """

    y_surrogate = model_pysr.predict(X_pysr)
    valid       = np.isfinite(y_surrogate) & np.isfinite(y_model_predictions)

    if valid.sum() < 2:
        return 0.0, y_surrogate, float("inf")

    r2  = r2_score(y_model_predictions[valid], y_surrogate[valid])
    mae = mean_absolute_error(y_model_predictions[valid], y_surrogate[valid])

    fidelity_score = max(0.0, min(1.0, r2)) * 100
    return round(fidelity_score, 2), y_surrogate, round(mae, 4)


# =============================================================================
# FUNCTION 5 — Format Equation For Display And Desmos
# =============================================================================

def format_equation(equation_str, axis_feature_1, axis_feature_2):
    """
    Format gplearn equation string for display and Desmos.
    gplearn uses X0/X1 as variable names in prefix notation.
    We replace variable names and keep the rest as-is for readability.
    """
    # Human readable — just replace variable names
    formatted = equation_str
    formatted = formatted.replace("X0", axis_feature_1)
    formatted = formatted.replace("X1", axis_feature_2)

    # Desmos version — replace variable names with x and y
    import re
    desmos_ready = equation_str
    # Use exact string replacement (case-sensitive, whole occurrences)
    desmos_ready = desmos_ready.replace(str(axis_feature_1), "x")
    desmos_ready = desmos_ready.replace(str(axis_feature_2), "y")
    desmos_ready = desmos_ready.replace("**", "^")
    # Remove any spaces around operators for Desmos compatibility
    desmos_ready = desmos_ready.replace(" + -", " - ")
    desmos_ready = desmos_ready.replace(" - -", " + ")

    return formatted, desmos_ready


# =============================================================================
# FUNCTION 6 — Equation Complexity Info
# =============================================================================

def get_equation_complexity(model_pysr):
    """
    Extract complexity metrics from PySR's best equation in the Pareto front.
    """
    try:
        eqs  = model_pysr.equations_
        best = eqs.iloc[eqs["score"].idxmax()]
        return {
            "complexity": int(best.get("complexity", 0)),
            "loss":        round(float(best.get("loss", 0)), 6),
            "score":       round(float(best.get("score", 0)), 6)
        }
    except Exception:
        return {"complexity": 0, "loss": 0.0, "score": 0.0}


# =============================================================================
# PRE-WARMING FUNCTION
# Runs a trivial 1-iteration PySR job on app startup to trigger Julia's
# JIT compilation. By the time the user reaches Symbolic Regression,
# Julia's compiled cache is warm and equation discovery is much faster.
# =============================================================================

def prewarm_julia():
    """
    Silent warmup: 1 PySR iteration on 20 synthetic points.
    Triggers Julia compilation without the user noticing.
    Returns True on success, False on failure.
    """
    try:
        X_tiny = pd.DataFrame({
            "x1": np.linspace(0, 1, 20),
            "x2": np.linspace(0, 1, 20)
        })
        y_tiny = (X_tiny["x1"] * 2.0 + X_tiny["x2"] * 0.5).values

        warmup_model = PySRRegressor(
            niterations        = 1,
            populations        = 2,
            parallelism        = "serial",
            random_state       = 42,
            deterministic      = True,
            verbosity          = 0,
            progress           = False,
            maxsize            = 5,
            tempdir            = "pysr_temp",
            delete_tempfiles   = True,
        )
        warmup_model.fit(X_tiny, y_tiny)
        return True
    except Exception:
        return False

def start_warmup_thread():
    """
    Run a more comprehensive PySR warmup in a background thread.
    This compiles more of Julia's JIT code upfront so the first
    real equation search is significantly faster.
    Called at app startup — non-blocking.
    """
    def _warmup_worker():
        try:
            X_warm = pd.DataFrame({
                "x1": np.linspace(0, 10, 50),
                "x2": np.linspace(0, 10, 50)
            })
            y_warm = (X_warm["x1"] * 2.5 + np.sqrt(X_warm["x2"])).values

            warm_model = PySRRegressor(
                niterations        = 5,
                populations        = 8,
                parallelism        = "serial",
                random_state       = 42,
                deterministic      = True,
                verbosity          = 0,
                progress           = False,
                maxsize            = 10,
                binary_operators   = ["+", "-", "*", "/"],
                unary_operators    = ["sqrt", "log"],
                tempdir            = "pysr_temp",
                delete_tempfiles   = True,
            )
            warm_model.fit(X_warm, y_warm)
        except Exception:
            pass  # Warmup failure is silent — doesn't affect the user

    t = threading.Thread(target=_warmup_worker, daemon=True)
    t.start()