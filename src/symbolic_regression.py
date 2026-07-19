# =============================================================================
# symbolic_regression.py
# Explainable AI Workbench — Symbolic Regression Layer
#
# Architecture (Updated):
#   - PySR now trains on ALL top features simultaneously
#   - x_feature  → PySR column "x"  → Desmos horizontal axis
#   - slider_features → PySR columns "a", "b", "c"... → Desmos sliders
#   - format_equation uses sympy.latex() for mathematically correct Desmos LaTeX
#   - Background threading prevents UI freeze
# =============================================================================

import numpy as np
import pandas as pd
import threading
import time as _time
import re
import sympy as sp
from pysr import PySRRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# MODULE-LEVEL THREAD COMMUNICATION
# These dicts persist across Streamlit reruns within the same process.
# Background thread writes here; Streamlit polls here every 3 seconds.
# =============================================================================

_thread_status  = {}
_thread_results = {}
_thread_errors  = {}
_thread_start   = {}


# =============================================================================
# COMPUTE MODE CONFIGURATIONS
# LOCAL (Windows development) — reduced for speed
# Switch to deployment values before pushing to Hugging Face Spaces
# =============================================================================

# ⚠️  DEPLOYMENT NOTE:
# Before pushing to Hugging Face Spaces, replace COMPUTE_MODES with:
#   Quick:       niterations=15, populations=20, timeout_in_seconds=120
#   Balanced:    niterations=40, populations=30, timeout_in_seconds=300
#   Deep Search: niterations=80, populations=40, timeout_in_seconds=600

COMPUTE_MODES = {
    "⚡ Quick": {
        "niterations": 5,
        "populations": 8,
        "timeout_in_seconds": 60,
        "description": "Fast exploration — good for simple relationships",
        "estimated_time": "~30–60 seconds"
    },
    "⚖️ Balanced": {
        "niterations": 15,
        "populations": 15,
        "timeout_in_seconds": 150,
        "description": "Recommended — good balance of speed and accuracy",
        "estimated_time": "~1–2 minutes"
    },
    "🔬 Deep Search": {
        "niterations": 25,
        "populations": 20,
        "timeout_in_seconds": 270,
        "description": "Thorough search — best for complex relationships (4 min max)",
        "estimated_time": "~3–4 minutes"
    }
}


# =============================================================================
# COMPLEXITY PREFERENCE CONFIGURATIONS
# Controls PySR's parsimony — higher parsimony = simpler equations
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
# FUNCTION 1 — Fidelity Rating
# =============================================================================

def get_fidelity_rating(fidelity_score):
    """
    Returns (rating, color_hex, message) based on fidelity score (0–100).

    Fidelity = R²(surrogate_predictions, model_predictions)
    NOT accuracy against ground truth — measures surrogate faithfulness.
    100% fidelity on Linear Regression is CORRECT and EXPECTED.
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
# FUNCTION 2 — Variable Mapping
# Maps feature names to single-letter Desmos variable names.
# These become PySR's DataFrame column names so sympy uses them directly.
# =============================================================================

def get_variable_mapping(x_feature, slider_features):
    """
    Create bidirectional mapping between feature names and Desmos variable names.

    x_feature            → "x"  (horizontal axis in Desmos)
    slider_features[0]   → "a"  (first Desmos slider)
    slider_features[1]   → "b"  (second Desmos slider)
    slider_features[2]   → "c"  (third Desmos slider)
    ...and so on...

    Why single letters?
        Desmos only supports single-letter variable names.
        Multi-character names like "SquareFootage" are interpreted as
        multiplications of individual letters (S×q×u×a×r×e...) which
        causes the warning triangle and prevents curve plotting.

    Why name PySR columns this way?
        PySR uses DataFrame column names as sympy Symbol names.
        If columns are named "x", "a", "b", PySR's sympy output
        automatically contains x, a, b — which are valid Desmos variables.

    Parameters:
        x_feature      : str  — the single x-axis feature
        slider_features: list — all other top features (become sliders)

    Returns:
        var_to_feature : dict — {"x": "Present_Price", "a": "Year", ...}
        feature_to_var : dict — {"Present_Price": "x", "Year": "a", ...}
    """
    var_to_feature = {"x": x_feature}
    feature_to_var = {x_feature: "x"}

    for i, feat in enumerate(slider_features):
        var = chr(ord('a') + i)   # a, b, c, d, e, f...
        var_to_feature[var]  = feat
        feature_to_var[feat] = var

    return var_to_feature, feature_to_var


# =============================================================================
# FUNCTION 3 — Prepare PySR Training Data (REDESIGNED)
# Now trains on ALL top features simultaneously using real data rows.
# =============================================================================

def prepare_pysr_data(
    df_clean,
    target_column,
    model,
    feature_names,
    x_feature,
    slider_features,
    anchored_values,
    problem_type,
    n_samples=1000,
    random_state=42
):
    """
    Build X_pysr and y_pysr for surrogate training.

    KEY DESIGN DECISIONS:
    ─────────────────────
    1. REAL DATA ROWS — not synthetic grids.
       We sample actual rows from df_clean for the top features.
       This ensures PySR learns from realistic feature value distributions
       and realistic feature correlations, not arbitrary combinations.

    2. ALL TOP FEATURES — not just 2.
       x_feature and ALL slider_features are passed to PySR simultaneously.
       PySR can then discover equations that involve any combination of them.
       This is architecturally richer than the old 2-feature approach.

    3. NON-TOP FEATURES ANCHORED.
       Features not in top features are fixed at their anchored (median)
       values for every row. This matches what the user set in Phase 4.

    4. PROPER VARIABLE NAMING.
       X_pysr columns are named "x", "a", "b", "c"... via get_variable_mapping.
       PySR's internal sympy engine uses column names as symbol names.
       The resulting sympy expression contains x, a, b, c — valid Desmos vars.

    Parameters:
        df_clean        : pd.DataFrame — cleaned dataset
        target_column   : str
        model           : fitted ML model (RF, XGBoost, etc.)
        feature_names   : list[str] — all feature column names in order
        x_feature       : str — the single x-axis feature
        slider_features : list[str] — all other top features
        anchored_values : dict — {feature: value} for non-top features
        problem_type    : "regression" or "classification"
        n_samples       : int — max rows to use for PySR training
        random_state    : int — for reproducible sampling

    Returns:
        X_pysr : pd.DataFrame — columns named "x", "a", "b"... (top features)
        y_pysr : np.array    — model predictions (surrogate target)
    """
    top_features = [x_feature] + list(slider_features)
    _, feature_to_var = get_variable_mapping(x_feature, slider_features)

    # ── Sample real rows from df_clean ───────────────────────────────────────
    n = min(n_samples, len(df_clean))
    if len(df_clean) > n_samples:
        df_sample = df_clean.sample(n=n, random_state=random_state)
    else:
        df_sample = df_clean.copy()

    df_sample = df_sample.reset_index(drop=True)

    # ── Build full feature matrix for model prediction ────────────────────────
    # Top features: use actual sampled values
    # Non-top features: fixed at anchored (median) values
    X_full = np.zeros((n, len(feature_names)))

    for i, feat in enumerate(feature_names):
        if feat == target_column:
            continue
        if feat in top_features and feat in df_sample.columns:
            X_full[:, i] = df_sample[feat].values.astype(float)
        else:
            # Use anchored value (median set by user in Phase 4)
            fallback = float(df_clean[feat].median()) if feat in df_clean.columns else 0.0
            X_full[:, i] = float(anchored_values.get(feat, fallback))

    # Apply NaN safety
    X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Get model predictions (surrogate target) ──────────────────────────────
    if problem_type == "regression":
        y_pysr = model.predict(X_full)
    else:
        # For classification: surrogate approximates probability of positive class
        y_pysr = model.predict_proba(X_full)[:, 1]

    y_pysr = np.nan_to_num(y_pysr, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Build X_pysr with Desmos variable names as column names ───────────────
    # Column names become PySR's sympy Symbol names
    X_pysr = df_sample[top_features].copy().astype(float)
    X_pysr = X_pysr.rename(columns=feature_to_var)
    # Columns are now named "x", "a", "b", "c"...

    return X_pysr, y_pysr


# =============================================================================
# FUNCTION 4 — Run PySR (core engine, called inside background thread)
# =============================================================================

def run_pysr(
    X_pysr,
    y_pysr,
    compute_mode="⚖️ Balanced",
    complexity_mode="Balanced",
    random_state=42
):
    """
    Run PySR symbolic regression. CPU-intensive — always called from
    a background daemon thread, never from Streamlit's main thread.

    X_pysr columns must be named "x", "a", "b", "c"... (from prepare_pysr_data).
    PySR uses these column names as sympy Symbol names, so the output
    sympy expression automatically contains the correct Desmos variable names.

    Returns:
        model_pysr    : fitted PySRRegressor (stores .sympy() and .equations_)
        best_eq_str   : str — best equation as sympy string (for display/cache)
        all_equations : pd.DataFrame — full Pareto front
    """
    mode_config       = COMPUTE_MODES[compute_mode]
    complexity_config = COMPLEXITY_MODES[complexity_mode]

    model_pysr = PySRRegressor(
        niterations        = mode_config["niterations"],
        populations        = mode_config["populations"],
        timeout_in_seconds = mode_config["timeout_in_seconds"],
        parsimony          = complexity_config["parsimony"],

        # Required for deterministic stable runs on Windows
        parallelism        = "serial",
        random_state       = random_state,
        deterministic      = True,

        # Mathematical operators available to PySR
        binary_operators   = ["+", "-", "*", "/", "^"],
        unary_operators    = ["sqrt", "log", "exp", "abs", "sin", "cos"],

        # Equation complexity bounds
        maxsize            = 20,
        maxdepth           = 6,

        # Suppress all Julia/PySR output
        verbosity          = 0,
        progress           = False,

        # Temp file management
        tempdir            = "pysr_temp",
        delete_tempfiles   = True,
    )

    model_pysr.fit(X_pysr, y_pysr)

    best_eq_str   = str(model_pysr.sympy())
    all_equations = model_pysr.equations_

    return model_pysr, best_eq_str, all_equations


# =============================================================================
# FUNCTION 5 — Start Background Thread
# =============================================================================

def start_pysr_thread(
    X_pysr,
    y_pysr,
    compute_mode,
    complexity_mode,
    cache_key,
    random_state=42
):
    """
    Launch PySR in a daemon background thread with a hard-timeout watchdog.

    Two threads are launched:
    1. Worker thread  — runs PySR, writes result to _thread_results
    2. Watchdog thread — if worker overruns hard limit, forces error status

    The worker thread checks _thread_status before writing its result —
    if the watchdog has already set status to "error", the worker result
    is discarded (prevents overwriting timeout error with late success).
    """
    soft_timeout = COMPUTE_MODES[compute_mode]["timeout_in_seconds"]
    hard_limit   = soft_timeout + 30  # 30-second grace beyond PySR's soft timeout

    def _worker():
        try:
            _thread_status[cache_key]  = "running"
            model, best_eq, all_eqs   = run_pysr(
                X_pysr, y_pysr,
                compute_mode, complexity_mode,
                random_state
            )
            # Only write result if watchdog hasn't already timed us out
            if _thread_status.get(cache_key) != "error":
                _thread_results[cache_key] = (model, best_eq, all_eqs)
                _thread_status[cache_key]  = "complete"
        except Exception as exc:
            if _thread_status.get(cache_key) != "error":
                _thread_errors[cache_key]  = str(exc)
                _thread_status[cache_key]  = "error"

    def _watchdog():
        """Forcefully marks job as error if PySR overruns the hard limit."""
        _time.sleep(hard_limit)
        if _thread_status.get(cache_key) == "running":
            _thread_errors[cache_key] = (
                f"Hard timeout reached ({hard_limit}s). "
                f"PySR exceeded the maximum allowed time. "
                f"Try Quick or Balanced mode for faster results."
            )
            _thread_status[cache_key] = "error"

    # Initialize state
    _thread_status[cache_key]  = "starting"
    _thread_start[cache_key]   = _time.time()
    _thread_results.pop(cache_key, None)
    _thread_errors.pop(cache_key, None)

    # Launch both threads as daemons (auto-killed on process exit)
    threading.Thread(target=_worker,   daemon=True).start()
    threading.Thread(target=_watchdog, daemon=True).start()


# =============================================================================
# THREAD STATUS HELPERS (unchanged)
# =============================================================================

def get_thread_status(cache_key):
    return _thread_status.get(cache_key, "idle")

def get_thread_result(cache_key):
    return _thread_results.get(cache_key)

def get_thread_error(cache_key):
    return _thread_errors.get(cache_key)

def get_elapsed_seconds(cache_key):
    start = _thread_start.get(cache_key)
    return int(_time.time() - start) if start else 0

def clear_thread(cache_key):
    _thread_status.pop(cache_key, None)
    _thread_results.pop(cache_key, None)
    _thread_errors.pop(cache_key, None)
    _thread_start.pop(cache_key, None)


# =============================================================================
# FUNCTION 6 — Compute Fidelity Score (unchanged)
# =============================================================================

def compute_fidelity(model_pysr, model_original, X_pysr, y_model_predictions):
    """
    Fidelity = R²(surrogate_predictions, original_model_predictions) × 100

    This is NOT accuracy against ground truth.
    It measures how faithfully the surrogate copies the black-box model.
    100% fidelity on Linear Regression is correct — it IS already an equation.
    """
    y_surrogate = model_pysr.predict(X_pysr)
    valid       = np.isfinite(y_surrogate) & np.isfinite(y_model_predictions)

    if valid.sum() < 2:
        return 0.0, y_surrogate, float("inf")

    r2  = r2_score(y_model_predictions[valid], y_surrogate[valid])
    mae = mean_absolute_error(y_model_predictions[valid], y_surrogate[valid])

    return round(max(0.0, min(1.0, r2)) * 100, 2), y_surrogate, round(mae, 4)


# =============================================================================
# FUNCTION 7 — Format Equation (COMPLETELY REWRITTEN)
# Uses sympy.latex() for mathematically correct Desmos LaTeX.
# =============================================================================

def format_equation(pysr_model, var_to_feature):
    """
    Convert PySR's best equation to two forms:
    1. Human-readable string with actual feature names (for display)
    2. Desmos-compatible LaTeX string with x, a, b, c variables (for graph)

    WHY sympy.latex() INSTEAD OF STRING REPLACEMENT:
    ─────────────────────────────────────────────────
    PySR outputs equations in Python/sympy notation (e.g. sqrt(x), cos(x)).
    Simple string replacement like sqrt→\\sqrt breaks on nested expressions
    and complex equations, causing the Desmos warning triangle.

    sympy.latex() produces mathematically correct LaTeX:
        sqrt(x)    -> \\sqrt{x}
        cos(x)     -> \\cos{\\left(x \\right)}
        log(x)     -> \\log{\\left(x \\right)}  (fixed to \\\1n below)
        x**1.5     → x^{1.5}
        abs(x)     -> \\left|{x}\\right|
        a/b        -> \\frac{a}{b}

    The only post-processing needed:
        \\log -> \\ln  (sympy uses \\\1og for natural log; Desmos uses \\\1n)

    VARIABLE NAMES:
    ───────────────
    Since X_pysr columns were named "x", "a", "b", "c" in prepare_pysr_data,
    PySR's sympy expression contains sp.Symbol("x"), sp.Symbol("a") etc.
    These are exactly the Desmos variable names — no substitution needed
    for the Desmos LaTeX version.

    For the human-readable version, we substitute x→feature_name etc.
    using regex with word boundaries (safe for single-letter variable names).

    Parameters:
        pysr_model     : fitted PySRRegressor — has .sympy() method
        var_to_feature : dict — {"x": "Present_Price", "a": "Year", ...}

    Returns:
        human_readable : str — equation with real feature names
        desmos_latex   : str — valid Desmos LaTeX with x, a, b, c variables
    """

    # ── Get sympy expression from PySR ────────────────────────────────────────
    try:
        sympy_expr = pysr_model.sympy()
    except Exception:
        # Fallback if sympy() fails
        return "Equation could not be formatted", "y = 0"

    # ── 1. Desmos LaTeX via sympy.latex() ─────────────────────────────────────
    desmos_latex = sp.latex(sympy_expr)

    # Fix: sympy uses \\\1og for natural logarithm; Desmos uses \\\1n
    # This is the only essential post-processing needed
    desmos_latex = desmos_latex.replace(r'\\\1og', r'\\\1n')

    # ── 2. Human-readable with feature names ──────────────────────────────────
    # Start with sympy's string representation
    human_readable = str(sympy_expr)

    # Substitute single-letter variable names with actual feature names
    # Sort by variable name length descending (safety measure, all are length 1 here)
    # Use \\\1 word boundaries to avoid partial matches
    for var_name, feat_name in sorted(
        var_to_feature.items(), key=lambda kv: -len(kv[0])
    ):
        # \\\1 word boundary ensures "a" doesn't match inside "abs" or "sqrt"
        human_readable = re.sub(
            r'\b' + re.escape(var_name) + r'\b',
            lambda m: feat_name,
            human_readable
        )

    return human_readable, desmos_latex


# =============================================================================
# FUNCTION 8 — Equation Complexity Info (unchanged)
# =============================================================================

def get_equation_complexity(model_pysr):
    """Extract complexity metrics from PySR's best Pareto-front equation."""
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
# PRE-WARMING FUNCTIONS
# Run minimal PySR jobs at startup to trigger Julia's JIT compilation.
# By the time the user reaches Symbolic Regression, Julia is warmer.
# =============================================================================

def prewarm_julia():
    """
    Synchronous warmup: 1-iteration PySR on 20 synthetic points.
    Called at app startup — blocks briefly but fast (~5s after first run).
    Returns True on success, False on failure.
    """
    try:
        X_tiny = pd.DataFrame({"x": np.linspace(0, 1, 20), "a": np.linspace(0, 1, 20)})
        y_tiny = (X_tiny["x"] * 2.0 + X_tiny["a"] * 0.5).values

        m = PySRRegressor(
            niterations=1, populations=2,
            parallelism="serial", random_state=42, deterministic=True,
            verbosity=0, progress=False, maxsize=5,
            tempdir="pysr_temp", delete_tempfiles=True
        )
        m.fit(X_tiny, y_tiny)
        return True
    except Exception:
        return False


def start_warmup_thread():
    """
    Non-blocking warmup: 5-iteration PySR in a background thread.
    Compiles more of Julia's JIT code than prewarm_julia().
    Called at app startup — never blocks the UI.
    """
    def _warmup_worker():
        try:
            X_warm = pd.DataFrame({
                "x": np.linspace(0, 10, 50),
                "a": np.linspace(0, 10, 50),
                "b": np.linspace(0, 10, 50)
            })
            y_warm = (X_warm["x"] * 2.5 + X_warm["a"] ** 0.5).values

            m = PySRRegressor(
                niterations=5, populations=8,
                parallelism="serial", random_state=42, deterministic=True,
                verbosity=0, progress=False, maxsize=10,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sqrt", "log"],
                tempdir="pysr_temp", delete_tempfiles=True
            )
            m.fit(X_warm, y_warm)
        except Exception:
            pass   # Warmup failure is silent — never affects the user

    threading.Thread(target=_warmup_worker, daemon=True).start()