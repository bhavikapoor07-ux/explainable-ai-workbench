# =============================================================================
# app.py
# Explainable AI Workbench — Main Application Entry Point
# Day 6: Desmos Visualization
# =============================================================================

import time
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.data_handler import (
    load_data,
    load_demo_dataset,
    validate_data,
    detect_problem_type,
    get_dataset_summary,
    clean_data,
    get_encoding_maps
)
from src.model_arena import (
    train_all_models,
    get_comparison_dataframe,
    get_interpretability_warning,
    get_best_model
)
from src.feature_importance import (
    compute_permutation_importance,
    select_top_features,
    get_slider_anchors,
    prepare_anchored_values
)
from src.symbolic_regression import (
    COMPUTE_MODES,
    COMPLEXITY_MODES,
    prepare_pysr_data,
    compute_fidelity,
    format_equation,
    get_equation_complexity,
    get_fidelity_rating,
    prewarm_julia,
    start_pysr_thread,
    get_thread_status,
    get_thread_result,
    get_thread_error,
    get_elapsed_seconds,
    clear_thread,
)
from src.desmos_visualizer import (
    build_desmos_html,
    get_feature_bounds,
)

# =============================================================================
# PAGE CONFIGURATION
# Must be the first Streamlit command in the script
# =============================================================================

st.set_page_config(
    page_title="Explainable AI Workbench",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================

st.markdown("""
<style>
    /* Main background */
    .main { background-color: #0e1117; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border: 1px solid #2d3154;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4fc3f7;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8892b0;
        margin-top: 5px;
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #1e2130, transparent);
        border-left: 4px solid #4fc3f7;
        padding: 10px 20px;
        border-radius: 0 8px 8px 0;
        margin: 20px 0 15px 0;
    }

    /* Problem type badge */
    .badge-regression {
        background: linear-gradient(135deg, #1a472a, #2d6a4f);
        border: 1px solid #40916c;
        color: #74c69d;
        padding: 8px 20px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-classification {
        background: linear-gradient(135deg, #1a237e, #283593);
        border: 1px solid #3949ab;
        color: #7986cb;
        padding: 8px 20px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1.1rem;
        display: inline-block;
    }

    /* Warning and info boxes */
    .custom-warning {
        background: #2d2000;
        border: 1px solid #f0a500;
        border-radius: 8px;
        padding: 12px 16px;
        color: #f0c040;
        margin: 8px 0;
    }
    .custom-info {
        background: #001a2d;
        border: 1px solid #0288d1;
        border-radius: 8px;
        padding: 12px 16px;
        color: #4fc3f7;
        margin: 8px 0;
    }

    /* Fix dropdown cursor — show pointer instead of text cursor */
    .stSelectbox div[data-baseweb="select"] * {
        cursor: pointer !important;
    }
    .stSelectbox div[data-baseweb="select"] {
        cursor: pointer !important;
    }

    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE INITIALIZATION
# Streamlit reruns the entire script on every interaction.
# Session state preserves data across reruns.
# =============================================================================

if "df_raw" not in st.session_state:
    st.session_state.df_raw = None          # Original uploaded DataFrame
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None        # Cleaned DataFrame for modeling
if "target_column" not in st.session_state:
    st.session_state.target_column = None   # User-selected target column
if "problem_type" not in st.session_state:
    st.session_state.problem_type = None    # "regression" or "classification"
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None    # Name of dataset
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False    # Flag: data is ready
if "is_demo" not in st.session_state:
    st.session_state.is_demo = False        # Flag: currently using demo dataset
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0       # Incrementing this clears the file uploader

# Model Arena state
if "arena_results" not in st.session_state:
    st.session_state.arena_results = None       # List of model evaluation results
if "trained_models" not in st.session_state:
    st.session_state.trained_models = None      # Dict of trained model objects
if "data_splits" not in st.session_state:
    st.session_state.data_splits = None         # Train/test splits
if "selected_model_name" not in st.session_state:
    st.session_state.selected_model_name = None # User's chosen model
if "arena_complete" not in st.session_state:
    st.session_state.arena_complete = False     # Flag: model selected and ready
if "show_arena" not in st.session_state:
    st.session_state.show_arena = False         # Flag: user clicked proceed

# Feature Importance state
if "importance_df" not in st.session_state:
    st.session_state.importance_df = None       # Full importance DataFrame
if "top_features" not in st.session_state:
    st.session_state.top_features = None        # Adaptively selected features
if "axis_feature_1" not in st.session_state:
    st.session_state.axis_feature_1 = None      # User's X axis choice
if "axis_feature_2" not in st.session_state:
    st.session_state.axis_feature_2 = None      # User's Y axis choice
if "anchored_values" not in st.session_state:
    st.session_state.anchored_values = {}       # Median values for remaining features
if "importance_threshold" not in st.session_state:
    st.session_state.importance_threshold = 85  # Cumulative importance threshold
if "importance_complete" not in st.session_state:
    st.session_state.importance_complete = False
if "show_importance" not in st.session_state:
    st.session_state.show_importance = False
if "encoding_maps" not in st.session_state:
    st.session_state.encoding_maps = {}        # {col: {0:'CNG', 1:'Diesel'}}

# Symbolic Regression state
if "show_symbolic" not in st.session_state:
    st.session_state.show_symbolic = False
if "pysr_model" not in st.session_state:
    st.session_state.pysr_model = None
if "best_equation" not in st.session_state:
    st.session_state.best_equation = None
if "desmos_equation" not in st.session_state:
    st.session_state.desmos_equation = None
if "fidelity_score" not in st.session_state:
    st.session_state.fidelity_score = None
if "symbolic_complete" not in st.session_state:
    st.session_state.symbolic_complete = False
if "pysr_cache" not in st.session_state:
    st.session_state.pysr_cache = {}           # cache per feature pair
if "show_desmos" not in st.session_state:
    st.session_state.show_desmos = False
if "desmos_complete" not in st.session_state:
    st.session_state.desmos_complete = False
if "show_gemini" not in st.session_state:
    st.session_state.show_gemini = False
if "julia_warmed" not in st.session_state:
    st.session_state.julia_warmed = False
# Thread management state
if "pysr_waiting_for" not in st.session_state:
    st.session_state.pysr_waiting_for = None  # cache_key we're polling for
if "pysr_pending_X" not in st.session_state:
    st.session_state.pysr_pending_X = None    # X stored while thread runs
if "pysr_pending_y" not in st.session_state:
    st.session_state.pysr_pending_y = None    # y stored while thread runs

# =============================================================================
# JULIA PRE-WARMING — Runs silently on first app load
# Compiles Julia in the background while user reads the UI
# By the time user reaches Symbolic Regression, Julia is already warm
# =============================================================================

if not st.session_state.julia_warmed:
    warm_container = st.empty()
    warm_container.markdown("""
    <div style='background:linear-gradient(135deg,#1a1f35,#1e2540);
                border:1px solid #4fc3f7;
                border-radius:10px;
                padding:16px 24px;
                margin-bottom:16px;
                text-align:center;'>
        <span style='font-size:1.1rem;'>⚙️</span>
        <span style='color:#8892b0; margin-left:8px; font-size:0.95rem;'>
            Initializing Julia + PySR engine in the background...
            This takes 1-2 minutes on first load and dramatically
            speeds up equation discovery later.
            You can start uploading your dataset right now.
        </span>
    </div>
    """, unsafe_allow_html=True)
    prewarm_julia()
    st.session_state.julia_warmed = True
    warm_container.empty()

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("## 🔬 XAI Workbench")
    st.markdown("---")

    # Progress tracker
    st.markdown("### 📍 Progress Tracker")
    st.markdown(
        "<p style='color:#8892b0; font-size:0.78rem; margin-top:-10px;'>"
        "Scroll down to see each section.</p>",
        unsafe_allow_html=True
    )
    steps = {
        "Data Upload": st.session_state.data_loaded,
        "Model Arena": st.session_state.arena_complete,
        "Feature Importance": st.session_state.importance_complete,
        "Symbolic Regression": st.session_state.symbolic_complete,
        "Desmos Visualization": st.session_state.desmos_complete,
        "AI Explanation": False,
    }
    icons = {
        True: "✅",
        False: "⬜"
    }
    for step, done in steps.items():
        icon = "✅" if done else "⬜"
        color = "#4fc3f7" if done else "#8892b0"
        st.markdown(
            f"<p style='color:{color}; margin:4px 0;'>{icon} {step}</p>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "<small style='color:#8892b0'>Transforms black-box ML models into "
        "understandable mathematical relationships.</small>",
        unsafe_allow_html=True
    )

# =============================================================================
# MAIN HEADER
# =============================================================================

st.markdown("""
<div style='text-align:center; padding: 30px 0 10px 0;'>
    <h1 style='font-size:2.8rem; font-weight:800; color:#4fc3f7; letter-spacing:2px;'>
        🔬 Explainable AI Workbench
    </h1>
    <p style='color:#8892b0; font-size:1.1rem; margin-top:-10px;'>
        Transform black-box models into understandable mathematical relationships
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# PHASE 1 — SECTION 1: DATA UPLOAD
# =============================================================================

st.markdown("""
<div class='section-header'>
    <h2 style='margin:0; color:#e0e0e0;'>📂 Phase 1 — Data Upload</h2>
</div>
""", unsafe_allow_html=True)

# Two options: Upload CSV or Use Demo Dataset
upload_tab, demo_tab = st.tabs(["📤 Upload Your CSV", "🎮 Use Demo Dataset"])

with upload_tab:
    uploaded_file = st.file_uploader(
        "Upload your dataset (CSV format)",
        type=["csv"],
        help="Upload any CSV file. The app will auto-detect the problem type after you select the target column.",
        key=f"uploader_{st.session_state.uploader_key}"
    )

    if uploaded_file is not None:
        df, error = load_data(uploaded_file)
        if error:
            st.error(f"❌ {error}")
        else:
            # Only reset arena if this is a genuinely new file
            is_new_file = (st.session_state.dataset_name != uploaded_file.name)
            if is_new_file:
                st.session_state.arena_results = None
                st.session_state.trained_models = None
                st.session_state.data_splits = None
                st.session_state.selected_model_name = None
                st.session_state.arena_complete = False
                st.session_state.show_arena = False

            st.session_state.df_raw = df
            st.session_state.dataset_name = uploaded_file.name
            st.session_state.data_loaded = True
            st.session_state.is_demo = False
            st.success(f"✅ Dataset loaded successfully — **{uploaded_file.name}**")

    else:
        # No file uploaded — check if demo is active and show message
        if st.session_state.is_demo and st.session_state.data_loaded:
            st.markdown("""
            <div class='custom-info'>
                🎮 You are currently using a demo dataset.
                If you want to proceed with your own data, upload a CSV file below.
            </div>
            """, unsafe_allow_html=True)
            # Also show which demo is active
            st.info(f"✅ Dataset loaded successfully — **{st.session_state.dataset_name}**")

with demo_tab:
    st.markdown("Don't have a dataset? Use one of our built-in demos:")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("📈 Student Performance\n(Regression Demo)", use_container_width=True):
            df, name = load_demo_dataset("regression")
            st.session_state.df_raw = df
            st.session_state.dataset_name = name
            st.session_state.data_loaded = True
            st.session_state.is_demo = True
            st.session_state.uploader_key += 1
            # Reset arena for new dataset
            st.session_state.arena_results = None
            st.session_state.trained_models = None
            st.session_state.data_splits = None
            st.session_state.selected_model_name = None
            st.session_state.arena_complete = False
            st.session_state.show_arena = False
            st.rerun()  # ← force full rerun so all state updates reflect immediately

    with col2:
        if st.button("🏦 Loan Approval\n(Classification Demo)", use_container_width=True):
            df, name = load_demo_dataset("classification")
            st.session_state.df_raw = df
            st.session_state.dataset_name = name
            st.session_state.data_loaded = True
            st.session_state.is_demo = True
            st.session_state.uploader_key += 1
            # Reset arena for new dataset
            st.session_state.arena_results = None
            st.session_state.trained_models = None
            st.session_state.data_splits = None
            st.session_state.selected_model_name = None
            st.session_state.arena_complete = False
            st.session_state.show_arena = False
            st.rerun()  # ← force full rerun so all state updates reflect immediately

    # Show active demo status inside demo tab as well
    if st.session_state.is_demo and st.session_state.data_loaded:
        st.success(f"✅ Dataset loaded successfully — **{st.session_state.dataset_name}**")

# =============================================================================
# PHASE 1 — SECTION 2: TARGET COLUMN SELECTION & PROBLEM TYPE DETECTION
# Only shown after data is loaded
# =============================================================================

if st.session_state.data_loaded and st.session_state.df_raw is not None:

    df = st.session_state.df_raw
    st.markdown("---")

    st.markdown("""
    <div class='section-header'>
        <h2 style='margin:0; color:#e0e0e0;'>🎯 Target Column Selection</h2>
    </div>
    """, unsafe_allow_html=True)

    col_select, col_info = st.columns([1, 2])

    with col_select:
        target_column = st.selectbox(
            "Select your target column",
            options=df.columns.tolist(),
            help="This is the column your model will learn to predict.",
            index=len(df.columns) - 1  # Default to last column (common convention)
        )

    if target_column:
        # Reset arena if target column changed — prevents stale cached results
        if st.session_state.target_column != target_column:
            st.session_state.arena_results = None
            st.session_state.trained_models = None
            st.session_state.data_splits = None
            st.session_state.selected_model_name = None
            st.session_state.arena_complete = False
            st.session_state.show_arena = False

        st.session_state.target_column = target_column

        # Auto-detect problem type
        problem_type, reason, confidence = detect_problem_type(df, target_column)
        st.session_state.problem_type = problem_type

        with col_info:
            st.markdown("<br>", unsafe_allow_html=True)
            if problem_type == "regression":
                st.markdown(f"""
                <div style='margin-top:8px;'>
                    <span class='badge-regression'>📈 Regression Task Detected</span>
                    <p style='color:#8892b0; margin-top:10px; font-size:0.9rem;'>
                        {reason}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='margin-top:8px;'>
                    <span class='badge-classification'>🏷️ Classification Task Detected</span>
                    <p style='color:#8892b0; margin-top:10px; font-size:0.9rem;'>
                        {reason}
                    </p>
                </div>
                """, unsafe_allow_html=True)

            if confidence == "medium":
                st.markdown("""
                <div class='custom-warning'>
                    ⚠️ Detection confidence is medium. If this looks wrong, try selecting a different target column.
                </div>
                """, unsafe_allow_html=True)

    # =============================================================================
    # PHASE 1 — SECTION 3: DATA VALIDATION
    # =============================================================================

        st.markdown("---")
        st.markdown("""
        <div class='section-header'>
            <h2 style='margin:0; color:#e0e0e0;'>🔍 Data Validation</h2>
        </div>
        """, unsafe_allow_html=True)

        validation_report = validate_data(df)

        for warning in validation_report["warnings"]:
            st.markdown(f"""
            <div class='custom-warning'>⚠️ {warning}</div>
            """, unsafe_allow_html=True)

        for info in validation_report["info"]:
            st.markdown(f"""
            <div class='custom-info'>ℹ️ {info}</div>
            """, unsafe_allow_html=True)

        # Apply cleaning
        df_clean, cleaning_log = clean_data(df, target_column)
        st.session_state.df_clean = df_clean

        with st.expander("🧹 Data Cleaning Log", expanded=False):
            for log in cleaning_log:
                st.markdown(f"- {log}")

    # =============================================================================
    # PHASE 1 — SECTION 4: DATASET OVERVIEW METRICS
    # =============================================================================

        st.markdown("---")
        st.markdown("""
        <div class='section-header'>
            <h2 style='margin:0; color:#e0e0e0;'>📊 Dataset Overview</h2>
        </div>
        """, unsafe_allow_html=True)

        summary = get_dataset_summary(df, target_column)

        # Metric cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{summary['n_rows']:,}</div>
                <div class='metric-label'>Total Rows</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{summary['n_features']}</div>
                <div class='metric-label'>Feature Columns</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            missing_color = "#ff6b6b" if summary['total_missing'] > 0 else "#4fc3f7"
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value' style='color:{missing_color};'>{summary['total_missing']}</div>
                <div class='metric-label'>Missing Values</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{df[target_column].nunique()}</div>
                <div class='metric-label'>Unique Target Values</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    # =============================================================================
    # PHASE 1 — SECTION 5: DATA PREVIEW
    # =============================================================================

        preview_tab, stats_tab, target_tab = st.tabs([
            "👁️ Data Preview",
            "📐 Statistical Summary",
            "🎯 Target Distribution"
        ])

        with preview_tab:
            st.dataframe(
                df.head(10).style.highlight_null(color="#3d1a1a"),
                use_container_width=True
            )
            st.caption(f"Showing first 10 rows of {summary['n_rows']:,} total rows.")

        with stats_tab:
            if summary["statistics"]:
                stats_df = pd.DataFrame(summary["statistics"]).T
                stats_df.index.name = "Feature"
                stats_df = stats_df.reset_index()
                st.dataframe(
                    stats_df.style.format({
                        col: "{:.4f}" for col in stats_df.columns if col != "Feature" and col != "missing"
                    }),
                    use_container_width=True
                )
            else:
                st.info("No numeric columns found for statistical summary.")

        with target_tab:
            target_data = df[target_column].dropna()

            if summary["target_type"] == "categorical":
                # Bar chart for classification target
                value_counts = target_data.value_counts().reset_index()
                value_counts.columns = [target_column, "count"]

                fig = px.bar(
                    value_counts,
                    x=target_column,
                    y="count",
                    title=f"Class Distribution — {target_column}",
                    color="count",
                    color_continuous_scale="Blues",
                    template="plotly_dark"
                )
                fig.update_layout(
                    plot_bgcolor="#0e1117",
                    paper_bgcolor="#0e1117",
                    font_color="#e0e0e0",
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

            else:
                # Histogram for regression target
                fig = px.histogram(
                    df,
                    x=target_column,
                    nbins=40,
                    title=f"Target Distribution — {target_column}",
                    color_discrete_sequence=["#4fc3f7"],
                    template="plotly_dark",
                    marginal="box"
                )
                fig.update_layout(
                    plot_bgcolor="#0e1117",
                    paper_bgcolor="#0e1117",
                    font_color="#e0e0e0",
                    bargap=0.05
                )
                st.plotly_chart(fig, use_container_width=True)

                # Show key stats below chart
                dist = summary["target_distribution"]
                sc1, sc2, sc3, sc4, sc5 = st.columns(5)
                sc1.metric("Min", f"{dist['min']:.2f}")
                sc2.metric("Q1", f"{dist['q1']:.2f}")
                sc3.metric("Median", f"{dist['median']:.2f}")
                sc4.metric("Q3", f"{dist['q3']:.2f}")
                sc5.metric("Max", f"{dist['max']:.2f}")

    # ==========================================================================
    # PHASE 1 — PROCEED BUTTON
    # ==========================================================================

        st.markdown("---")
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            if st.button(
                "🚀 Proceed to Model Arena →",
                use_container_width=True,
                type="primary"
            ):
                # Explicitly save everything needed before rerun
                df_clean_now, _ = clean_data(df, target_column)
                st.session_state.df_clean = df_clean_now
                st.session_state.target_column = target_column
                pt, _, _ = detect_problem_type(df, target_column)
                st.session_state.problem_type = pt
                st.session_state.show_arena = True
                # Store encoding maps from raw data
                st.session_state.encoding_maps = get_encoding_maps(df, target_column)
                st.rerun()

# =============================================================================
# PHASE 2 — MODEL ARENA
# Only shown after user clicks Proceed to Model Arena
# =============================================================================

if (
    st.session_state.data_loaded
    and st.session_state.df_clean is not None
    and st.session_state.target_column is not None
    and st.session_state.problem_type is not None
    and st.session_state.show_arena
):
    st.markdown("---")
    st.markdown("""
    <div class='section-header'>
        <h2 style='margin:0; color:#e0e0e0;'>⚔️ Phase 2 — Model Arena</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "<p style='color:#8892b0;'>Training all models simultaneously and evaluating their performance. "
        "You will then choose which model to use for explainability analysis.</p>",
        unsafe_allow_html=True
    )

    # ── Train models (only if not already trained) ──────────────────────────
    if st.session_state.arena_results is None:
        with st.spinner("⚔️ Training all models in the arena... Please wait."):
            results, trained_models, data_splits = train_all_models(
                st.session_state.df_clean,
                st.session_state.target_column,
                st.session_state.problem_type
            )
            st.session_state.arena_results = results
            st.session_state.trained_models = trained_models
            st.session_state.data_splits = data_splits
        st.success("✅ All models trained successfully!")

    results      = st.session_state.arena_results
    problem_type = st.session_state.problem_type

    # ── Best model recommendation ────────────────────────────────────────────
    best_model_name = get_best_model(results, problem_type)
    st.markdown(f"""
    <div class='custom-info'>
        🏆 Best performing model based on primary metric:
        <strong>{best_model_name}</strong> —
        but the final choice is yours. Consider interpretability vs performance.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Comparison Table ─────────────────────────────────────────────────────
    st.markdown("### 📋 Model Comparison Table")
    df_comparison = get_comparison_dataframe(results, problem_type)
    st.dataframe(df_comparison, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Visual Comparison Charts ──────────────────────────────────────────────
    st.markdown("### 📊 Visual Comparison")

    model_names = [r["model_name"] for r in results]
    colors      = ["#4fc3f7", "#81c784", "#ffb74d"]

    if problem_type == "regression":
        chart_tab1, chart_tab2, chart_tab3 = st.tabs(["R²", "MAE", "RMSE"])

        with chart_tab1:
            fig = go.Figure(go.Bar(
                x=model_names,
                y=[r["R²"] for r in results],
                marker_color=colors,
                text=[f"{r['R²']:.4f}" for r in results],
                textposition="outside"
            ))
            fig.update_layout(
                title="R² Score (higher is better ↑)",
                yaxis_title="R²",
                template="plotly_dark",
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117",
                font_color="#e0e0e0",
                yaxis=dict(range=[0, 1.1])
            )
            st.plotly_chart(fig, use_container_width=True)

        with chart_tab2:
            fig = go.Figure(go.Bar(
                x=model_names,
                y=[r["MAE"] for r in results],
                marker_color=colors,
                text=[f"{r['MAE']:.4f}" for r in results],
                textposition="outside"
            ))
            fig.update_layout(
                title="Mean Absolute Error (lower is better ↓)",
                yaxis_title="MAE",
                template="plotly_dark",
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117",
                font_color="#e0e0e0"
            )
            st.plotly_chart(fig, use_container_width=True)

        with chart_tab3:
            fig = go.Figure(go.Bar(
                x=model_names,
                y=[r["RMSE"] for r in results],
                marker_color=colors,
                text=[f"{r['RMSE']:.4f}" for r in results],
                textposition="outside"
            ))
            fig.update_layout(
                title="Root Mean Squared Error (lower is better ↓)",
                yaxis_title="RMSE",
                template="plotly_dark",
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117",
                font_color="#e0e0e0"
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        chart_tab1, chart_tab2, chart_tab3, chart_tab4 = st.tabs(
            ["Accuracy", "Precision", "Recall", "F1 Score"]
        )

        metrics_map = {
            "Accuracy":  ("Accuracy (higher is better ↑)",   "Accuracy"),
            "Precision": ("Precision (higher is better ↑)",  "Precision"),
            "Recall":    ("Recall (higher is better ↑)",     "Recall"),
            "F1 Score":  ("F1 Score (higher is better ↑)",   "F1 Score"),
        }
        for tab, (metric_key, (title, ylabel)) in zip(
            [chart_tab1, chart_tab2, chart_tab3, chart_tab4],
            metrics_map.items()
        ):
            with tab:
                fig = go.Figure(go.Bar(
                    x=model_names,
                    y=[r[metric_key] for r in results],
                    marker_color=colors,
                    text=[f"{r[metric_key]:.4f}" for r in results],
                    textposition="outside"
                ))
                fig.update_layout(
                    title=title,
                    yaxis_title=ylabel,
                    template="plotly_dark",
                    plot_bgcolor="#0e1117",
                    paper_bgcolor="#0e1117",
                    font_color="#e0e0e0",
                    yaxis=dict(range=[0, 1.1])
                )
                st.plotly_chart(fig, use_container_width=True)

    # ── Model Selection ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 Select Your Model")
    st.markdown(
        "<p style='color:#8892b0;'>Choose the model whose predictions you want to explain. "
        "Higher accuracy isn't always the right choice — consider the interpretability tradeoff.</p>",
        unsafe_allow_html=True
    )

    selected_model = st.radio(
        "Which model do you want to proceed with?",
        options=model_names,
        index=model_names.index(best_model_name),  # default = best model
        horizontal=True
    )

    # Show interpretability warning instantly on selection
    if selected_model:
        warning_msg, warning_level = get_interpretability_warning(
            selected_model, problem_type
        )
        if warning_level == "info":
            st.info(warning_msg)
        else:
            st.warning(warning_msg)

    # ── Confirm Selection Button ──────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col_b1, col_b2, col_b3 = st.columns([1, 2, 1])
    with col_b2:
        if st.button(
            f"✅ Confirm — Use {selected_model} →",
            use_container_width=True,
            type="primary"
        ):
            st.session_state.selected_model_name = selected_model
            st.session_state.arena_complete = True
            # Reset importance if model changed
            st.session_state.importance_df = None
            st.session_state.top_features = None
            st.session_state.importance_complete = False
            st.session_state.show_importance = True
            st.rerun()

# =============================================================================
# PHASE 3 — FEATURE IMPORTANCE ENGINE
# =============================================================================

if (
    st.session_state.arena_complete
    and st.session_state.selected_model_name is not None
    and st.session_state.df_clean is not None
    and st.session_state.get("show_importance", False)
):
    st.markdown("---")
    st.markdown("""
    <div class='section-header'>
        <h2 style='margin:0; color:#e0e0e0;'>🔍 Phase 3 — Feature Importance Engine</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f"<p style='color:#8892b0;'>Computing permutation importance for "
        f"<strong style='color:#4fc3f7;'>{st.session_state.selected_model_name}</strong> — "
        f"measuring how much each feature contributes to predictions.</p>",
        unsafe_allow_html=True
    )

    # ── Compute importance (only if not already computed) ────────────────────
    if st.session_state.importance_df is None:
        with st.spinner("🔍 Computing permutation feature importance..."):
            model      = st.session_state.trained_models[st.session_state.selected_model_name]
            X_test     = st.session_state.data_splits["X_test"]
            y_test     = st.session_state.data_splits["y_test"]
            feat_names = st.session_state.data_splits["feature_names"]

            importance_df = compute_permutation_importance(
                model, X_test, y_test,
                feat_names,
                st.session_state.problem_type
            )
            st.session_state.importance_df = importance_df
        st.success("✅ Feature importance computed successfully!")

    importance_df = st.session_state.importance_df

    # ── Coverage Threshold Selector ───────────────────────────────────────────
    st.markdown("### ⚙️ Coverage Threshold")
    st.markdown(
        "<p style='color:#8892b0; font-size:0.9rem;'>Select how much of the model's "
        "behaviour you want to explain. Higher = more features included.</p>",
        unsafe_allow_html=True
    )

    # Use st.slider with step instead of select_slider to avoid jumping bug
    threshold = st.slider(
        "Cumulative Importance Coverage",
        min_value=70,
        max_value=100,
        value=st.session_state.importance_threshold,
        step=5,
        format="%d%%",
        key="threshold_slider"
    )
    st.session_state.importance_threshold = threshold

    # Recompute top features based on threshold
    top_features, n_selected = select_top_features(importance_df, threshold)
    st.session_state.top_features = top_features

    st.markdown(f"""
    <div class='custom-info'>
        ℹ️ <strong>{n_selected} feature(s)</strong> selected to explain
        <strong>{threshold}%</strong> of model behaviour:
        {', '.join([f'<strong>{f}</strong>' for f in top_features])}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature Importance Table ──────────────────────────────────────────────
    st.markdown("### 📋 Feature Importance Breakdown")

    display_df = importance_df[["rank", "feature", "importance_pct", "cumulative_pct", "std"]].copy()
    display_df.columns = ["Rank", "Feature", "Importance %", "Cumulative %", "Std Dev"]
    display_df["In Top Features"] = display_df["Feature"].apply(
        lambda x: "✅ Selected" if x in top_features else "—"
    )
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature Importance Bar Chart ─────────────────────────────────────────
    st.markdown("### 📊 Feature Importance Chart")

    bar_colors = [
        "#4fc3f7" if f in top_features else "#455a64"
        for f in importance_df["feature"]
    ]

    fig_imp = go.Figure(go.Bar(
        x=importance_df["importance_pct"],
        y=importance_df["feature"],
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:.1f}%" for v in importance_df["importance_pct"]],
        textposition="outside",
        error_x=dict(
            type="data",
            array=importance_df["std"],
            visible=True,
            color="#8892b0"
        )
    ))
    fig_imp.update_layout(
        title=f"Permutation Feature Importance — {st.session_state.selected_model_name}",
        xaxis_title="Importance (%)",
        yaxis_title="Feature",
        template="plotly_dark",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#e0e0e0",
        yaxis=dict(autorange="reversed"),
        height=max(300, len(importance_df) * 55)
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown(
        "<p style='color:#8892b0; font-size:0.85rem;'>"
        "🔵 Blue bars = selected for explanation &nbsp;|&nbsp; "
        "Grey bars = below threshold &nbsp;|&nbsp; "
        "Error bars = stability across permutation repeats</p>",
        unsafe_allow_html=True
    )

    # ── Axis Feature Selection ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎯 Select Visualization Axes")
    st.markdown(
        "<p style='color:#8892b0;'>Choose <strong>2 features</strong> from the top selected features "
        "to use as the X and Y axes in the Desmos visualization. "
        "Remaining features will be anchored at their median values with interactive sliders.</p>",
        unsafe_allow_html=True
    )

    if len(top_features) < 2:
        st.error("❌ Not enough features selected. Lower the coverage threshold to include more features.")
    else:
        col_ax1, col_ax2 = st.columns(2)

        with col_ax1:
            axis_1 = st.selectbox(
                "📐 X-Axis Feature",
                options=top_features,
                index=0,
                key="axis1_select"
            )

        with col_ax2:
            axis_2_options = [f for f in top_features if f != axis_1]
            axis_2 = st.selectbox(
                "📐 Y-Axis Feature",
                options=axis_2_options,
                index=0,
                key="axis2_select"
            )

        st.session_state.axis_feature_1 = axis_1
        st.session_state.axis_feature_2 = axis_2

        # ── Remaining Features Sliders ────────────────────────────────────────
        remaining_features = [
            f for f in st.session_state.data_splits["feature_names"]
            if f != axis_1 and f != axis_2
        ]

        # Get encoding maps stored during data loading
        encoding_maps = st.session_state.encoding_maps
        df_clean = st.session_state.df_clean

        if remaining_features:
            st.markdown("---")
            st.markdown("### 🎛️ Anchor Remaining Features")
            st.markdown(
                "<p style='color:#8892b0;'>These features are held at fixed values while "
                "the graph visualizes the relationship between your two selected axes. "
                "Adjust them to explore different conditions.</p>",
                unsafe_allow_html=True
            )

            anchored_values = {}
            cols_per_row = 3
            for i in range(0, len(remaining_features), cols_per_row):
                row_features = remaining_features[i:i + cols_per_row]
                cols = st.columns(len(row_features))

                for col, feature in zip(cols, row_features):
                    with col:
                        if feature in encoding_maps:
                            # ── Encoded categorical feature ──────────────────
                            enc_map = encoding_maps[feature]
                            # options as integers: [0, 1, 2]
                            options = sorted(enc_map.keys())
                            # median index as default
                            default_idx = len(options) // 2

                            selected_val = st.select_slider(
                                label=f"**{feature}**",
                                options=options,
                                value=options[default_idx],
                                format_func=lambda v, m=enc_map: f"{v}  ({m[v]})",
                                key=f"slider_{feature}"
                            )

                        else:
                            # ── Continuous numeric feature ───────────────────
                            anchors, anchor_labels, anchor_values = get_slider_anchors(
                                df_clean,
                                feature,
                                st.session_state.target_column
                            )
                            default_idx = (
                                anchor_labels.index("Median")
                                if "Median" in anchor_labels
                                else len(anchor_labels) // 2
                            )
                            selected_val = st.select_slider(
                                label=f"**{feature}**",
                                options=anchor_values,
                                value=anchor_values[default_idx],
                                format_func=lambda v, labels=anchor_labels, values=anchor_values: (
                                    f"{labels[values.index(v)]}  ({v})"
                                ),
                                key=f"slider_{feature}"
                            )

                        anchored_values[feature] = selected_val

            st.session_state.anchored_values = anchored_values

            # Show current anchored values summary
            with st.expander("📌 Current Anchor Values Summary", expanded=False):
                anchor_rows = []
                for k, v in anchored_values.items():
                    if k in encoding_maps:
                        display_val = f"{v}  ({encoding_maps[k][v]})"
                    else:
                        display_val = str(v)
                    anchor_rows.append({"Feature": k, "Anchored At": display_val})
                st.dataframe(
                    pd.DataFrame(anchor_rows),
                    use_container_width=True,
                    hide_index=True
                )

        # ── Proceed Button ────────────────────────────────────────────────────
        st.markdown("---")
        col_f1, col_f2, col_f3 = st.columns([1, 2, 1])
        with col_f2:
            if st.button(
                "🧠 Proceed to Symbolic Regression →",
                use_container_width=True,
                type="primary"
            ):
                st.session_state.importance_complete = True
                st.session_state.show_symbolic = True
                st.success(
                    f"✅ Features locked in! "
                    f"X-Axis: **{axis_1}** | Y-Axis: **{axis_2}**"
                )
                st.rerun()

        # Mark importance as complete as soon as axes are selected
        # (not just when proceed is clicked)
        if axis_1 and axis_2:
            st.session_state.importance_complete = True

# =============================================================================
# PHASE 4 — SYMBOLIC REGRESSION
# =============================================================================

if (
    st.session_state.importance_complete
    and st.session_state.axis_feature_1 is not None
    and st.session_state.axis_feature_2 is not None
    and st.session_state.get("show_symbolic", False)
):
    st.markdown("---")
    st.markdown("""
    <div class='section-header'>
        <h2 style='margin:0; color:#e0e0e0;'>🧠 Phase 4 — Symbolic Regression</h2>
    </div>
    """, unsafe_allow_html=True)

    ax1 = st.session_state.axis_feature_1
    ax2 = st.session_state.axis_feature_2

    st.markdown(
        f"<p style='color:#8892b0;'>Discovering a mathematical equation that approximates "
        f"<strong style='color:#4fc3f7;'>{st.session_state.selected_model_name}</strong>'s "
        f"predictions using features "
        f"<strong style='color:#4fc3f7;'>{ax1}</strong> and "
        f"<strong style='color:#4fc3f7;'>{ax2}</strong>.</p>",
        unsafe_allow_html=True
    )

    # ── Settings ─────────────────────────────────────────────────────────────
    st.markdown("### ⚙️ Symbolic Regression Settings")
    set_col1, set_col2 = st.columns(2)

    with set_col1:
        compute_mode = st.radio(
            "🚀 Compute Mode",
            options=list(COMPUTE_MODES.keys()),
            index=1,  # default: Balanced
            key="compute_mode_select"
        )
        mode_info = COMPUTE_MODES[compute_mode]
        st.markdown(
            f"<p style='color:#8892b0; font-size:0.85rem;'>"
            f"{mode_info['description']}<br>"
            f"⏱️ Estimated time: <strong>{mode_info['estimated_time']}</strong></p>",
            unsafe_allow_html=True
        )

    with set_col2:
        complexity_mode = st.radio(
            "📐 Equation Complexity Preference",
            options=list(COMPLEXITY_MODES.keys()),
            index=1,  # default: Balanced
            key="complexity_mode_select"
        )
        comp_info = COMPLEXITY_MODES[complexity_mode]
        st.markdown(
            f"<p style='color:#8892b0; font-size:0.85rem;'>"
            f"{comp_info['description']}</p>",
            unsafe_allow_html=True
        )

    # ── Cache key — unique per feature pair + settings ────────────────────────
    cache_key = f"{ax1}__{ax2}__{compute_mode}__{complexity_mode}"

    # ── Run PySR Button + readiness indicator ─────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)

    waiting_for        = st.session_state.get("pysr_waiting_for")
    current_thr_status = get_thread_status(cache_key)
    is_running         = (
        waiting_for == cache_key
        and current_thr_status in ["starting", "running"]
    )

    if not st.session_state.julia_warmed:
        st.markdown("""
        <div class='custom-warning'>
            ⚠️ <strong>Julia is still initializing.</strong>
            Please wait — the banner at the top will disappear when ready.
        </div>
        """, unsafe_allow_html=True)
    elif not is_running:
        st.markdown(f"""
        <div class='custom-info'>
            ✅ <strong>Julia + PySR engine is ready.</strong> &nbsp;
            Quick: ~30–90 sec &nbsp;|&nbsp;
            Balanced: ~1–3 min &nbsp;|&nbsp;
            Deep Search: ~3–5 min (hard timeout at 5 min).
        </div>
        """, unsafe_allow_html=True)

    col_r1, col_r2, col_r3 = st.columns([1, 2, 1])
    with col_r2:
        run_button = st.button(
            "🔬 Discover Equation",
            use_container_width=True,
            type="primary",
            key="run_pysr_btn",
            disabled=is_running   # Prevent double-clicking while running
        )

    # ── STEP A: Button clicked → prepare data, launch background thread ───────
    if run_button:
        # Clear any previous result for this exact cache key
        if cache_key in st.session_state.pysr_cache:
            del st.session_state.pysr_cache[cache_key]
        clear_thread(cache_key)

        # Prepare training data (fast, no Julia involved)
        with st.spinner("📊 Preparing training data from model predictions..."):
            X_pysr, y_pysr = prepare_pysr_data(
                df_clean       = st.session_state.df_clean,
                target_column  = st.session_state.target_column,
                model          = st.session_state.trained_models[
                                     st.session_state.selected_model_name],
                feature_names  = st.session_state.data_splits["feature_names"],
                axis_feature_1 = ax1,
                axis_feature_2 = ax2,
                anchored_values= st.session_state.anchored_values,
                problem_type   = st.session_state.problem_type
            )

        # Persist X and y so the polling loop can compute fidelity when done
        st.session_state.pysr_pending_X = X_pysr
        st.session_state.pysr_pending_y = y_pysr

        # Launch PySR in a daemon background thread
        start_pysr_thread(X_pysr, y_pysr, compute_mode, complexity_mode, cache_key)
        st.session_state.pysr_waiting_for = cache_key
        st.rerun()

    # ── STEP B: Thread is running → show live progress, poll every 3s ─────────
    waiting_for    = st.session_state.get("pysr_waiting_for")
    thread_status  = get_thread_status(cache_key)

    if waiting_for == cache_key and thread_status in ["starting", "running"]:
        elapsed   = get_elapsed_seconds(cache_key)
        timeout   = COMPUTE_MODES[compute_mode]["timeout_in_seconds"]
        remaining = max(0, timeout - elapsed)
        progress  = min(100, int(elapsed / timeout * 100))

        # Phase-specific message based on elapsed time
        if elapsed < 15:
            phase_msg = "Initializing Julia engine and compiling equation grammar..."
        elif elapsed < 40:
            phase_msg = "Exploring initial population of candidate equations..."
        elif elapsed < 90:
            phase_msg = "Evolving equations through genetic programming operators..."
        else:
            phase_msg = (
                f"Refining best candidates across populations... "
                f"Auto-stop in {remaining}s."
            )

        st.markdown(f"""
        <div style='background:#1a1f35; border:1px solid #4fc3f7;
                    border-radius:10px; padding:20px 24px; margin:12px 0;'>
            <div style='display:flex; align-items:center; gap:12px; margin-bottom:10px;'>
                <span style='font-size:1.5rem;'>🔬</span>
                <span style='color:#4fc3f7; font-weight:700; font-size:1.05rem;'>
                    PySR Equation Search In Progress
                </span>
            </div>
            <div style='color:#e0e0e0; font-size:0.92rem; margin-bottom:14px;'>
                {phase_msg}
            </div>
            <div style='display:flex; justify-content:space-between; margin-bottom:10px;'>
                <span style='color:#8892b0; font-size:0.85rem;'>
                    ⏱️ Elapsed: <strong style='color:#4fc3f7;'>{elapsed}s</strong>
                </span>
                <span style='color:#8892b0; font-size:0.85rem;'>
                    Mode: <strong style='color:#4fc3f7;'>{compute_mode}</strong>
                </span>
                <span style='color:#8892b0; font-size:0.85rem;'>
                    Hard cap: <strong style='color:#4fc3f7;'>{timeout}s</strong>
                </span>
            </div>
            <div style='background:#0e1117; border-radius:6px;
                        height:6px; overflow:hidden;'>
                <div style='background:linear-gradient(90deg,#4fc3f7,#0288d1);
                            height:6px; width:{progress}%;
                            transition:width 0.5s ease;'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner(
            f"⏳ Julia + PySR running — {compute_mode} / {complexity_mode} — "
            f"please wait... ({elapsed}s elapsed)"
        ):
            st.info(
                "🔒 Julia is running in the background. "
                "Do not close or refresh this tab. "
                "This page updates automatically every 3 seconds."
            )
            time.sleep(3)   # 3-second poll interval

        st.rerun()

    # ── STEP C: Thread complete → collect result, compute fidelity, cache ──────
    elif waiting_for == cache_key and thread_status == "complete":
        with st.spinner("🎯 Equation found — computing fidelity and formatting..."):
            raw_model, raw_eq, raw_all_eqs = get_thread_result(cache_key)

            X_pysr = st.session_state.pysr_pending_X
            y_pysr = st.session_state.pysr_pending_y

            fidelity, y_surrogate, mae = compute_fidelity(
                raw_model, None, X_pysr, y_pysr
            )
            formatted_eq, desmos_eq = format_equation(raw_eq, ax1, ax2)
            complexity_info         = get_equation_complexity(raw_model)

            # Store everything in the per-pair cache
            st.session_state.pysr_cache[cache_key] = {
                "pysr_model":      raw_model,
                "best_equation":   formatted_eq,
                "desmos_equation": desmos_eq,
                "fidelity_score":  fidelity,
                "mae":             mae,
                "all_equations":   raw_all_eqs,
                "complexity_info": complexity_info,
                "X_pysr":          X_pysr,
                "y_pysr":          y_pysr,
                "y_surrogate":     y_surrogate,
            }

            # Top-level session state for Desmos phase
            st.session_state.pysr_model        = raw_model
            st.session_state.best_equation     = formatted_eq
            st.session_state.desmos_equation   = desmos_eq
            st.session_state.fidelity_score    = fidelity
            st.session_state.symbolic_complete = True

            # Clean up thread state
            clear_thread(cache_key)
            st.session_state.pysr_waiting_for = None

        st.success("✅ Equation discovered successfully!")
        st.rerun()

    # ── STEP D: Thread error → report and clean up ────────────────────────────
    elif waiting_for == cache_key and thread_status == "error":
        error_msg = get_thread_error(cache_key)
        clear_thread(cache_key)
        st.session_state.pysr_waiting_for = None
        st.error(f"❌ PySR encountered an error: {error_msg}")
        st.info(
            "💡 Try Quick mode first. "
            "If this is the first run after an app restart, "
            "Julia needs 30–60 s to re-initialize — wait and try again."
        )

    # ── STEP E: Display cached result ─────────────────────────────────────────
    if (
        cache_key in st.session_state.pysr_cache
        and st.session_state.get("pysr_waiting_for") != cache_key
    ):
        cached = st.session_state.pysr_cache[cache_key]

        st.markdown("---")
        st.markdown("### 📐 Discovered Equation")

        # Equation display box
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#1a1f35,#1e2540);
                    border:1px solid #4fc3f7;
                    border-radius:12px;
                    padding:24px 32px;
                    text-align:center;
                    margin:16px 0;'>
            <p style='color:#8892b0; font-size:0.85rem; margin-bottom:8px;'>
                Surrogate Equation
            </p>
            <p style='color:#4fc3f7; font-size:1.5rem; font-weight:700;
                      font-family:monospace; letter-spacing:1px;'>
                f({ax1}, {ax2}) = {cached['best_equation']}
            </p>
            <p style='color:#8892b0; font-size:0.8rem; margin-top:12px;'>
                Holding remaining features at anchored values
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Fidelity Score ────────────────────────────────────────────────
        st.markdown("### 🎯 Fidelity Score")
        fidelity = cached["fidelity_score"]
        rating, color, message = get_fidelity_rating(fidelity)

        f_col1, f_col2, f_col3 = st.columns([1, 2, 1])
        with f_col2:
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#1a1f35,#1e2540);
                        border:2px solid {color};
                        border-radius:12px;
                        padding:24px;
                        text-align:center;'>
                <div style='font-size:3rem; font-weight:800; color:{color};'>
                    {fidelity}%
                </div>
                <div style='font-size:1.3rem; font-weight:600; color:{color};
                            margin-top:4px;'>
                    {rating}
                </div>
                <div style='color:#8892b0; font-size:0.9rem; margin-top:8px;'>
                    {message}
                </div>
                <div style='color:#8892b0; font-size:0.85rem; margin-top:8px;'>
                    MAE between surrogate and model: <strong>{cached['mae']}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Fidelity Scale Reference ──────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📊 Fidelity Scale Reference", expanded=False):
            fidelity_scale = pd.DataFrame([
                {"Score Range": "95% and above", "Rating": "🟢 Excellent",
                 "Meaning": "Surrogate closely mirrors the model"},
                {"Score Range": "85% – 94%",     "Rating": "🔵 Good",
                 "Meaning": "Strong approximation"},
                {"Score Range": "70% – 84%",     "Rating": "🟡 Fair",
                 "Meaning": "Captures main trend, misses complexity"},
                {"Score Range": "50% – 69%",     "Rating": "🟠 Weak",
                 "Meaning": "Rough approximation — try Balanced mode"},
                {"Score Range": "Below 50%",     "Rating": "🔴 Unreliable",
                 "Meaning": "Poor fit — try Deep Search or different features"},
            ])
            st.dataframe(fidelity_scale, use_container_width=True, hide_index=True)

        # ── Equation Complexity Info ──────────────────────────────────────
        ci = cached["complexity_info"]
        st.markdown("<br>", unsafe_allow_html=True)
        ci_col1, ci_col2, ci_col3 = st.columns(3)
        with ci_col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{ci['complexity']}</div>
                <div class='metric-label'>Equation Complexity</div>
            </div>""", unsafe_allow_html=True)
        with ci_col2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{ci['loss']:.4f}</div>
                <div class='metric-label'>PySR Loss</div>
            </div>""", unsafe_allow_html=True)
        with ci_col3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{fidelity}%</div>
                <div class='metric-label'>Fidelity Score</div>
            </div>""", unsafe_allow_html=True)

        # ── All Discovered Equations Table + Manual Selector ─────────────
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📋 All Discovered Equations (PySR Pareto Front)", expanded=False):
            st.markdown(
                "<p style='color:#8892b0; font-size:0.85rem;'>"
                "PySR returns a Pareto front — equations at different "
                "complexity vs accuracy tradeoffs. The best equation "
                "is automatically selected based on your complexity preference. "
                "You can also manually select any equation from the table below.</p>",
                unsafe_allow_html=True
            )
            try:
                eq_df = cached["all_equations"][
                    ["complexity", "loss", "equation"]
                ].copy()
                eq_df.columns = ["Complexity", "Loss", "Equation"]
                st.dataframe(eq_df, use_container_width=True, hide_index=True)

                # ── Manual Equation Selection ─────────────────────────────
                st.markdown("---")
                st.markdown(
                    "**🎯 Select Equation Manually**",
                )
                st.markdown(
                    "<p style='color:#8892b0; font-size:0.85rem;'>"
                    "Want to use a specific equation from the table above? "
                    "Select it by complexity level. Higher complexity = "
                    "more accurate but harder to interpret.</p>",
                    unsafe_allow_html=True
                )

                complexity_options = eq_df["Complexity"].tolist()
                equation_options  = eq_df["Equation"].tolist()
                loss_options      = eq_df["Loss"].tolist()

                # Build display labels for the radio buttons
                radio_labels = [
                    f"Complexity {c}  |  Loss {l:.4f}  →  {e[:60]}{'...' if len(str(e)) > 60 else ''}"
                    for c, l, e in zip(complexity_options, loss_options, equation_options)
                ]

                # Find the currently active equation's index
                current_eq = cached["best_equation"]
                try:
                    current_idx = equation_options.index(current_eq)
                except ValueError:
                    current_idx = len(equation_options) - 1

                selected_label = st.radio(
                    "Choose equation:",
                    options=radio_labels,
                    index=current_idx,
                    key="manual_eq_select"
                )

                selected_idx = radio_labels.index(selected_label)
                selected_eq  = str(equation_options[selected_idx])
                selected_complexity = complexity_options[selected_idx]
                selected_loss = loss_options[selected_idx]

                # Apply button
                if st.button(
                    "✅ Use This Equation",
                    key="apply_manual_eq",
                    use_container_width=True
                ):
                    # Format the manually selected equation for Desmos
                    formatted_eq, desmos_eq = format_equation(
                        selected_eq, ax1, ax2
                    )
                    # Update the cache with manually selected equation
                    st.session_state.pysr_cache[cache_key]["best_equation"]   = formatted_eq
                    st.session_state.pysr_cache[cache_key]["desmos_equation"] = desmos_eq
                    st.session_state.pysr_cache[cache_key]["complexity_info"]["complexity"] = int(selected_complexity)
                    st.session_state.pysr_cache[cache_key]["complexity_info"]["loss"] = float(selected_loss)
                    st.session_state.best_equation   = formatted_eq
                    st.session_state.desmos_equation = desmos_eq
                    st.success(
                        f"✅ Equation updated! "
                        f"Now using complexity {selected_complexity} equation."
                    )
                    st.rerun()

            except Exception:
                st.info("Equation table not available for this run.")

        # ── Desmos Equation Preview ───────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🔗 Desmos-Ready Equation")
        st.markdown(
            "<p style='color:#8892b0; font-size:0.9rem;'>"
            "This is the equation formatted for Desmos visualization "
            "(variables renamed to x and y):</p>",
            unsafe_allow_html=True
        )
        st.code(f"f(x, y) = {cached['desmos_equation']}", language="latex")
        st.caption(
            f"Where x = {ax1} and y = {ax2}"
        )

        # ── Info message ──────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class='custom-info'>
            💡 <strong>Want a different equation?</strong>
            Change the compute mode or complexity preference above
            and click "Discover Equation" again for a fresh search.
            Or open the Pareto Front table above to manually pick
            any equation by complexity level.
        </div>
        """, unsafe_allow_html=True)

        # ── Proceed Button ────────────────────────────────────────────────
        st.markdown("---")
        col_p1, col_p2, col_p3 = st.columns([1, 2, 1])
        with col_p2:
            if st.button(
                "🎨 Proceed to Desmos Visualization →",
                use_container_width=True,
                type="primary",
                key="proceed_desmos_btn"
            ):
                st.session_state.symbolic_complete = True
                st.session_state.show_desmos = True
                st.success("✅ Equation locked in! Proceeding to Desmos Visualization.")
                st.rerun()

# =============================================================================
# PHASE 5 — DESMOS VISUALIZATION
# =============================================================================

if (
    st.session_state.symbolic_complete
    and st.session_state.best_equation is not None
    and st.session_state.desmos_equation is not None
    and st.session_state.get("show_desmos", False)
):
    st.markdown("---")
    st.markdown("""
    <div class='section-header'>
        <h2 style='margin:0; color:#e0e0e0;'>🎨 Phase 5 — Interactive Desmos Visualization</h2>
    </div>
    """, unsafe_allow_html=True)

    ax1 = st.session_state.axis_feature_1
    ax2 = st.session_state.axis_feature_2

    st.markdown(
        f"<p style='color:#8892b0;'>Visualizing the surrogate equation as a live interactive graph. "
        f"The curve shows how <strong style='color:#4fc3f7;'>{ax1}</strong> "
        f"affects the predicted value. "
        f"Drag the <strong style='color:#81c784;'>a</strong> slider inside the graph "
        f"to instantly explore different values of "
        f"<strong style='color:#81c784;'>{ax2}</strong> — no recomputation needed.</p>",
        unsafe_allow_html=True
    )

    # ── Equation reminder ─────────────────────────────────────────────────────
    best_eq   = st.session_state.best_equation
    desmos_eq = st.session_state.desmos_equation
    fidelity  = st.session_state.fidelity_score

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#1a1f35,#1e2540);
                border:1px solid #4fc3f7; border-radius:10px;
                padding:16px 24px; margin-bottom:16px; text-align:center;'>
        <p style='color:#8892b0; font-size:0.8rem; margin-bottom:6px;'>
            Surrogate Equation Being Visualized
        </p>
        <p style='color:#4fc3f7; font-size:1.05rem; font-weight:600;
                  font-family:monospace; word-break:break-all;'>
            f({ax1}, {ax2}) = {best_eq}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Get axis feature bounds ───────────────────────────────────────────────
    df_clean     = st.session_state.df_clean
    ax1_min, ax1_max = get_feature_bounds(df_clean, ax1)
    ax2_min, ax2_max = get_feature_bounds(df_clean, ax2)

    # Current value of axis_feature_2 — from anchored_values or median
    ax2_current = float(
        st.session_state.anchored_values.get(
            ax2,
            float(df_clean[ax2].median())
        )
    )

    # ── Streamlit-native legend (since HTML component is now minimal) ─────────
    leg1, leg2, leg3, leg4 = st.columns(4)
    with leg1:
        st.markdown(
            "<div style='background:#1a1f35; border:1px solid #2d3154; "
            "border-radius:8px; padding:8px 14px; text-align:center;'>"
            "<span style='color:#4fc3f7;'>●</span> "
            "<span style='color:#8892b0; font-size:0.82rem;'>Surrogate Equation</span>"
            "</div>", unsafe_allow_html=True
        )
    with leg2:
        st.markdown(
            f"<div style='background:#1a1f35; border:1px solid #2d3154; "
            f"border-radius:8px; padding:8px 14px; text-align:center;'>"
            f"<span style='color:#ff9800;'>●</span> "
            f"<span style='color:#8892b0; font-size:0.82rem;'>x = <strong style='color:#e0e0e0;'>{ax1}</strong></span>"
            f"</div>", unsafe_allow_html=True
        )
    with leg3:
        st.markdown(
            f"<div style='background:#1a1f35; border:1px solid #2d3154; "
            f"border-radius:8px; padding:8px 14px; text-align:center;'>"
            f"<span style='color:#81c784;'>●</span> "
            f"<span style='color:#8892b0; font-size:0.82rem;'>slider a = <strong style='color:#e0e0e0;'>{ax2}</strong></span>"
            f"</div>", unsafe_allow_html=True
        )
    with leg4:
        st.markdown(
            f"<div style='background:#1a1f35; border:1px solid #4caf50; "
            f"border-radius:8px; padding:8px 14px; text-align:center;'>"
            f"<span style='color:#4caf50; font-size:0.82rem;'>✅ Fidelity: {fidelity}%</span>"
            f"</div>", unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Build and render Desmos component ─────────────────────────────────────
    desmos_html = build_desmos_html(
        desmos_equation   = desmos_eq,
        axis_feature_1    = ax1,
        axis_feature_2    = ax2,
        ax2_current_value = ax2_current,
        ax2_min           = ax2_min,
        ax2_max           = ax2_max,
        ax1_min           = ax1_min,
        ax1_max           = ax1_max,
        fidelity_score    = fidelity,
        height            = 500
    )

    # height=510 gives 500px calculator + 10px iframe buffer
    components.html(desmos_html, height=510, scrolling=False)

    st.caption(
        f"x = {ax1}  |  a = {ax2} (current: {round(ax2_current, 2)})  |  "
        f"y = Predicted Value  —  Drag slider 'a' in the expression panel to explore interactively →"
    )

    # ── How to use cards ──────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("""
        <div class='metric-card'>
            <div style='font-size:1.6rem;'>🖱️</div>
            <div style='color:#4fc3f7; font-weight:600; margin:8px 0 4px 0;'>
                Drag Slider 'a'
            </div>
            <div class='metric-label'>
                In the left expression panel, drag the 'a' slider
                to change the value of the second feature instantly.
                The curve updates in real time — no recomputation.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with g2:
        st.markdown("""
        <div class='metric-card'>
            <div style='font-size:1.6rem;'>🔍</div>
            <div style='color:#4fc3f7; font-weight:600; margin:8px 0 4px 0;'>
                Zoom and Pan
            </div>
            <div class='metric-label'>
                Scroll to zoom in or out.
                Click and drag on the graph area to pan
                across the mathematical relationship.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with g3:
        st.markdown("""
        <div class='metric-card'>
            <div style='font-size:1.6rem;'>🔄</div>
            <div style='color:#4fc3f7; font-weight:600; margin:8px 0 4px 0;'>
                Recompute Surrogate
            </div>
            <div class='metric-label'>
                To explore different anchor values, go back to
                Phase 4 sliders, adjust them, then click
                Discover Equation again for a fresh surrogate.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Fixed feature values reminder ─────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    anchored      = st.session_state.anchored_values
    encoding_maps = st.session_state.get("encoding_maps", {})

    if anchored:
        with st.expander("📌 Features Fixed During Equation Discovery", expanded=False):
            st.markdown(
                "<p style='color:#8892b0; font-size:0.85rem;'>"
                "These features were held at fixed values when PySR discovered the equation. "
                "They appear as numeric constants in the curve. "
                "To explore different fixed values, go back to Phase 4 sliders "
                "and rerun equation discovery.</p>",
                unsafe_allow_html=True
            )
            anchor_rows = []
            for feat, val in anchored.items():
                if feat in encoding_maps:
                    enc_label   = encoding_maps[feat].get(int(val), str(val))
                    display_val = f"{val}  ({enc_label})"
                else:
                    display_val = str(val)
                anchor_rows.append({"Feature": feat, "Fixed At": display_val})

            st.dataframe(
                pd.DataFrame(anchor_rows),
                use_container_width=True,
                hide_index=True
            )

    # ── Scientific interpretation note ────────────────────────────────────────
    st.markdown(f"""
    <div class='custom-info'>
        🔬 <strong>How to read this graph:</strong>
        The curve shows how <strong>{ax1}</strong> (x-axis) affects
        the predicted value (y-axis) while <strong>{ax2}</strong>
        is held at the slider value <strong>a</strong>.
        <br><br>
        A <em>straight line</em> = linear (proportional) relationship. &nbsp;
        A <em>curve</em> = diminishing or accelerating returns. &nbsp;
        An <em>S-shape</em> = threshold effect. &nbsp;
        <em>Oscillation</em> = periodic mathematical influence.
    </div>
    """, unsafe_allow_html=True)

    # ── Proceed to Gemini AI Explanation ──────────────────────────────────────
    st.markdown("---")
    col_d1, col_d2, col_d3 = st.columns([1, 2, 1])
    with col_d2:
        if st.button(
            "🤖 Proceed to AI Explanation →",
            use_container_width=True,
            type="primary",
            key="proceed_gemini_btn"
        ):
            st.session_state.desmos_complete = True
            st.session_state.show_gemini    = True
            st.rerun()

# =============================================================================
# EMPTY STATE — No data loaded yet
# =============================================================================

else:
    if not st.session_state.data_loaded:
        st.markdown("""
        <div style='text-align:center; padding: 60px 20px; color:#8892b0;'>
            <div style='font-size:4rem;'>📂</div>
            <h3 style='color:#4fc3f7;'>Upload a dataset or choose a demo to get started</h3>
            <p>Supported format: CSV files</p>
        </div>
        """, unsafe_allow_html=True)