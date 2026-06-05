# =============================================================================
# app.py
# Explainable AI Workbench — Main Application Entry Point
# Day 3: Model Arena Module
# =============================================================================

import streamlit as st
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
    clean_data
)
from src.model_arena import (
    train_all_models,
    get_comparison_dataframe,
    get_interpretability_warning,
    get_best_model
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

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("## 🔬 XAI Workbench")
    st.markdown("---")

    # Progress tracker
    st.markdown("### 📍 Progress")
    steps = {
        "✅ Data Upload": st.session_state.data_loaded,
        "⚙️ Model Arena": st.session_state.arena_complete,
        "⬜ Feature Importance": False,
        "⬜ Symbolic Regression": False,
        "⬜ Desmos Visualization": False,
        "⬜ AI Explanation": False,
    }
    for step, done in steps.items():
        if done:
            st.markdown(f"<span style='color:#4fc3f7'>{step}</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color:#8892b0'>{step}</span>", unsafe_allow_html=True)

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
            st.success(
                f"✅ **{selected_model}** selected! "
                f"Feature Importance analysis is coming in Day 4."
            )
            st.balloons()

# =============================================================================
# EMPTY STATE — No data loaded yet
# =============================================================================

else:
    st.markdown("""
    <div style='text-align:center; padding: 60px 20px; color:#8892b0;'>
        <div style='font-size:4rem;'>📂</div>
        <h3 style='color:#4fc3f7;'>Upload a dataset or choose a demo to get started</h3>
        <p>Supported format: CSV files</p>
    </div>
    """, unsafe_allow_html=True)
