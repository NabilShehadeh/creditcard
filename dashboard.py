#!/usr/bin/env python3
"""
Interactive Credit Card Fraud Detection Dashboard.

Uses data from creditcard.csv with heatmap, scatter plot, and sliders.
Black and creme theme. Integrates with repo: DataProcessor, feature stats, model metrics.
"""

import sys
from pathlib import Path

# Ensure project root and src are on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Repo integrations
try:
    from data.processor import DataProcessor
    from features.engineering import FeatureEngineer
    REPO_AVAILABLE = True
except Exception:
    REPO_AVAILABLE = False

# ----- Theme: black and white, high contrast -----
WHITE = "#ffffff"
BLACK = "#000000"
GREY_DARK = "#333333"
GREY_MID = "#666666"
GREY_LIGHT = "#cccccc"
GREY_GRID = "#e0e0e0"

# Dict for fig.update_layout(**PLOTLY_LAYOUT) — Plotly expects mapping, not Layout
PLOTLY_LAYOUT = {
    "paper_bgcolor": WHITE,
    "plot_bgcolor": WHITE,
    "font": {"color": BLACK, "family": "Inter, system-ui, sans-serif"},
    "title_font": {"color": BLACK, "size": 16},
    "xaxis": {"gridcolor": GREY_GRID, "zerolinecolor": GREY_LIGHT, "linecolor": BLACK, "tickfont": {"color": BLACK}},
    "yaxis": {"gridcolor": GREY_GRID, "zerolinecolor": GREY_LIGHT, "linecolor": BLACK, "tickfont": {"color": BLACK}},
    "colorway": [BLACK, GREY_DARK, GREY_MID, GREY_LIGHT],
    "margin": {"l": 60, "r": 40, "t": 50, "b": 50},
    "hoverlabel": {"bgcolor": WHITE, "font": {"color": BLACK}, "bordercolor": BLACK},
}

def generate_builtin_sample_data(n_samples: int = 15000):
    """Generate built-in sample data so the dashboard runs with no local CSV or downloads."""
    np.random.seed(42)
    data = {
        "Time": np.cumsum(np.random.exponential(800, n_samples)),
        "Amount": np.random.lognormal(3, 1.4, n_samples),
        "Class": np.random.binomial(1, 0.002, n_samples),
    }
    for i in range(1, 29):
        data[f"V{i}"] = np.random.normal(0, 1, n_samples)
    df = pd.DataFrame(data)
    # Slight fraud pattern for realism
    fraud_mask = df["Class"] == 1
    if fraud_mask.sum() > 0:
        df.loc[fraud_mask, "Amount"] *= np.random.uniform(1.5, 4, fraud_mask.sum())
        df.loc[fraud_mask, "V1"] += np.random.normal(1.5, 0.5, fraud_mask.sum())
        df.loc[fraud_mask, "V4"] += np.random.normal(1, 0.5, fraud_mask.sum())
    return df


@st.cache_data(ttl=300)
def load_data(file_path: str):
    """Load data: built-in sample (no path), or from CSV path. No local download required by default."""
    # Empty or missing path → use built-in sample so dashboard works without any CSV
    if not file_path or not str(file_path).strip():
        if REPO_AVAILABLE:
            try:
                processor = DataProcessor()
                return processor.load_data(None)
            except Exception:
                pass
        return generate_builtin_sample_data()
    path = Path(file_path)
    if not path.exists():
        if REPO_AVAILABLE:
            try:
                processor = DataProcessor()
                return processor.load_data(None)
            except Exception:
                pass
        return generate_builtin_sample_data()
    if REPO_AVAILABLE:
        try:
            processor = DataProcessor()
            return processor.load_data(file_path)
        except Exception:
            pass
    return pd.read_csv(file_path)


def apply_filters(df: pd.DataFrame, amount_range, time_range, max_points: int):
    """Apply slider filters and optional subsample."""
    out = df.copy()
    if "Amount" in out.columns:
        out = out[(out["Amount"] >= amount_range[0]) & (out["Amount"] <= amount_range[1])]
    if "Time" in out.columns:
        out = out[(out["Time"] >= time_range[0]) & (out["Time"] <= time_range[1])]
    if len(out) > max_points:
        out = out.sample(n=max_points, random_state=42)
    return out


def main():
    st.set_page_config(
        page_title="Credit Card Fraud Detection Dashboard",
        page_icon="💳",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for black and creme theme
    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {WHITE}; }}
        .stSidebar {{ background-color: {GREY_GRID}; }}
        h1, h2, h3 {{ color: {BLACK}; }}
        .stMetric label {{ color: {GREY_DARK}; }}
        .stSlider label {{ color: {BLACK}; }}
        div[data-testid="stSidebar"] .stMarkdown {{ color: {BLACK}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Credit Card Fraud Detection — Interactive Dashboard")
    st.markdown("Explore the dataset with correlation heatmap, scatter plot, and filters. Black & white theme with high contrast. **No download required:** leave the CSV path empty to use built-in sample data.")

    # Sidebar: data source and filters
    with st.sidebar:
        st.header("Data & filters")
        csv_path = st.text_input(
            "CSV path (optional)",
            value="",
            placeholder="Leave empty for built-in sample — no download needed",
            help="Leave empty to use built-in sample data (no local file required). Or enter a path to your creditcard.csv.",
        )
        st.caption("Default: built-in sample data. Dataset: Time, V1–V28, Amount, Class (0=normal, 1=fraud)")

        st.subheader("Sliders")
        # Use defaults that work for the Kaggle dataset (Time in sec, Amount typically < 25k)
        df_loaded = load_data(csv_path)
        if df_loaded.empty:
            amount_min, amount_max = 0.0, 500.0
            time_min, time_max = 0, 172000
        else:
            amount_min = float(df_loaded["Amount"].min()) if "Amount" in df_loaded.columns else 0.0
            amount_max = float(df_loaded["Amount"].max()) if "Amount" in df_loaded.columns else 500.0
            time_min = int(df_loaded["Time"].min()) if "Time" in df_loaded.columns else 0
            time_max = int(df_loaded["Time"].max()) if "Time" in df_loaded.columns else 172000

        amount_step = max((amount_max - amount_min) / 500, 0.01) if amount_max > amount_min else 1.0
        amount_range = st.slider(
            "Amount range",
            min_value=amount_min,
            max_value=amount_max,
            value=(amount_min, amount_max),
            step=amount_step,
        )
        time_range = st.slider(
            "Time range (sec)",
            min_value=time_min,
            max_value=time_max,
            value=(time_min, time_max),
            step=max(1, (time_max - time_min) // 200),
        )
        max_points = st.slider(
            "Max points in scatter (performance)",
            500,
            50000,
            5000,
            500,
        )

    # Load and filter (uses built-in sample when CSV path is empty)
    df = load_data(csv_path)
    if df.empty:
        st.error("No data loaded. Leave CSV path empty for built-in sample, or check your file path.")
        return

    # Ensure Class is numeric
    if "Class" in df.columns and df["Class"].dtype == object:
        df["Class"] = pd.to_numeric(df["Class"], errors="coerce").fillna(0).astype(int)
    elif "Class" in df.columns:
        df["Class"] = df["Class"].astype(int)

    df_filtered = apply_filters(df, amount_range, time_range, max_points)

    # Repo tie-in: summary using processor description
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total transactions", f"{len(df_filtered):,}")
    with col2:
        fraud_count = int((df_filtered["Class"] == 1).sum()) if "Class" in df_filtered.columns else 0
        st.metric("Fraud count", f"{fraud_count:,}")
    with col3:
        fraud_pct = (df_filtered["Class"].mean() * 100) if "Class" in df_filtered.columns else 0
        st.metric("Fraud rate %", f"{fraud_pct:.2f}%")
    with col4:
        st.metric("Amount range", f"${amount_range[0]:.0f} – ${amount_range[1]:.0f}")

    st.divider()

    # Heatmap: correlation of numeric features (V1–V28, Amount, Time)
    st.subheader("Correlation heatmap")
    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
    # Limit to key columns for readability
    heatmap_cols = [c for c in numeric_cols if c in ["Time", "Amount"] or (c.startswith("V") and c[1:].isdigit())]
    if len(heatmap_cols) > 30:
        heatmap_cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)][:12]  # subset of V's
    corr = df_filtered[heatmap_cols].corr()

    fig_heat = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale=[[0, WHITE], [0.5, GREY_LIGHT], [1, BLACK]],
            zmin=-1,
            zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(color=BLACK, size=9),
        )
    )
    heatmap_layout = {**PLOTLY_LAYOUT, "xaxis": {**PLOTLY_LAYOUT.get("xaxis", {}), "tickangle": -45}}
    fig_heat.update_layout(
        **heatmap_layout,
        title="Feature correlation (filtered data)",
        height=500,
    )
    st.plotly_chart(fig_heat, width="stretch")

    # Scatter: selectable axes, colored by Class
    st.subheader("Scatter plot (colored by Class)")
    scatter_cols = [c for c in numeric_cols if c != "Class"]
    if not scatter_cols:
        st.warning("No numeric columns for scatter.")
    else:
        c1, c2 = st.columns(2)
        x_default = scatter_cols.index("V1") if "V1" in scatter_cols else 0
        y_default = scatter_cols.index("V2") if "V2" in scatter_cols else min(1, len(scatter_cols) - 1)
        with c1:
            x_axis = st.selectbox("X axis", scatter_cols, index=x_default)
        with c2:
            y_axis = st.selectbox("Y axis", scatter_cols, index=y_default)

        fig_scatter = px.scatter(
            df_filtered,
            x=x_axis,
            y=y_axis,
            color="Class" if "Class" in df_filtered.columns else None,
            color_discrete_sequence=[BLACK, GREY_MID],
            opacity=0.6,
            title=f"{y_axis} vs {x_axis}",
        )
        fig_scatter.update_layout(
            **PLOTLY_LAYOUT,
            height=450,
            legend=dict(title="Class", bgcolor=WHITE, bordercolor=BLACK),
        )
        st.plotly_chart(fig_scatter, width="stretch")

    # Repo: feature importance (if FeatureEngineer available and we have Class)
    if REPO_AVAILABLE and "Class" in df.columns and df["Class"].nunique() > 1:
        st.divider()
        st.subheader("Feature importance (from repo)")
        try:
            engineer = FeatureEngineer()
            numeric_only = [c for c in df.select_dtypes(include=[np.number]).columns if c != "Class"]
            df_feat = df[numeric_only + ["Class"]].copy()
            importance = engineer.get_feature_importance(df_feat, "Class")
            imp_series = pd.Series(importance).sort_values(ascending=True).tail(15)
            fig_bar = go.Figure(
                go.Bar(
                    x=imp_series.values,
                    y=imp_series.index,
                    orientation="h",
                    marker_color=BLACK,
                )
            )
            fig_bar.update_layout(
                **PLOTLY_LAYOUT,
                title="Top 15 features by |correlation| with Class",
                height=400,
                xaxis_title="|Correlation|",
            )
            st.plotly_chart(fig_bar, width="stretch")
        except Exception as e:
            st.caption(f"Feature importance unavailable: {e}")

    # Model performance metrics (from README)
    st.divider()
    st.subheader("Model performance (benchmark from repo)")
    metrics_df = pd.DataFrame({
        "Model": ["Logistic Regression", "LightGBM", "Autoencoder", "Ensemble"],
        "PR-AUC": [0.742, 0.856, 0.623, 0.871],
        "ROC-AUC": [0.891, 0.943, 0.789, 0.951],
        "Precision@100": [0.23, 0.31, 0.18, 0.35],
        "Recall@100": [0.45, 0.67, 0.34, 0.72],
    })
    try:
        st.dataframe(metrics_df.style.background_gradient(subset=["PR-AUC", "ROC-AUC"], cmap="Greys"), width="stretch")
    except ImportError:
        st.dataframe(metrics_df, width="stretch")
    st.caption("From README: trained on credit card fraud data. Train with `python train_models.py` and run API with `python api/main.py`.")

    st.sidebar.divider()
    data_source = "built-in sample (no file)" if not (csv_path and str(csv_path).strip()) else csv_path
    st.sidebar.caption(f"Data: {data_source}. Uses DataProcessor/FeatureEngineer when available.")


if __name__ == "__main__":
    main()
