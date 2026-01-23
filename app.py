import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error


# -----------------------------
# Configuration
# -----------------------------
MEDIA_CHANNELS = [
    'TV_Impressions', 'YouTube_Impressions', 'Facebook_Impressions',
    'Instagram_Impressions', 'Print_Readership', 'Radio_Listenership'
]

PROMO_CHANNELS = ['Feature_Flag', 'Display_Flag', 'TPR_Flag', 'Trade_Spend']
DISTRIBUTION_VARS = ['Weighted_Distribution', 'Numeric_Distribution', 'TDP']
PRICE_VARS = ['Net_Price', 'CPI']
EXTERNAL_VARS = ['GDP_Growth', 'Festival_Index', 'Rainfall_Index']


st.set_page_config(
    page_title="Marketing Mix Model Dashboard",
    page_icon="📈",
    layout="wide"
)


# -----------------------------
# Data Loading
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("synthetic_mmm_weekly_india.csv")

    if "Week" in df.columns:
        df["Week"] = pd.to_datetime(df["Week"])

    return df


# -----------------------------
# Modeling
# -----------------------------
def prepare_data(df, target):
    features = (
        MEDIA_CHANNELS
        + PROMO_CHANNELS
        + DISTRIBUTION_VARS
        + PRICE_VARS
        + EXTERNAL_VARS
    )
    features = [f for f in features if f in df.columns]

    X = df[features].fillna(0)
    y = df[target]

    return X, y, features


@st.cache_data
def build_model(df, target):
    X, y, features = prepare_data(df, target)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(Xs, y)

    y_pred = model.predict(Xs)

    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mape = np.mean(np.abs((y - y_pred) / np.maximum(y, 1))) * 100

    importance = pd.DataFrame({
        "Feature": features,
        "Coefficient": model.coef_,
        "Abs": np.abs(model.coef_)
    }).sort_values("Abs", ascending=False)

    return model, scaler, importance, r2, mae, mape, y, y_pred, features


# -----------------------------
# Visuals
# -----------------------------
def plot_importance(df):
    fig = px.bar(
        df.head(15),
        x="Abs",
        y="Feature",
        orientation="h",
        title="Top Feature Importance"
    )
    fig.update_layout(height=500)
    return fig


def plot_actual_vs_pred(y, yhat):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y.values, mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(y=yhat, mode="lines", name="Predicted", line=dict(dash="dash")))
    fig.update_layout(title="Actual vs Predicted Sales", height=400)
    return fig


def plot_residuals(y, yhat):
    residuals = y.values - yhat
    fig = px.histogram(residuals, nbins=30, title="Residual Distribution")
    fig.update_layout(height=400)
    return fig


def plot_contribution(df, model, scaler, features):
    X = df[features].fillna(0)
    Xs = scaler.transform(X)

    contribution = (Xs * model.coef_).sum(axis=0)

    contrib_df = pd.DataFrame({
        "Channel": features,
        "Contribution": contribution
    }).sort_values("Contribution")

    fig = px.bar(
        contrib_df,
        x="Contribution",
        y="Channel",
        orientation="h",
        title="Channel Contribution"
    )
    fig.update_layout(height=500)

    return fig, contrib_df


def plot_efficiency(df, contrib_df):
    data = []

    for ch in MEDIA_CHANNELS:
        if ch in df.columns and ch in contrib_df["Channel"].values:
            spend = df[ch].sum()
            cont = contrib_df.loc[
                contrib_df["Channel"] == ch, "Contribution"
            ].sum()

            if spend > 0:
                data.append({
                    "Channel": ch,
                    "Efficiency_Index": cont / spend
                })

    if not data:
        fig = go.Figure()
        fig.add_annotation(text="No efficiency data available", showarrow=False)
        return fig

    ef = pd.DataFrame(data).sort_values("Efficiency_Index")

    fig = px.bar(
        ef,
        x="Efficiency_Index",
        y="Channel",
        orientation="h",
        title="Marketing Efficiency Index"
    )
    fig.update_layout(height=400)
    return fig


# -----------------------------
# Main App
# -----------------------------
def main():
    st.title("📈 Marketing Mix Model Dashboard")

    with st.sidebar:
        st.header("Configuration")

        df = load_data()

        brands = ["All"] + sorted(df["Brand"].unique().tolist())
        geos = ["All"] + sorted(df["Geo"].unique().tolist())

        brand = st.selectbox("Brand", brands)
        geo = st.selectbox("Geo", geos)
        target = st.selectbox("Target Variable", ["Sales_Value", "Sales_Units"])

        run_analysis = st.button("🚀 Run Analysis", type="primary")

    if run_analysis:
        filtered_df = df.copy()

        if brand != "All":
            filtered_df = filtered_df[filtered_df["Brand"] == brand]
        if geo != "All":
            filtered_df = filtered_df[filtered_df["Geo"] == geo]

        if filtered_df.empty:
            st.error("⚠️ No data after filtering")
            return

        with st.spinner("Building model..."):
            model, scaler, imp, r2, mae, mape, y, yhat, features = build_model(
                filtered_df, target
            )

        st.subheader("📊 Model Performance")
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("R²", f"{r2:.3f}")
        c2.metric("MAE", f"{mae:,.0f}")
        c3.metric("MAPE", f"{mape:.2f}%")
        c4.metric("Total Sales", f"{filtered_df[target].sum():,.0f}")

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Feature Importance",
            "Contribution",
            "Actual vs Predicted",
            "Residuals",
            "Efficiency Index",
            "Data Table"
        ])

        with tab1:
            st.plotly_chart(plot_importance(imp), use_container_width=True)

        with tab2:
            fig, contrib_df = plot_contribution(
                filtered_df, model, scaler, features
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.plotly_chart(
                plot_actual_vs_pred(y, yhat),
                use_container_width=True
            )

        with tab4:
            st.plotly_chart(
                plot_residuals(y, yhat),
                use_container_width=True
            )

        with tab5:
            st.plotly_chart(
                plot_efficiency(filtered_df, contrib_df),
                use_container_width=True
            )

        with tab6:
            st.subheader("Top 20 Features")
            st.dataframe(imp.head(20), use_container_width=True)

    else:
        st.info("👈 Configure filters and click **Run Analysis**")
        st.subheader("Data Preview")
        st.dataframe(load_data().head(10), use_container_width=True)


if __name__ == "__main__":
    main()
