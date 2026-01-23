import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Configuration
# -----------------------------
MEDIA_CHANNELS = [
    'TV_Impressions', 'YouTube_Impressions', 'Facebook_Impressions',
    'Instagram_Impressions', 'Print_Readership', 'Radio_Listenership'
]

PROMO_CHANNELS = ['Feature_Flag', 'Display_Flag', 'TPR_Flag', 'Trade_Spend']
PRICE_VARS = ['Net_Price']

st.set_page_config(
    page_title="Executive Marketing Performance Dashboard",
    page_icon="🌍",
    layout="wide"
)


# -----------------------------
# Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("synthetic_mmm_weekly_india.csv")
    return df


# -----------------------------
# Model
# -----------------------------
def build_model(df, target):
    features = MEDIA_CHANNELS + PROMO_CHANNELS + PRICE_VARS
    features = [f for f in features if f in df.columns]

    X = df[features].fillna(0)
    y = df[target]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(Xs, y)

    importance = pd.DataFrame({
        "Driver": features,
        "Impact": model.coef_,
        "Abs": np.abs(model.coef_)
    }).sort_values("Abs", ascending=False)

    return model, scaler, importance, features


# -----------------------------
# Visuals
# -----------------------------
def plot_geo_heatmap(df, target):
    geo_sales = df.groupby("Geo")[target].sum().reset_index()

    fig = px.choropleth(
        geo_sales,
        locations="Geo",
        locationmode="country names",
        color=target,
        color_continuous_scale="Blues",
        title="Global Sales Distribution"
    )
    fig.update_layout(height=450)
    return fig


def compute_channel_contribution(df, model, scaler, features, target):
    X = df[features].fillna(0)
    Xs = scaler.transform(X)

    raw_contribution = (Xs * model.coef_).sum(axis=0)

    contrib_df = pd.DataFrame({
        "Channel": features,
        "Raw_Contribution": raw_contribution
    })

    total_sales = df[target].sum()

    contrib_df["Sales_Contribution"] = (
        contrib_df["Raw_Contribution"] /
        contrib_df["Raw_Contribution"].sum()
    ) * total_sales

    contrib_df["Share_%"] = (
        contrib_df["Sales_Contribution"] / total_sales * 100
    )

    return contrib_df.sort_values("Sales_Contribution", ascending=False)


def plot_channel_contribution(contrib_df):
    fig = px.bar(
        contrib_df,
        x="Sales_Contribution",
        y="Channel",
        orientation="h",
        title="How Much Sales Each Channel Contributes"
    )
    fig.update_layout(height=450)
    return fig


def plot_efficiency(df, contrib_df):
    rows = []

    for ch in MEDIA_CHANNELS:
        if ch in df.columns and ch in contrib_df["Channel"].values:
            activity = df[ch].sum()
            sales = contrib_df.loc[
                contrib_df["Channel"] == ch, "Sales_Contribution"
            ].values[0]

            if activity > 0:
                rows.append({
                    "Channel": ch,
                    "Sales per Unit Activity": sales / activity
                })

    eff_df = pd.DataFrame(rows).sort_values("Sales per Unit Activity")

    fig = px.bar(
        eff_df,
        x="Sales per Unit Activity",
        y="Channel",
        orientation="h",
        title="Which Channels Generate More Sales per Unit of Activity"
    )
    fig.update_layout(height=400)

    return fig, eff_df


# -----------------------------
# App
# -----------------------------
def main():
    st.title("🌍 Executive Marketing Performance Dashboard")
    st.caption("A business-first view of where sales come from and what drives them")

    df = load_data()

    with st.sidebar:
        brand = st.selectbox("Brand", ["All"] + sorted(df["Brand"].unique()), index=0)
        geo = st.selectbox("Geo", ["All"] + sorted(df["Geo"].unique()), index=0)
        target = st.selectbox("Sales Metric", ["Sales_Units", "Sales_Value"], index=0)
        run = st.button("View Dashboard", type="primary")

    if not run:
        return

    if brand != "All":
        df = df[df["Brand"] == brand]
    if geo != "All":
        df = df[df["Geo"] == geo]

    model, scaler, imp, features = build_model(df, target)

    # -----------------------------
    # GEO HEATMAP
    # -----------------------------
    st.plotly_chart(
        plot_geo_heatmap(df, target),
        use_container_width=True
    )

    # -----------------------------
    # KPI CARDS (2 x 2)
    # -----------------------------
    total_sales_m = df[target].sum() / 1_000_000
    avg_price = df["Net_Price"].mean()
    top_driver = imp.iloc[0]["Driver"]

    contrib_df = compute_channel_contribution(df, model, scaler, features, target)
    eff_fig, eff_df = plot_efficiency(df, contrib_df)
    best_channel = eff_df.iloc[-1]["Channel"]

    c1, c2 = st.columns(2)
    c1.metric("Total Sales", f"{total_sales_m:.2f} M")
    c2.metric("Average Net Price", f"{avg_price:.2f}", help="Average selling price per unit")

    c3, c4 = st.columns(2)
    c3.metric("Top Sales Driver", top_driver)
    c4.metric("Most Efficient Channel", best_channel)

    st.divider()

    # -----------------------------
    # BUSINESS DRIVERS (TEXT FIRST)
    # -----------------------------
    st.subheader("📌 What Is Driving Sales?")
    for _, row in imp.head(5).iterrows():
        direction = "increases" if row["Impact"] > 0 else "reduces"
        st.write(f"• **{row['Driver']}** strongly {direction} sales")

    st.divider()

    # -----------------------------
    # CHANNEL CONTRIBUTION
    # -----------------------------
    st.subheader("📊 Channel Contribution to Total Sales")
    st.plotly_chart(
        plot_channel_contribution(contrib_df),
        use_container_width=True
    )

    st.caption("This chart shows how much sales each channel contributes in absolute terms.")

    st.divider()

    # -----------------------------
    # MARKETING EFFICIENCY
    # -----------------------------
    st.subheader("⚡ Marketing Efficiency")
    st.plotly_chart(eff_fig, use_container_width=True)
    st.caption(
        "Higher bars indicate channels that generate more sales per unit of activity."
    )


if __name__ == "__main__":
    main()
