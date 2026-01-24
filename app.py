import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# -----------------------------
# CONFIG
# -----------------------------
MEDIA_CHANNELS = [
    'TV_Impressions', 'YouTube_Impressions', 'Facebook_Impressions',
    'Instagram_Impressions', 'Print_Readership', 'Radio_Listenership'
]

PROMO_FLAGS = ['Feature_Flag', 'Display_Flag', 'TPR_Flag']
PRICE_VARS = ['Net_Price']

st.set_page_config(
    page_title="Marketing Performance Dashboard",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("synthetic_mmm_weekly_india.csv")

# -----------------------------
# MODEL (For Contribution Logic)
# -----------------------------
def build_model(df, target):
    features = MEDIA_CHANNELS + PROMO_FLAGS + PRICE_VARS
    features = [f for f in features if f in df.columns]

    X = df[features].fillna(0)
    y = df[target]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(Xs, y)

    return model, scaler, features

# -----------------------------
# BUSINESS LOGIC
# -----------------------------
def channel_contribution(df, model, scaler, features, target):
    X = scaler.transform(df[features].fillna(0))
    impact = np.abs(X * model.coef_)
    channel_sales = impact.sum(axis=0)

    contrib = pd.DataFrame({
        "Channel": features,
        "Sales Contribution": channel_sales
    })

    contrib["% Contribution"] = contrib["Sales Contribution"] / contrib["Sales Contribution"].sum() * 100
    return contrib.sort_values("Sales Contribution", ascending=False)

def efficiency_table(df, contrib_df):
    rows = []
    for ch in MEDIA_CHANNELS:
        if ch in df.columns and ch in contrib_df["Channel"].values:
            spend = df[ch].sum()
            sales = contrib_df.loc[contrib_df["Channel"] == ch, "Sales Contribution"].values[0]
            if spend > 0:
                rows.append({
                    "Channel": ch,
                    "Spend": spend,
                    "Sales Impact": sales,
                    "Sales per 1K Impressions": sales / (spend / 1000)
                })
    return pd.DataFrame(rows)

# -----------------------------
# APP
# -----------------------------
def main():
    st.title("📊 Marketing Performance Dashboard")

    df = load_data()

    with st.sidebar:
        brand = st.selectbox("Brand", ["All"] + sorted(df["Brand"].unique()))
        target = st.selectbox("Sales Metric", ["Sales_Value", "Sales_Units"])
        run = st.button("View Dashboard", type="primary")

    if not run:
        return

    if brand != "All":
        df = df[df["Brand"] == brand]

    model, scaler, features = build_model(df, target)
    contrib_df = channel_contribution(df, model, scaler, features, target)
    eff_df = efficiency_table(df, contrib_df)

    # -----------------------------
    # KPI CARDS
    # -----------------------------
    total_sales = df[target].sum()
    total_units = df["Sales_Units"].sum()
    avg_price = df["Net_Price"].mean()

    yearly = df.groupby("Year")[target].sum().reset_index()
    yoy_growth = ((yearly.iloc[-1][target] / yearly.iloc[-2][target]) - 1) * 100

    top_channel = contrib_df.iloc[0]["Channel"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sales Value", f"{total_sales/1e6:.2f} M")
    c2.metric("Total Sales Units", f"{total_units/1e6:.2f} M")
    c3.metric("Revenue Growth (YoY)", f"{yoy_growth:.1f}%")
    c4.metric("Average Net Price", f"{avg_price:.2f}")

    st.divider()

    # -----------------------------
    # CHANNEL CONTRIBUTION
    # -----------------------------
    st.subheader("Sales Contribution by Channel")

    fig_contrib = px.bar(
        contrib_df,
        x="Sales Contribution",
        y="Channel",
        orientation="h"
    )
    st.plotly_chart(fig_contrib, use_container_width=True)

    st.divider()

    # -----------------------------
    # EFFICIENCY & SPEND IMPACT
    # -----------------------------
    st.subheader("Marketing Spend vs Sales Impact")

    fig_scatter = px.scatter(
        eff_df,
        x="Spend",
        y="Sales Impact",
        size="Sales Impact",
        color="Channel",
        title="Low Spend / High Return & High Spend / High Impact Channels"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()

    st.subheader("Sales per 1,000 Impressions")
    fig_eff = px.bar(
        eff_df,
        x="Sales per 1K Impressions",
        y="Channel",
        orientation="h"
    )
    st.plotly_chart(fig_eff, use_container_width=True)

    st.divider()

    # -----------------------------
    # YEARLY TREND
    # -----------------------------
    st.subheader("Year-wise Sales Trend")

    fig_trend = px.line(
        yearly,
        x="Year",
        y=target,
        markers=True
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.divider()

    # -----------------------------
    # PROMOTION LIFT
    # -----------------------------
    st.subheader("Sales Lift During Promotions")

    promo_sales = df.copy()
    promo_sales["Promo"] = np.where(
        promo_sales[PROMO_FLAGS].sum(axis=1) > 0, "Promotion", "No Promotion"
    )

    promo_summary = promo_sales.groupby("Promo")[target].sum().reset_index()

    fig_promo = px.bar(
        promo_summary,
        x="Promo",
        y=target
    )
    st.plotly_chart(fig_promo, use_container_width=True)


if __name__ == "__main__":
    main()
