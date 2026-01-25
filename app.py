import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Marketing Performance Dashboard",
    page_icon="📊",
    layout="wide"
)

# =============================
# CONFIG
# =============================
MEDIA_CHANNELS = [
    'TV_Impressions',
    'YouTube_Impressions',
    'Facebook_Impressions',
    'Instagram_Impressions',
    'Print_Readership',
    'Radio_Listenership'
]

PROMO_FLAGS = ['Feature_Flag', 'Display_Flag', 'TPR_Flag']
PRICE_VARS = ['Net_Price']

# =============================
# DATA LOADER
# =============================
@st.cache_data
def load_data():
    return pd.read_csv("synthetic_mmm_weekly_india.csv")

# =============================
# MODEL (ONLY FOR CONTRIBUTION LOGIC)
# =============================
def build_model(df, target):
    features = MEDIA_CHANNELS + PROMO_FLAGS + PRICE_VARS
    features = [f for f in features if f in df.columns]

    X = df[features].fillna(0)
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    return model, scaler, features

# =============================
# BUSINESS CALCULATIONS
# =============================
def channel_contribution(df, model, scaler, features, target):
    X_scaled = scaler.transform(df[features].fillna(0))
    impact = np.abs(X_scaled * model.coef_)
    channel_sales = impact.sum(axis=0)

    contrib = pd.DataFrame({
        "Channel": features,
        "Sales Contribution": channel_sales
    })

    contrib["% Contribution"] = (
        contrib["Sales Contribution"] /
        contrib["Sales Contribution"].sum()
    ) * 100

    return contrib.sort_values("Sales Contribution", ascending=False)


def efficiency_table(df, contrib_df):
    rows = []

    for ch in MEDIA_CHANNELS:
        if ch in df.columns and ch in contrib_df["Channel"].values:
            spend = df[ch].sum()
            sales = contrib_df.loc[
                contrib_df["Channel"] == ch, "Sales Contribution"
            ].values[0]

            if spend > 0:
                rows.append({
                    "Channel": ch,
                    "Spend": spend,
                    "Sales Impact": sales,
                    "Sales per 1K Impressions": sales / (spend / 1000)
                })

    return pd.DataFrame(rows)

# =============================
# APP
# =============================
def main():
    st.title("📊 Marketing Performance Dashboard")

    df = load_data()

    # -----------------------------
    # WEEK → YEAR & QUARTER
    # -----------------------------
    df["Week"] = pd.to_datetime(df["Week"])
    df["Year"] = df["Week"].dt.year
    df["Quarter"] = df["Week"].dt.quarter
    df["YearQuarter"] = df["Year"].astype(str) + "-Q" + df["Quarter"].astype(str)
    df["Month"] = df["Week"].dt.to_period('M').astype(str)

    # -----------------------------
    # SIDEBAR
    # -----------------------------
    with st.sidebar:
        st.header("Filters")
        brand = st.selectbox("Brand", ["All"] + sorted(df["Brand"].unique()))
        target = st.selectbox("Sales Metric", ["Sales_Value", "Sales_Units"])
        run = st.button("View Dashboard", type="primary")

    if not run:
        st.info("👈 Select filters and click **View Dashboard**")
        return

    if brand != "All":
        df = df[df["Brand"] == brand]

    # -----------------------------
    # MODEL & DATA PREP
    # -----------------------------
    model, scaler, features = build_model(df, target)
    contrib_df = channel_contribution(df, model, scaler, features, target)
    eff_df = efficiency_table(df, contrib_df)

    # =============================
    # KPI CARDS (EXECUTIVE SNAPSHOT)
    # =============================
    total_sales = df["Sales_Value"].sum()
    total_units = df["Sales_Units"].sum()
    avg_price = df["Net_Price"].mean()

    yearly = df.groupby("Year")[target].sum().reset_index()

    yoy_growth = (
        (yearly.iloc[-1][target] / yearly.iloc[-2][target]) - 1
    ) * 100 if len(yearly) > 1 else 0

    top_channel = contrib_df.iloc[0]["Channel"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sales Value", f"{total_sales/1e6:.2f} M")
    c2.metric("Total Sales Units", f"{total_units/1e6:.2f} M")
    c3.metric("Revenue Growth (YoY)", f"{yoy_growth:.1f}%")
    c4.metric("Average Net Price", f"{avg_price:.2f}")

    st.divider()

    # =============================
    # TIME-BASED TRENDS (MOVED TO TOP)
    # =============================
    st.subheader("📈 Sales Performance Over Time")

    # Create tabs for different time views
    tab1, tab2, tab3 = st.tabs(["Quarterly Trend", "Monthly Trend", "Yearly Trend"])
    
    with tab1:
        quarterly = df.groupby("YearQuarter")[target].sum().reset_index()
        fig_quarter = px.line(
            quarterly,
            x="YearQuarter",
            y=target,
            markers=True,
            title="Quarter-wise Sales Performance"
        )
        fig_quarter.update_layout(xaxis_title="Quarter", yaxis_title="Sales")
        st.plotly_chart(fig_quarter, use_container_width=True)
    
    with tab2:
        monthly = df.groupby("Month")[target].sum().reset_index()
        fig_month = px.bar(
            monthly,
            x="Month",
            y=target,
            title="Month-wise Sales Performance"
        )
        fig_month.update_layout(xaxis_title="Month", yaxis_title="Sales")
        st.plotly_chart(fig_month, use_container_width=True)
    
    with tab3:
        fig_year = px.line(
            yearly,
            x="Year",
            y=target,
            markers=True,
            title="Year-wise Sales Trend"
        )
        fig_year.update_layout(xaxis_title="Year", yaxis_title="Sales")
        st.plotly_chart(fig_year, use_container_width=True)

    st.divider()

    # =============================
    # CHANNEL CONTRIBUTION (SIDE BY SIDE)
    # =============================
    st.subheader("📊 Channel Performance Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Absolute Sales Contribution")
        fig_contrib = px.bar(
            contrib_df,
            x="Sales Contribution",
            y="Channel",
            orientation="h"
        )
        fig_contrib.update_layout(showlegend=False)
        st.plotly_chart(fig_contrib, use_container_width=True)

    with col2:
        st.markdown("#### Percentage Distribution")
        fig_share = px.pie(
            contrib_df,
            names="Channel",
            values="% Contribution",
            hole=0.5
        )
        st.plotly_chart(fig_share, use_container_width=True)

    st.success(
        f"🏆 **Top Performing Channel:** {top_channel} contributes the highest share of total sales."
    )

    st.divider()

    # =============================
    # MARKETING EFFICIENCY
    # =============================
    st.subheader("💡 Marketing Efficiency Insights")

    fig_eff = px.bar(
        eff_df.sort_values("Sales per 1K Impressions", ascending=False),
        x="Sales per 1K Impressions",
        y="Channel",
        orientation="h",
        title="Sales Generated per 1,000 Impressions (Efficiency Metric)",
        color="Sales per 1K Impressions",
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig_eff, use_container_width=True)

    best_eff = eff_df.sort_values(
        "Sales per 1K Impressions", ascending=False
    ).iloc[0]["Channel"]

    st.info(
        f"💡 **Efficiency Insight:** {best_eff} delivers the highest sales per 1,000 impressions, making it the most cost-effective channel."
    )

    st.divider()

    # =============================
    # PROMOTION IMPACT
    # =============================
    st.subheader("🎯 Promotional Impact on Sales")

    promo_df = df.copy()
    promo_df["Promo"] = np.where(
        promo_df[PROMO_FLAGS].sum(axis=1) > 0,
        "Promotion",
        "No Promotion"
    )

    promo_summary = promo_df.groupby("Promo")[target].sum().reset_index()

    fig_promo = px.bar(
        promo_summary,
        x="Promo",
        y=target,
        title="Sales Lift During Promotional Periods",
        color="Promo",
        color_discrete_map={"Promotion": "#1f77b4", "No Promotion": "#aec7e8"}
    )
    st.plotly_chart(fig_promo, use_container_width=True)

    if len(promo_summary) > 1:
        promo_sales = promo_summary.loc[promo_summary["Promo"] == "Promotion", target].values
        no_promo_sales = promo_summary.loc[promo_summary["Promo"] == "No Promotion", target].values
        
        if len(promo_sales) > 0 and len(no_promo_sales) > 0:
            if promo_sales[0] > no_promo_sales[0]:
                lift = ((promo_sales[0] / no_promo_sales[0]) - 1) * 100
                st.success(f"📈 Promotions drive **{lift:.1f}% higher sales** compared to non-promotion periods.")
            else:
                st.warning("⚠️ Promotions do not significantly outperform non-promotion periods.")

# =============================
# RUN APP
# =============================
if __name__ == "__main__":
    main()
