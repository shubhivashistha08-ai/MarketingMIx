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
DISTRIBUTION_VARS = ['Weighted_Distribution', 'Numeric_Distribution', 'TDP']
PRICE_VARS = ['Net_Price']
EXTERNAL_VARS = ['GDP_Growth', 'Festival_Index', 'Rainfall_Index']


st.set_page_config(
    page_title="Business Marketing Performance Dashboard",
    page_icon="📊",
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

    importance = pd.DataFrame({
        "Driver": features,
        "Impact": model.coef_,
        "Abs": np.abs(model.coef_)
    }).sort_values("Abs", ascending=False)

    return model, scaler, importance, features


# -----------------------------
# Business Visuals
# -----------------------------
def plot_business_drivers(imp_df):
    fig = px.bar(
        imp_df.head(12),
        x="Abs",
        y="Driver",
        orientation="h",
        title="Key Business Drivers of Sales"
    )
    fig.update_layout(height=500)
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
        title="Sales Contribution by Channel & Levers"
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

    ef = pd.DataFrame(data).sort_values("Efficiency_Index")

    fig = px.bar(
        ef,
        x="Efficiency_Index",
        y="Channel",
        orientation="h",
        title="Marketing Efficiency Index (Higher = Better)"
    )
    fig.update_layout(height=400)
    return fig, ef


# -----------------------------
# Main App
# -----------------------------
def main():
    st.title("📊 Business Marketing Performance Dashboard")
    st.caption("A simple, business-first view of what drives your sales")

    with st.sidebar:
        st.header("View Settings")

        df = load_data()

        brands = ["All"] + sorted(df["Brand"].unique().tolist())
        geos = ["All"] + sorted(df["Geo"].unique().tolist())

        brand = st.selectbox("Brand", brands, index=0)
        geo = st.selectbox("Geo", geos, index=0)

        target = st.selectbox(
            "Sales Metric",
            ["Sales_Units", "Sales_Value"],
            index=0
        )

        run_analysis = st.button("🚀 View Dashboard", type="primary")

    if not run_analysis:
        st.info("👈 Select filters and click **View Dashboard**")
        return

    # Filter data
    filtered_df = df.copy()

    if brand != "All":
        filtered_df = filtered_df[filtered_df["Brand"] == brand]
    if geo != "All":
        filtered_df = filtered_df[filtered_df["Geo"] == geo]

    # Build model
    model, scaler, imp, features = build_model(filtered_df, target)

    # -----------------------------
    # Business KPIs
    # -----------------------------
    total_sales = filtered_df[target].sum()
    total_sales_m = total_sales / 1_000_000

    avg_price = filtered_df["Net_Price"].mean() if "Net_Price" in filtered_df else 0

    top_driver = imp.iloc[0]["Driver"]

    fig_contrib, contrib_df = plot_contribution(
        filtered_df, model, scaler, features
    )

    _, eff_df = plot_efficiency(filtered_df, contrib_df)
    top_eff_channel = eff_df.iloc[-1]["Channel"] if not eff_df.empty else "N/A"

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Sales",
        f"{total_sales_m:.2f} M",
        help="Total sales across selected brand & geography"
    )

    col2.metric(
        "Average Net Price",
        f"{avg_price:.2f}",
        help="Average selling price"
    )

    col3.metric(
        "Top Sales Driver",
        top_driver,
        help="Strongest driver impacting sales"
    )

    col4.metric(
        "Most Efficient Channel",
        top_eff_channel,
        help="Channel generating highest sales per unit of activity"
    )

    # -----------------------------
    # Tabs
    # -----------------------------
    tab1, tab2, tab3 = st.tabs([
        "Business Drivers",
        "Channel Contribution",
        "Marketing Efficiency"
    ])

    with tab1:
        st.plotly_chart(
            plot_business_drivers(imp),
            use_container_width=True
        )

    with tab2:
        st.plotly_chart(fig_contrib, use_container_width=True)

    with tab3:
        st.plotly_chart(
            plot_efficiency(filtered_df, contrib_df)[0],
            use_container_width=True
        )


if __name__ == "__main__":
    main()
