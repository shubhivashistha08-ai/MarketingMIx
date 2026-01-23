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

PROMO_CHANNELS = ['Feature_Flag', 'Display_Flag', 'TPR_Flag', 'Trade_Spend']
PRICE_VARS = ['Net_Price']

REGION_COORDS = {
    "Central": (23.2599, 77.4126),
    "East": (22.5726, 88.3639),
    "Metro Delhi": (28.6139, 77.2090),
    "Metro Mumbai": (19.0760, 72.8777),
    "North": (30.7333, 76.7794),
    "NorthEast": (26.1445, 91.7362),
    "South": (12.9716, 77.5946),
    "West": (23.0225, 72.5714)
}

st.set_page_config(
    page_title="India Marketing Performance Dashboard",
    page_icon="🇮🇳",
    layout="wide"
)


# -----------------------------
# DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("synthetic_mmm_weekly_india.csv")


# -----------------------------
# MODEL
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
# VISUALS
# -----------------------------
def plot_india_heatmap(df, target):
    geo_sales = df.groupby("Geo")[target].sum().reset_index()

    geo_sales["lat"] = geo_sales["Geo"].map(lambda x: REGION_COORDS.get(x, (None, None))[0])
    geo_sales["lon"] = geo_sales["Geo"].map(lambda x: REGION_COORDS.get(x, (None, None))[1])

    fig = px.scatter_geo(
        geo_sales,
        lat="lat",
        lon="lon",
        size=target,
        color=target,
        hover_name="Geo",
        scope="asia",
        projection="mercator",
        title="Sales Intensity Across India Regions",
        color_continuous_scale="Blues",
        size_max=40
    )

    fig.update_geos(
        center=dict(lat=22.5, lon=78.9),
        lataxis_range=[5, 35],
        lonaxis_range=[65, 95],
        showland=True
    )

    fig.update_layout(height=500)
    return fig


def compute_contribution(df, model, scaler, features, target):
    X = df[features].fillna(0)
    Xs = scaler.transform(X)

    raw = (Xs * model.coef_).sum(axis=0)
    total_sales = df[target].sum()

    contrib_df = pd.DataFrame({
        "Channel": features,
        "Sales Contribution": (raw / raw.sum()) * total_sales
    }).sort_values("Sales Contribution", ascending=False)

    return contrib_df


def plot_contribution(contrib_df):
    fig = px.bar(
        contrib_df,
        x="Sales Contribution",
        y="Channel",
        orientation="h",
        title="How Much Sales Each Channel Generates"
    )
    fig.update_layout(height=450)
    return fig


def plot_efficiency(df, contrib_df):
    rows = []

    for ch in MEDIA_CHANNELS:
        if ch in df.columns and ch in contrib_df["Channel"].values:
            activity = df[ch].sum()
            sales = contrib_df.loc[
                contrib_df["Channel"] == ch, "Sales Contribution"
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
# APP
# -----------------------------
def main():
    st.title("🇮🇳 India Marketing Performance Dashboard")
    st.caption("Business-friendly view of regional sales performance and marketing impact")

    df = load_data()

    with st.sidebar:
        brand = st.selectbox("Brand", ["All"] + sorted(df["Brand"].unique()), index=0)
        target = st.selectbox("Sales Metric", ["Sales_Units", "Sales_Value"], index=0)
        run = st.button("View Dashboard", type="primary")

    if not run:
        return

    if brand != "All":
        df = df[df["Brand"] == brand]

    model, scaler, imp, features = build_model(df, target)

    # -----------------------------
    # KPI CARDS
    # -----------------------------
    total_sales_m = df[target].sum() / 1_000_000
    avg_price = df["Net_Price"].mean()
    top_driver = imp.iloc[0]["Driver"]

    contrib_df = compute_contribution(df, model, scaler, features, target)
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
    # INDIA HEATMAP (MIDDLE)
    # -----------------------------
    st.plotly_chart(
        plot_india_heatmap(df, target),
        use_container_width=True
    )

    st.divider()

    # -----------------------------
    # BUSINESS DRIVERS
    # -----------------------------
    st.subheader("📌 What Is Driving Sales?")
    for _, row in imp.head(5).iterrows():
        effect = "increases" if row["Impact"] > 0 else "reduces"
        st.write(f"• **{row['Driver']}** strongly {effect} sales")

    st.divider()

    # -----------------------------
    # CHANNEL CONTRIBUTION
    # -----------------------------
    st.subheader("📊 Channel Contribution to Total Sales")
    st.plotly_chart(
        plot_contribution(contrib_df),
        use_container_width=True
    )

    st.divider()

    # -----------------------------
    # MARKETING EFFICIENCY
    # -----------------------------
    st.subheader("⚡ Marketing Efficiency")
    st.plotly_chart(eff_fig, use_container_width=True)
    st.caption("Channels on the right generate more sales per unit of marketing activity.")


if __name__ == "__main__":
    main()
