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
        "AbsImpact": np.abs(model.coef_)
    }).sort_values("AbsImpact", ascending=False)

    return model, scaler, importance, features


# -----------------------------
# BUSINESS LOGIC
# -----------------------------
def compute_positive_contribution(df, model, scaler, features, target):
    """
    Converts model output into positive, business-friendly
    sales attribution that sums to total sales.
    """
    X = df[features].fillna(0)
    Xs = scaler.transform(X)

    raw = np.abs((Xs * model.coef_).sum(axis=0))
    total_sales = df[target].sum()

    contrib_df = pd.DataFrame({
        "Channel": features,
        "Sales Contribution": (raw / raw.sum()) * total_sales
    }).sort_values("Sales Contribution", ascending=False)

    return contrib_df


def compute_efficiency(df, contrib_df):
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
                    "Sales Generated per Unit Activity": sales / activity
                })

    return pd.DataFrame(rows).sort_values(
        "Sales Generated per Unit Activity", ascending=False
    )


# -----------------------------
# APP
# -----------------------------
def main():
    st.title("📊 Marketing Performance Dashboard")

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

    contrib_df = compute_positive_contribution(df, model, scaler, features, target)
    eff_df = compute_efficiency(df, contrib_df)

    top_driver = imp.iloc[0]["Driver"]
    best_channel = eff_df.iloc[0]["Channel"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sales", f"{total_sales_m:.2f} M")
    c2.metric("Average Net Price", f"{avg_price:.2f}", help="Average selling price per unit")
    c3.metric("Top Sales Driver", top_driver)
    c4.metric("Most Efficient Channel", best_channel)

    st.divider()

    # -----------------------------
    # WHAT IS DRIVING SALES (CARDS)
    # -----------------------------
    st.subheader("What Is Driving Sales?")

    cards = st.columns(4)
    for i, (_, row) in enumerate(imp.head(4).iterrows()):
        strength = "Strong Impact" if row["AbsImpact"] > imp["AbsImpact"].median() else "Moderate Impact"
        cards[i].metric(
            row["Driver"],
            strength,
            help="This driver has a significant influence on sales movement"
        )

    st.divider()

    # -----------------------------
    # CHANNEL CONTRIBUTION
    # -----------------------------
    st.subheader("Channel Contribution to Total Sales")

    fig_contrib = px.bar(
        contrib_df,
        x="Sales Contribution",
        y="Channel",
        orientation="h"
    )
    fig_contrib.update_layout(height=450)

    st.plotly_chart(fig_contrib, use_container_width=True)
    st.caption("Shows how much each channel contributes to total sales in absolute terms.")

    st.divider()

    # -----------------------------
    # MARKETING EFFICIENCY
    # -----------------------------
    st.subheader("Marketing Efficiency")

    fig_eff = px.bar(
        eff_df,
        x="Sales Generated per Unit Activity",
        y="Channel",
        orientation="h"
    )
    fig_eff.update_layout(height=400)

    st.plotly_chart(fig_eff, use_container_width=True)
    st.caption("Channels on the right generate more sales for the same level of effort.")

    st.divider()

    # -----------------------------
    # REGIONAL SALES INTENSITY
    # -----------------------------
    st.subheader("Sales Intensity Across India Regions")

    region_sales = df.groupby("Geo")[target].sum().reset_index()

    fig_region = px.bar(
        region_sales,
        x=target,
        y="Geo",
        orientation="h"
    )
    fig_region.update_layout(height=400)

    st.plotly_chart(fig_region, use_container_width=True)
    st.caption("Highlights which regions contribute more to total sales.")


if __name__ == "__main__":
    main()
