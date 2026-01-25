import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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
DISTRIBUTION_VARS = ['Weighted_Distribution', 'Numeric_Distribution', 'TDP']
EXTERNAL_VARS = ['CPI', 'GDP_Growth', 'Festival_Index', 'Rainfall_Index']

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


def calculate_roi_by_channel(df, target):
    """Calculate ROI for each media channel"""
    roi_data = []
    
    for channel in MEDIA_CHANNELS:
        if channel in df.columns:
            total_spend = df[channel].sum()
            # Correlation as proxy for sales driven
            correlation = df[channel].corr(df[target])
            estimated_sales = df[target].sum() * abs(correlation) * 0.1  # Simplified estimate
            
            if total_spend > 0:
                roi = (estimated_sales / total_spend) * 100
                roi_data.append({
                    'Channel': channel.replace('_', ' '),
                    'ROI (%)': roi,
                    'Total Spend': total_spend,
                    'Estimated Sales': estimated_sales
                })
    
    return pd.DataFrame(roi_data).sort_values('ROI (%)', ascending=False)


def geo_performance(df, target):
    """Analyze performance by geography"""
    geo_stats = df.groupby('Geo').agg({
        target: 'sum',
        'Weighted_Distribution': 'mean',
        'Net_Price': 'mean'
    }).reset_index()
    
    geo_stats.columns = ['Geo', 'Total Sales', 'Avg Distribution', 'Avg Price']
    return geo_stats.sort_values('Total Sales', ascending=False)


def sku_performance(df, target):
    """Analyze top and bottom performing SKUs"""
    sku_stats = df.groupby('SKU').agg({
        target: 'sum',
        'Net_Price': 'mean',
        'Weighted_Distribution': 'mean'
    }).reset_index()
    
    sku_stats.columns = ['SKU', 'Total Sales', 'Avg Price', 'Avg Distribution']
    return sku_stats.sort_values('Total Sales', ascending=False)


def price_elasticity_analysis(df, target):
    """Analyze relationship between price and sales"""
    price_bins = pd.qcut(df['Net_Price'], q=5, duplicates='drop')
    price_sales = df.groupby(price_bins)[target].mean().reset_index()
    price_sales['Price_Range'] = price_sales['Net_Price'].astype(str)
    return price_sales


def festival_impact(df, target):
    """Analyze sales during festival periods"""
    df_copy = df.copy()
    df_copy['Festival_Period'] = df_copy['Festival_Index'].apply(
        lambda x: 'High Festival' if x > df['Festival_Index'].quantile(0.75) else 
                  'Low Festival' if x < df['Festival_Index'].quantile(0.25) else 'Normal'
    )
    
    festival_stats = df_copy.groupby('Festival_Period')[target].mean().reset_index()
    return festival_stats.sort_values(target, ascending=False)

# =============================
# APP
# =============================
def main():
    st.title("📊 Marketing Mix Model Dashboard")
    st.markdown("*Comprehensive analysis of marketing effectiveness across channels, geographies, and time*")

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
        st.header("🎯 Filters")
        brand = st.selectbox("Brand", ["All"] + sorted(df["Brand"].unique()))
        geo = st.selectbox("Geography", ["All"] + sorted(df["Geo"].unique()))
        target = st.selectbox("Sales Metric", ["Sales_Value", "Sales_Units"])
        
        st.divider()
        st.markdown("### Analysis Sections")
        show_time = st.checkbox("Time Trends", value=True)
        show_channel = st.checkbox("Channel Performance", value=True)
        show_geo = st.checkbox("Geography Analysis", value=True)
        show_sku = st.checkbox("SKU Performance", value=True)
        show_external = st.checkbox("External Factors", value=True)
        
        st.divider()
        run = st.button("🚀 View Dashboard", type="primary", use_container_width=True)

    if not run:
        st.info("👈 Configure filters and click **View Dashboard** to generate insights")
        return

    # Apply filters
    if brand != "All":
        df = df[df["Brand"] == brand]
    if geo != "All":
        df = df[df["Geo"] == geo]

    # -----------------------------
    # MODEL & DATA PREP
    # -----------------------------
    model, scaler, features = build_model(df, target)
    contrib_df = channel_contribution(df, model, scaler, features, target)
    eff_df = efficiency_table(df, contrib_df)

    # =============================
    # KPI CARDS (EXECUTIVE SNAPSHOT)
    # =============================
    st.subheader("📈 Executive Summary")
    
    total_sales_value = df["Sales_Value"].sum()
    total_units = df["Sales_Units"].sum()
    avg_price = df["Net_Price"].mean()
    avg_distribution = df["Weighted_Distribution"].mean()

    yearly = df.groupby("Year")[target].sum().reset_index()

    yoy_growth = (
        (yearly.iloc[-1][target] / yearly.iloc[-2][target]) - 1
    ) * 100 if len(yearly) > 1 else 0

    top_channel = contrib_df.iloc[0]["Channel"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Sales Value", f"₹{total_sales_value/1e6:.2f}M")
    c2.metric("Total Units Sold", f"{total_units/1e6:.2f}M")
    c3.metric("YoY Growth", f"{yoy_growth:.1f}%", delta=f"{yoy_growth:.1f}%")
    c4.metric("Avg. Net Price", f"₹{avg_price:.2f}")
    c5.metric("Avg. Distribution", f"{avg_distribution:.1f}%")

    st.divider()

    # =============================
    # TIME-BASED TRENDS
    # =============================
    if show_time:
        st.subheader("📅 Sales Performance Over Time")

        tab1, tab2, tab3 = st.tabs(["📊 Quarterly Trend", "📅 Monthly Trend", "📈 Yearly Trend"])
        
        with tab1:
            quarterly = df.groupby("YearQuarter")[target].sum().reset_index()
            fig_quarter = px.line(
                quarterly,
                x="YearQuarter",
                y=target,
                markers=True,
                title="Quarter-wise Sales Performance"
            )
            fig_quarter.update_layout(xaxis_title="Quarter", yaxis_title="Sales", hovermode='x unified')
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
    if show_channel:
        st.subheader("📺 Channel Performance Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Absolute Sales Contribution")
            fig_contrib = px.bar(
                contrib_df,
                x="Sales Contribution",
                y="Channel",
                orientation="h",
                color="Sales Contribution",
                color_continuous_scale="Blues"
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

        # Channel Efficiency
        st.markdown("#### 💡 Channel Efficiency (Sales per 1,000 Impressions)")
        
        col3, col4 = st.columns([2, 1])
        
        with col3:
            fig_eff = px.bar(
                eff_df.sort_values("Sales per 1K Impressions", ascending=False),
                x="Sales per 1K Impressions",
                y="Channel",
                orientation="h",
                color="Sales per 1K Impressions",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_eff, use_container_width=True)
        
        with col4:
            st.markdown("**Efficiency Ranking**")
            for idx, row in eff_df.sort_values("Sales per 1K Impressions", ascending=False).iterrows():
                st.metric(
                    row['Channel'].replace('_', ' '),
                    f"{row['Sales per 1K Impressions']:.2f}",
                    delta=None
                )

        # ROI Analysis
        st.markdown("#### 💰 Return on Investment (ROI) by Channel")
        roi_df = calculate_roi_by_channel(df, target)
        
        if not roi_df.empty:
            fig_roi = px.bar(
                roi_df,
                x='Channel',
                y='ROI (%)',
                title='Estimated ROI by Marketing Channel',
                color='ROI (%)',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_roi, use_container_width=True)
            
            best_roi_channel = roi_df.iloc[0]['Channel']
            best_roi_value = roi_df.iloc[0]['ROI (%)']
            st.info(f"💎 **Best ROI:** {best_roi_channel} with {best_roi_value:.1f}% return on investment")

        st.divider()

    # =============================
    # GEOGRAPHY ANALYSIS
    # =============================
    if show_geo:
        st.subheader("🗺️ Geographic Performance")
        
        geo_stats = geo_performance(df, target)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_geo_sales = px.bar(
                geo_stats.head(10),
                x='Geo',
                y='Total Sales',
                title='Top 10 Geographies by Sales',
                color='Total Sales',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_geo_sales, use_container_width=True)
        
        with col2:
            fig_geo_dist = px.scatter(
                geo_stats,
                x='Avg Distribution',
                y='Total Sales',
                size='Avg Price',
                hover_data=['Geo'],
                title='Sales vs Distribution by Geography',
                labels={'Avg Distribution': 'Average Distribution (%)', 'Total Sales': 'Total Sales'}
            )
            st.plotly_chart(fig_geo_dist, use_container_width=True)
        
        # Top and Bottom Geos
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("**🏆 Top 5 Geographies**")
            st.dataframe(geo_stats.head(5)[['Geo', 'Total Sales']].reset_index(drop=True), hide_index=True)
        
        with col4:
            st.markdown("**⚠️ Bottom 5 Geographies**")
            st.dataframe(geo_stats.tail(5)[['Geo', 'Total Sales']].reset_index(drop=True), hide_index=True)

        st.divider()

    # =============================
    # SKU PERFORMANCE
    # =============================
    if show_sku:
        st.subheader("📦 SKU Performance Analysis")
        
        sku_stats = sku_performance(df, target)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🌟 Top 10 SKUs by Sales**")
            fig_top_sku = px.bar(
                sku_stats.head(10),
                x='Total Sales',
                y='SKU',
                orientation='h',
                color='Avg Price',
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig_top_sku, use_container_width=True)
        
        with col2:
            st.markdown("**⚡ SKU Price-Sales Relationship**")
            fig_sku_price = px.scatter(
                sku_stats.head(20),
                x='Avg Price',
                y='Total Sales',
                size='Avg Distribution',
                hover_data=['SKU'],
                color='Total Sales',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_sku_price, use_container_width=True)

        st.divider()

    # =============================
    # EXTERNAL FACTORS IMPACT
    # =============================
    if show_external:
        st.subheader("🌍 External Factors Impact")
        
        tab1, tab2, tab3, tab4 = st.tabs(["🎊 Festival Impact", "💰 Price Elasticity", "🎯 Promotion Impact", "📊 Economic Factors"])
        
        with tab1:
            festival_stats = festival_impact(df, target)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig_festival = px.bar(
                    festival_stats,
                    x='Festival_Period',
                    y=target,
                    title='Average Sales by Festival Period',
                    color=target,
                    color_continuous_scale='Oranges'
                )
                st.plotly_chart(fig_festival, use_container_width=True)
            
            with col2:
                st.markdown("**Festival Performance**")
                for _, row in festival_stats.iterrows():
                    st.metric(row['Festival_Period'], f"{row[target]:,.0f}")
        
        with tab2:
            price_sales = price_elasticity_analysis(df, target)
            
            fig_price = px.line(
                price_sales,
                x='Price_Range',
                y=target,
                title='Price Elasticity: Sales Response to Price Changes',
                markers=True
            )
            fig_price.update_layout(xaxis_title='Price Range', yaxis_title='Average Sales')
            st.plotly_chart(fig_price, use_container_width=True)
            
            st.info("📉 **Insight:** Analyze how sales respond to different price points to optimize pricing strategy")
        
        with tab3:
            promo_df = df.copy()
            promo_df["Promo"] = np.where(
                promo_df[PROMO_FLAGS].sum(axis=1) > 0,
                "Promotion",
                "No Promotion"
            )
            
            promo_summary = promo_df.groupby("Promo")[target].sum().reset_index()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig_promo = px.bar(
                    promo_summary,
                    x="Promo",
                    y=target,
                    title="Sales Lift During Promotional Periods",
                    color="Promo",
                    color_discrete_map={"Promotion": "#1f77b4", "No Promotion": "#aec7e8"}
                )
                st.plotly_chart(fig_promo, use_container_width=True)
            
            with col2:
                if len(promo_summary) > 1:
                    promo_sales = promo_summary.loc[promo_summary["Promo"] == "Promotion", target].values
                    no_promo_sales = promo_summary.loc[promo_summary["Promo"] == "No Promotion", target].values
                    
                    if len(promo_sales) > 0 and len(no_promo_sales) > 0:
                        lift = ((promo_sales[0] / no_promo_sales[0]) - 1) * 100
                        st.metric("Promotional Lift", f"{lift:.1f}%")
                        
                        # Individual promo type analysis
                        st.markdown("**By Promotion Type**")
                        for promo in PROMO_FLAGS:
                            promo_impact = df[df[promo] == 1][target].sum()
                            st.metric(promo.replace('_', ' '), f"{promo_impact/1e6:.2f}M")
        
        with tab4:
            col1, col2 = st.columns(2)
            
            with col1:
                # GDP Growth Impact
                gdp_sales = df.groupby(pd.qcut(df['GDP_Growth'], q=4, duplicates='drop'))[target].mean().reset_index()
                gdp_sales['GDP_Range'] = gdp_sales['GDP_Growth'].astype(str)
                
                fig_gdp = px.bar(
                    gdp_sales,
                    x='GDP_Range',
                    y=target,
                    title='Sales vs GDP Growth',
                    color=target,
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_gdp, use_container_width=True)
            
            with col2:
                # CPI Impact
                cpi_sales = df.groupby(pd.qcut(df['CPI'], q=4, duplicates='drop'))[target].mean().reset_index()
                cpi_sales['CPI_Range'] = cpi_sales['CPI'].astype(str)
                
                fig_cpi = px.bar(
                    cpi_sales,
                    x='CPI_Range',
                    y=target,
                    title='Sales vs Consumer Price Index',
                    color=target,
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_cpi, use_container_width=True)

        st.divider()

    # =============================
    # RECOMMENDATIONS SECTION
    # =============================
    st.subheader("💡 Strategic Recommendations")
    
    rec_col1, rec_col2, rec_col3 = st.columns(3)
    
    with rec_col1:
        st.markdown("**🎯 Channel Optimization**")
        top_3_channels = contrib_df.head(3)['Channel'].tolist()
        st.write(f"Focus on: {', '.join([ch.replace('_', ' ') for ch in top_3_channels])}")
        
    with rec_col2:
        st.markdown("**📍 Geographic Expansion**")
        geo_stats = geo_performance(df, target)
        low_performing = geo_stats.tail(3)['Geo'].tolist()
        st.write(f"Opportunity zones: {', '.join(low_performing)}")
    
    with rec_col3:
        st.markdown("**💰 Pricing Strategy**")
        avg_price_metric = df['Net_Price'].mean()
        st.write(f"Current avg: ₹{avg_price_metric:.2f}")
        st.write("Monitor price elasticity for optimization")

# =============================
# RUN APP
# =============================
if __name__ == "__main__":
    main()
