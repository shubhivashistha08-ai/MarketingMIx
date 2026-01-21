# =====================================================
# Marketing Mix Modeling (MMM) Dashboard – Gradio App
# Dataset: synthetic_mmm_weekly_india.csv
# =====================================================

import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

# =====================================================
# Load Data
# =====================================================
def load_data():
    df = pd.read_csv("synthetic_mmm_weekly_india.csv")
    return df

df = load_data()

TARGET_COL = "Sales"
TIME_COL = "Week"

CHANNEL_COLS = [c for c in df.columns if c not in [TARGET_COL, TIME_COL]]

# =====================================================
# Train MMM Model
# =====================================================
def train_mmm(df):
    X = df[CHANNEL_COLS]
    y = df[TARGET_COL]

    model = LinearRegression()
    model.fit(X, y)

    contributions = pd.DataFrame({
        "Channel": CHANNEL_COLS,
        "Coefficient": model.coef_
    }).sort_values("Coefficient", ascending=False)

    return model, contributions

model, contributions = train_mmm(df)

# =====================================================
# ROI Calculation
# =====================================================
def calculate_roi(df, contributions):
    spend = df[CHANNEL_COLS].sum()

    roi_df = contributions.copy()
    roi_df["Total_Spend"] = roi_df["Channel"].map(spend)
    roi_df["ROI"] = roi_df["Coefficient"] / roi_df["Total_Spend"]

    roi_df = roi_df.replace([np.inf, -np.inf], 0).fillna(0)
    return roi_df.sort_values("ROI", ascending=False)

roi_df = calculate_roi(df, contributions)

# =====================================================
# KPIs
# =====================================================
total_sales = df[TARGET_COL].sum()
avg_weekly_sales = df[TARGET_COL].mean()
top_channel = contributions.iloc[0]["Channel"]

# =====================================================
# What-if Simulation
# =====================================================
def simulate_spend(tv_mult, digital_mult, search_mult, social_mult):
    sim_df = df.copy()

    multiplier_map = {
        "TV_Spend": tv_mult,
        "Digital_Spend": digital_mult,
        "Search_Spend": search_mult,
        "Social_Spend": social_mult
    }

    for col, mult in multiplier_map.items():
        if col in sim_df.columns:
            sim_df[col] *= mult

    preds = model.predict(sim_df[CHANNEL_COLS])
    sim_df["Predicted_Sales"] = preds

    fig = px.line(
        sim_df,
        x=TIME_COL,
        y="Predicted_Sales",
        title="Predicted Sales – What If Scenario",
        markers=True
    )

    return fig

# =====================================================
# Gradio UI
# =====================================================
with gr.Blocks(title="🇮🇳 Marketing Mix Model – India", theme=gr.themes.Soft()) as app:

    gr.Markdown("""
    # 📊 Marketing Mix Modeling Dashboard (India)
    ### Data-Driven Marketing Budget Optimization  
    Understand **which marketing channels drive sales**, measure **ROI**,  
    and run **what-if simulations** to optimize spend.
    """)

    # =========================
    # OVERVIEW TAB
    # =========================
    with gr.Tab("📈 Overview"):
        kpi1, kpi2, kpi3 = gr.Row(), gr.Row(), gr.Row()

        with kpi1:
            gr.Markdown(f"### 💰 Total Sales\n**{total_sales:,.0f}**")

        with kpi2:
            gr.Markdown(f"### 📆 Avg Weekly Sales\n**{avg_weekly_sales:,.0f}**")

        with kpi3:
            gr.Markdown(f"### 🥇 Top Channel\n**{top_channel}**")

        fig_sales = px.line(
            df,
            x=TIME_COL,
            y=TARGET_COL,
            title="Weekly Sales Trend",
            markers=True
        )
        gr.Plot(fig_sales)

    # =========================
    # CONTRIBUTION TAB
    # =========================
    with gr.Tab("📊 Channel Contribution"):
        fig_contrib = px.bar(
            contributions,
            x="Channel",
            y="Coefficient",
            title="Marketing Channel Contribution to Sales",
            color="Coefficient"
        )
        gr.Plot(fig_contrib)

        gr.Dataframe(
            contributions,
            label="Channel Contribution Table",
            interactive=False
        )

    # =========================
    # ROI TAB
    # =========================
    with gr.Tab("💰 ROI Analysis"):
        fig_roi = px.bar(
            roi_df,
            x="Channel",
            y="ROI",
            title="Return on Investment (ROI) by Channel",
            color="ROI"
        )
        gr.Plot(fig_roi)

        gr.Dataframe(
            roi_df.round(4),
            label="ROI Table",
            interactive=False
        )

    # =========================
    # WHAT-IF TAB
    # =========================
    with gr.Tab("🔮 What-If Simulation"):
        gr.Markdown("""
        Adjust marketing spend multipliers to simulate  
        **future sales impact**.
        """)

        with gr.Row():
            tv = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="TV Spend")
            digital = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Digital Spend")
            search = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Search Spend")
            social = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Social Spend")

        sim_plot = gr.Plot()

        simulate_btn = gr.Button("🚀 Run Simulation")
        simulate_btn.click(
            simulate_spend,
            inputs=[tv, digital, search, social],
            outputs=sim_plot
        )

    # =========================
    # DATA TAB
    # =========================
    with gr.Tab("📁 Dataset"):
        gr.Dataframe(df, interactive=False)

    gr.Markdown("""
    ---
    **Built with Gradio • Marketing Mix Modeling • India**  
    📌 Deployed via GitHub + Hugging Face Spaces (Free)
    """)

app.launch()

