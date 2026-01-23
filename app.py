import gradio as gr
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


# -----------------------------
# Data Loading
# -----------------------------
def load_data(file_obj=None):
    if file_obj is None:
        np.random.seed(42)
        n = 52
        df = pd.DataFrame({
            'Week': pd.date_range('2023-01-01', periods=n, freq='W'),
            'Brand': np.random.choice(['Brand_A', 'Brand_B'], n),
            'Geo': np.random.choice(['North', 'South', 'East', 'West'], n),
            'SKU': np.random.choice(['SKU_1', 'SKU_2', 'SKU_3'], n),
            'Sales_Value': np.random.uniform(50000, 150000, n),
            'Sales_Units': np.random.uniform(1000, 5000, n),
            'TV_Impressions': np.random.uniform(0, 100000, n),
            'YouTube_Impressions': np.random.uniform(0, 50000, n),
            'Facebook_Impressions': np.random.uniform(0, 80000, n),
            'Instagram_Impressions': np.random.uniform(0, 60000, n),
            'Print_Readership': np.random.uniform(0, 30000, n),
            'Radio_Listenership': np.random.uniform(0, 40000, n),
            'Feature_Flag': np.random.randint(0, 2, n),
            'Display_Flag': np.random.randint(0, 2, n),
            'TPR_Flag': np.random.randint(0, 2, n),
            'Trade_Spend': np.random.uniform(0, 10000, n),
            'Weighted_Distribution': np.random.uniform(50, 100, n),
            'Numeric_Distribution': np.random.uniform(40, 90, n),
            'TDP': np.random.uniform(30, 80, n),
            'Net_Price': np.random.uniform(80, 120, n),
            'CPI': np.random.uniform(95, 105, n),
            'GDP_Growth': np.random.uniform(2, 4, n),
            'Festival_Index': np.random.uniform(0, 2, n),
            'Rainfall_Index': np.random.uniform(0, 1.5, n),
        })
        return df

    if file_obj.name.endswith('.csv'):
        df = pd.read_csv(file_obj)
    else:
        df = pd.read_excel(file_obj)

    if 'Week' in df.columns:
        df['Week'] = pd.to_datetime(df['Week'])

    return df


# -----------------------------
# Modeling
# -----------------------------
def prepare_data(df, target):
    features = MEDIA_CHANNELS + PROMO_CHANNELS + DISTRIBUTION_VARS + PRICE_VARS + EXTERNAL_VARS
    features = [f for f in features if f in df.columns]
    X = df[features].fillna(0)
    y = df[target]
    return X, y, features


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

    return model, scaler, importance, r2, mae, mape, y, y_pred


# -----------------------------
# Visuals
# -----------------------------
def plot_importance(df):
    return px.bar(
        df.head(15),
        x="Abs",
        y="Feature",
        orientation="h",
        title="Top Feature Importance"
    )


def plot_actual_vs_pred(y, yhat):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y, mode="lines", name="Actual"))
    fig.add_trace(go.Scatter(y=yhat, mode="lines", name="Predicted", line=dict(dash="dash")))
    fig.update_layout(title="Actual vs Predicted Sales")
    return fig


def plot_residuals(y, yhat):
    res = y - yhat
    return px.histogram(res, nbins=30, title="Residual Distribution")


def plot_contribution(df, model, scaler, features):
    X = df[features].fillna(0)
    Xs = scaler.transform(X)
    contrib = pd.DataFrame(Xs * model.coef_, columns=features).sum()

    out = pd.DataFrame({
        "Channel": contrib.index,
        "Contribution": contrib.values
    }).sort_values("Contribution")

    return px.bar(out, x="Contribution", y="Channel", orientation="h",
                  title="Channel Contribution")


def plot_efficiency(df, contrib_df):
    data = []
    for ch in MEDIA_CHANNELS:
        if ch in df.columns:
            spend = df[ch].sum()
            cont = contrib_df[contrib_df["Channel"] == ch]["Contribution"].sum()
            if spend > 0:
                data.append({"Channel": ch, "Efficiency_Index": cont / spend})

    if not data:
        return go.Figure().add_annotation(text="No efficiency data", showarrow=False)

    ef = pd.DataFrame(data).sort_values("Efficiency_Index")
    return px.bar(
        ef, x="Efficiency_Index", y="Channel", orientation="h",
        title="Marketing Efficiency Index"
    )


# -----------------------------
# Main Analysis
# -----------------------------
def analyze(file, brand, geo, target):
    df = load_data(file)
    full_df = df.copy()

    if brand != "All":
        df = df[df["Brand"] == brand]
    if geo != "All":
        df = df[df["Geo"] == geo]

    if df.empty:
        return "⚠️ No data after filtering", None, None, None, None, None, ["All"], ["All"]

    model, scaler, imp, r2, mae, mape, y, yhat = build_model(df, target)

    summary = f"""
### 📊 Model Performance
- **R²:** {r2:.3f}
- **MAE:** ${mae:,.0f}
- **MAPE:** {mape:.2f}%
- **Total Sales:** ${df[target].sum():,.0f}
"""

    brands = ["All"] + sorted(full_df["Brand"].unique())
    geos = ["All"] + sorted(full_df["Geo"].unique())

    contrib_fig = plot_contribution(df, model, scaler, imp["Feature"].tolist())

    return (
        summary,
        plot_importance(imp),
        contrib_fig,
        plot_actual_vs_pred(y, yhat),
        plot_residuals(y, yhat),
        plot_efficiency(df, pd.DataFrame({"Channel": imp["Feature"], "Contribution": imp["Coefficient"]})),
        imp.head(20),
        brands,
        geos
    )


# -----------------------------
# UI
# -----------------------------
with gr.Blocks(title="Marketing Mix Model Dashboard") as demo:
    gr.Markdown("# 📈 Marketing Mix Model Dashboard")

    with gr.Row():
        file = gr.File(label="Upload CSV / Excel (optional)")
        brand = gr.Dropdown(["All"], value="All", label="Brand")
        geo = gr.Dropdown(["All"], value="All", label="Geo")
        target = gr.Dropdown(["Sales_Value", "Sales_Units"], value="Sales_Value")

    btn = gr.Button("🚀 Run Analysis")

    summary = gr.Markdown()

    with gr.Tabs():
        imp_plot = gr.Plot(label="Feature Importance")
        contrib_plot = gr.Plot(label="Contribution")
        actual_plot = gr.Plot(label="Actual vs Predicted")
        resid_plot = gr.Plot(label="Residuals")
        eff_plot = gr.Plot(label="Efficiency Index")
        imp_table = gr.Dataframe()

    btn.click(
        analyze,
        [file, brand, geo, target],
        [summary, imp_plot, contrib_plot, actual_plot, resid_plot, eff_plot, imp_table, brand, geo]
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
