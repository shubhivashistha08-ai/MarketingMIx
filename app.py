import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import io

# Marketing channels for analysis
MEDIA_CHANNELS = [
    'TV_Impressions', 'YouTube_Impressions', 'Facebook_Impressions', 
    'Instagram_Impressions', 'Print_Readership', 'Radio_Listenership'
]

PROMO_CHANNELS = ['Feature_Flag', 'Display_Flag', 'TPR_Flag', 'Trade_Spend']
DISTRIBUTION_VARS = ['Weighted_Distribution', 'Numeric_Distribution', 'TDP']
PRICE_VARS = ['Net_Price', 'CPI']
EXTERNAL_VARS = ['GDP_Growth', 'Festival_Index', 'Rainfall_Index']

def load_data(file_obj=None):
    """Load data from uploaded file or use sample data"""
    if file_obj is None:
        # Create sample data for demonstration
        np.random.seed(42)
        n_weeks = 52
        
        sample_data = {
            'Week': pd.date_range(start='2023-01-01', periods=n_weeks, freq='W'),
            'Brand': np.random.choice(['Brand_A', 'Brand_B'], n_weeks),
            'Geo': np.random.choice(['North', 'South', 'East', 'West'], n_weeks),
            'SKU': np.random.choice(['SKU_1', 'SKU_2', 'SKU_3'], n_weeks),
            'Sales_Value': np.random.uniform(50000, 150000, n_weeks),
            'Sales_Units': np.random.uniform(1000, 5000, n_weeks),
            'TV_Impressions': np.random.uniform(0, 100000, n_weeks),
            'YouTube_Impressions': np.random.uniform(0, 50000, n_weeks),
            'Facebook_Impressions': np.random.uniform(0, 80000, n_weeks),
            'Instagram_Impressions': np.random.uniform(0, 60000, n_weeks),
            'Print_Readership': np.random.uniform(0, 30000, n_weeks),
            'Radio_Listenership': np.random.uniform(0, 40000, n_weeks),
            'Feature_Flag': np.random.choice([0, 1], n_weeks),
            'Display_Flag': np.random.choice([0, 1], n_weeks),
            'TPR_Flag': np.random.choice([0, 1], n_weeks),
            'Trade_Spend': np.random.uniform(0, 10000, n_weeks),
            'Weighted_Distribution': np.random.uniform(50, 100, n_weeks),
            'Numeric_Distribution': np.random.uniform(40, 90, n_weeks),
            'TDP': np.random.uniform(30, 80, n_weeks),
            'Net_Price': np.random.uniform(80, 120, n_weeks),
            'CPI': np.random.uniform(95, 105, n_weeks),
            'GDP_Growth': np.random.uniform(2, 4, n_weeks),
            'Festival_Index': np.random.uniform(0, 2, n_weeks),
            'Rainfall_Index': np.random.uniform(0, 1.5, n_weeks)
        }
        
        df = pd.DataFrame(sample_data)
        return df
    
    try:
        # Read uploaded file
        if hasattr(file_obj, 'name'):
            if file_obj.name.endswith('.csv'):
                df = pd.read_csv(file_obj.name)
            elif file_obj.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_obj.name)
            else:
                raise ValueError("Unsupported file format. Please upload CSV or Excel file.")
        else:
            df = pd.read_csv(file_obj)
        
        # Convert Week to datetime if it exists
        if 'Week' in df.columns:
            df['Week'] = pd.to_datetime(df['Week'])
        return df
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")

def prepare_data(df, target='Sales_Value'):
    """Prepare data for modeling"""
    features = MEDIA_CHANNELS + PROMO_CHANNELS + DISTRIBUTION_VARS + PRICE_VARS + EXTERNAL_VARS
    available_features = [f for f in features if f in df.columns]
    
    X = df[available_features].copy()
    y = df[target].copy()
    X = X.fillna(0)
    
    return X, y, available_features

def build_mmm_model(df, target='Sales_Value'):
    """Build Marketing Mix Model"""
    X, y, features = prepare_data(df, target)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features)
    
    model = LinearRegression()
    model.fit(X_scaled_df, y)
    
    y_pred = model.predict(X_scaled_df)
    
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    
    importance_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_,
        'Abs_Coefficient': np.abs(model.coef_)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    return model, scaler, importance_df, r2, mae, mape, y, y_pred

def create_contribution_analysis(df, model, scaler, features, target='Sales_Value'):
    """Calculate contribution of each channel to sales"""
    X, y, _ = prepare_data(df, target)
    X_scaled = scaler.transform(X)
    
    contributions = pd.DataFrame(X_scaled * model.coef_, columns=features)
    contributions['Base'] = model.intercept_
    contributions['Total_Sales'] = y.values
    
    channel_contributions = {
        'TV': contributions['TV_Impressions'].sum() if 'TV_Impressions' in contributions else 0,
        'YouTube': contributions['YouTube_Impressions'].sum() if 'YouTube_Impressions' in contributions else 0,
        'Facebook': contributions['Facebook_Impressions'].sum() if 'Facebook_Impressions' in contributions else 0,
        'Instagram': contributions['Instagram_Impressions'].sum() if 'Instagram_Impressions' in contributions else 0,
        'Print': contributions['Print_Readership'].sum() if 'Print_Readership' in contributions else 0,
        'Radio': contributions['Radio_Listenership'].sum() if 'Radio_Listenership' in contributions else 0,
        'Trade_Spend': contributions['Trade_Spend'].sum() if 'Trade_Spend' in contributions else 0,
        'Promotions': sum([contributions[col].sum() for col in ['Feature_Flag', 'Display_Flag', 'TPR_Flag'] if col in contributions]),
        'Distribution': sum([contributions[col].sum() for col in DISTRIBUTION_VARS if col in contributions]),
        'Price': sum([contributions[col].sum() for col in PRICE_VARS if col in contributions]),
        'Base': contributions['Base'].sum()
    }
    
    return pd.DataFrame(list(channel_contributions.items()), columns=['Channel', 'Contribution'])

def plot_feature_importance(importance_df):
    """Plot feature importance"""
    fig = px.bar(
        importance_df.head(15),
        x='Abs_Coefficient',
        y='Feature',
        orientation='h',
        title='Top 15 Feature Importance (Absolute Coefficients)',
        labels={'Abs_Coefficient': 'Absolute Coefficient', 'Feature': 'Marketing Variable'},
        color='Abs_Coefficient',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=500, showlegend=False)
    return fig

def plot_channel_contribution(contrib_df):
    """Plot channel contribution"""
    contrib_df = contrib_df[contrib_df['Contribution'] != 0]
    contrib_df = contrib_df.sort_values('Contribution', ascending=True)
    
    total = contrib_df['Contribution'].sum()
    contrib_df['Percentage'] = (contrib_df['Contribution'] / total * 100).round(1)
    
    fig = px.bar(
        contrib_df,
        x='Contribution',
        y='Channel',
        orientation='h',
        title='Sales Contribution by Marketing Channel',
        labels={'Contribution': 'Sales Contribution ($)', 'Channel': 'Marketing Channel'},
        color='Contribution',
        color_continuous_scale='RdYlGn',
        text='Percentage'
    )
    fig.update_traces(texttemplate='%{text}%', textposition='outside')
    fig.update_layout(height=500)
    return fig

def plot_actual_vs_predicted(y_actual, y_pred):
    """Plot actual vs predicted sales"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(y_actual))),
        y=y_actual,
        mode='lines',
        name='Actual Sales',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=list(range(len(y_pred))),
        y=y_pred,
        mode='lines',
        name='Predicted Sales',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Actual vs Predicted Sales Over Time',
        xaxis_title='Time Period',
        yaxis_title='Sales Value',
        height=400,
        hovermode='x unified'
    )
    
    return fig

def plot_residuals(y_actual, y_pred):
    """Plot residuals distribution"""
    residuals = y_actual - y_pred
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=30,
        name='Residuals',
        marker_color='#636EFA'
    ))
    
    fig.update_layout(
        title='Residuals Distribution',
        xaxis_title='Residual (Actual - Predicted)',
        yaxis_title='Frequency',
        height=400,
        showlegend=False
    )
    
    return fig

def plot_roi_analysis(df, contrib_df):
    """Calculate and plot ROI for marketing channels"""
    roi_data = []
    
    media_mapping = {
        'TV_Impressions': 'TV',
        'YouTube_Impressions': 'YouTube',
        'Facebook_Impressions': 'Facebook',
        'Instagram_Impressions': 'Instagram',
        'Print_Readership': 'Print',
        'Radio_Listenership': 'Radio'
    }
    
    for col, channel_name in media_mapping.items():
        if col in df.columns:
            spend = df[col].sum()
            contrib = contrib_df[contrib_df['Channel'] == channel_name]['Contribution'].values
            if len(contrib) > 0 and spend > 0:
                roi = (contrib[0] / spend) * 100
                roi_data.append({
                    'Channel': channel_name,
                    'ROI': roi,
                    'Spend': spend,
                    'Contribution': contrib[0]
                })
    
    if 'Trade_Spend' in df.columns:
        spend = df['Trade_Spend'].sum()
        contrib = contrib_df[contrib_df['Channel'] == 'Trade_Spend']['Contribution'].values
        if len(contrib) > 0 and spend > 0:
            roi = (contrib[0] / spend) * 100
            roi_data.append({
                'Channel': 'Trade_Spend',
                'ROI': roi,
                'Spend': spend,
                'Contribution': contrib[0]
            })
    
    if len(roi_data) == 0:
        return go.Figure().add_annotation(text="No ROI data available", showarrow=False)
    
    roi_df = pd.DataFrame(roi_data)
    roi_df = roi_df.sort_values('ROI', ascending=True)
    
    fig = px.bar(
        roi_df,
        x='ROI',
        y='Channel',
        orientation='h',
        title='Return on Investment (ROI) by Channel',
        labels={'ROI': 'ROI (%)', 'Channel': 'Marketing Channel'},
        color='ROI',
        color_continuous_scale='RdYlGn',
        text='ROI'
    )
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(height=400)
    return fig

def get_data_summary(df):
    """Get summary statistics of the dataset"""
    summary = {
        'Total Rows': len(df),
        'Date Range': f"{df['Week'].min().strftime('%Y-%m-%d')} to {df['Week'].max().strftime('%Y-%m-%d')}" if 'Week' in df.columns else 'N/A',
        'Brands': df['Brand'].nunique() if 'Brand' in df.columns else 'N/A',
        'Geographies': df['Geo'].nunique() if 'Geo' in df.columns else 'N/A',
        'SKUs': df['SKU'].nunique() if 'SKU' in df.columns else 'N/A',
    }
    return summary

def analyze_data(file_obj, brand_filter, geo_filter, target_var):
    """Main analysis function"""
    try:
        df = load_data(file_obj)
        data_summary = get_data_summary(df)
        
        if brand_filter != "All" and 'Brand' in df.columns:
            df = df[df['Brand'] == brand_filter]
        if geo_filter != "All" and 'Geo' in df.columns:
            df = df[df['Geo'] == geo_filter]
        
        if len(df) == 0:
            return "⚠️ No data available after filtering. Please adjust your filters.", None, None, None, None, None, None, ["All"], ["All"]
        
        model, scaler, importance_df, r2, mae, mape, y_actual, y_pred = build_mmm_model(df, target_var)
        
        fig_importance = plot_feature_importance(importance_df)
        
        X, y, features = prepare_data(df, target_var)
        contrib_df = create_contribution_analysis(df, model, scaler, features, target_var)
        fig_contribution = plot_channel_contribution(contrib_df)
        
        fig_actual_pred = plot_actual_vs_predicted(y_actual, y_pred)
        fig_residuals = plot_residuals(y_actual, y_pred)
        fig_roi = plot_roi_analysis(df, contrib_df)
        
        summary = f"""
## 📊 Model Performance Metrics

| Metric | Value |
|--------|-------|
| **R² Score** | {r2:.4f} ({r2*100:.1f}% variance explained) |
| **Mean Absolute Error (MAE)** | ${mae:,.2f} |
| **Mean Absolute Percentage Error (MAPE)** | {mape:.2f}% |

## 📈 Data Summary

| Attribute | Value |
|-----------|-------|
| **Total Records** | {len(df):,} |
| **Date Range** | {data_summary['Date Range']} |
| **Brands Analyzed** | {data_summary['Brands']} |
| **Geographies** | {data_summary['Geographies']} |
| **Total Sales** | ${df[target_var].sum():,.2f} |
| **Average Sales** | ${df[target_var].mean():,.2f} |
| **Std Deviation** | ${df[target_var].std():,.2f} |

## 🎯 Key Insights

The Marketing Mix Model successfully explains **{r2*100:.1f}%** of sales variance, with an average prediction error of **${mae:,.2f}** per observation.

### Top 3 Contributing Factors:
"""
        for i, row in importance_df.head(3).iterrows():
            summary += f"\n{i+1}. **{row['Feature']}** (Coefficient: {row['Coefficient']:.4f})"
        
        # Update filter options
        brands = ["All"] + sorted(df['Brand'].unique().tolist()) if 'Brand' in df.columns else ["All"]
        geos = ["All"] + sorted(df['Geo'].unique().tolist()) if 'Geo' in df.columns else ["All"]
        
        return summary, fig_importance, fig_contribution, fig_actual_pred, fig_residuals, fig_roi, importance_df.head(20), brands, geos
    
    except Exception as e:
        error_msg = f"❌ **Error occurred:** {str(e)}\n\nPlease upload a valid CSV or Excel file with the required columns."
        return error_msg, None, None, None, None, None, None, ["All"], ["All"]

# Create Gradio interface
with gr.Blocks(title="Marketing Mix Model Dashboard", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📈 Marketing Mix Model Dashboard
    ### Analyze the impact of marketing channels on sales performance
    
    Upload your marketing data file (CSV or Excel) or use the sample data to get started.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📁 Data Upload")
            file_input = gr.File(label="Upload CSV or Excel file (optional - sample data will be used if no file uploaded)", file_types=['.csv', '.xlsx', '.xls'])
            
            gr.Markdown("### 🎛️ Analysis Controls")
            
            brand_filter = gr.Dropdown(choices=["All"], value="All", label="Filter by Brand")
            geo_filter = gr.Dropdown(choices=["All"], value="All", label="Filter by Geography")
            target_var = gr.Dropdown(
                choices=['Sales_Value', 'Sales_Units'], 
                value='Sales_Value', 
                label="Target Variable"
            )
            
            analyze_btn = gr.Button("🚀 Run Marketing Mix Analysis", variant="primary", size="lg")
            
            gr.Markdown("""
            ---
            ### 💡 Quick Tips
            - Upload your data file or use sample data
            - Start with "All" filters to see overall trends
            - Filter by specific Brand/Geo for detailed analysis
            - Compare Sales_Value vs Sales_Units for different insights
            """)
    
    with gr.Row():
        summary_output = gr.Markdown(label="Analysis Summary")
    
    with gr.Tabs():
        with gr.Tab("📊 Feature Importance"):
            gr.Markdown("### Which marketing variables drive sales the most?")
            importance_plot = gr.Plot(label="Feature Importance Visualization")
            importance_table = gr.Dataframe(label="Top 20 Features - Detailed Coefficients")
        
        with gr.Tab("💰 Channel Contribution"):
            gr.Markdown("### How much does each channel contribute to total sales?")
            contribution_plot = gr.Plot(label="Sales Contribution by Channel")
        
        with gr.Tab("📈 Model Performance"):
            gr.Markdown("### How well does the model predict actual sales?")
            actual_pred_plot = gr.Plot(label="Actual vs Predicted Sales")
            residuals_plot = gr.Plot(label="Residuals Distribution")
        
        with gr.Tab("💵 ROI Analysis"):
            gr.Markdown("### Which channels give the best return on investment?")
            roi_plot = gr.Plot(label="Return on Investment by Channel")
    
    gr.Markdown("""
    ---
    ## 📖 Understanding Your Results
    
    ### 📊 Feature Importance
    Shows which marketing variables have the strongest statistical relationship with sales. Higher absolute coefficients indicate stronger impact.
    
    ### 💰 Channel Contribution
    Breaks down how much each marketing channel contributes to your total sales. Use this to understand your marketing mix composition.
    
    ### 📈 Model Performance
    - **Actual vs Predicted**: Shows how closely the model predictions match real sales
    - **Residuals**: Distribution of prediction errors (should be normally distributed around zero)
    - **R² Score**: Percentage of sales variance explained by the model (higher is better)
    
    ### 💵 ROI Analysis
    Calculates return on investment for each channel. Formula: (Sales Contribution / Channel Spend) × 100%
    """)
    
    analyze_btn.click(
        fn=analyze_data,
        inputs=[file_input, brand_filter, geo_filter, target_var],
        outputs=[summary_output, importance_plot, contribution_plot, actual_pred_plot, residuals_plot, roi_plot, importance_table, brand_filter, geo_filter]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
