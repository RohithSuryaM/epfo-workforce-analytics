import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from io import BytesIO
from typing import Dict, List
from data_processor import EPFODataProcessor
import warnings
warnings.filterwarnings('ignore')

# Initialize data processor and load data
st.set_page_config(
    page_title="EPFO Workforce Analysis Tool",
    page_icon="📈",
    layout="wide"
)

@st.cache_data
def load_data(filepath):
    processor = EPFODataProcessor(filepath)
    df = processor.load_and_clean()
    
    # Convert years to datetime for forecasting
    df['year_end'] = df['year'].apply(lambda x: int(x.split('-')[0]) + 1)  # Using end year
    df['year_end'] = pd.to_datetime(df['year_end'].astype(str), format='%Y')
    
    return df, processor.get_summary_stats()

# Load data
try:
    df, summary_stats = load_data("MOSPI Format Corrected.xlsx")
except Exception as e:
    st.error(f"Failed to load data: {str(e)}")
    st.stop()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", 
                       ["Dashboard", "Forecasting", "Attrition Analysis", "Policy Simulator"])

# Dashboard Page with LARGER VISUALIZATIONS
if page == "Dashboard":
    st.title("📈 EPFO Workforce Dashboard")
    st.markdown("""
    ## Comprehensive Workforce Analytics Platform
    
    **Purpose**: This dashboard provides interactive visualization of EPFO workforce dynamics across different age groups and years.
    
    **Key Features**:
    - Real-time filtering by year and age group
    - Trend analysis for key workforce metrics
    - Detailed data exploration
    
    *Why this matters*: Helps policymakers understand workforce participation patterns and identify vulnerable segments.
    """)
    # Display summary stats
    st.subheader("Data Overview")
    st.markdown("""
    *Understanding the dataset scope*:
    - **Years**: Time period covered
    - **Age Groups**: Demographic segmentation
    - **Records**: Total data points available
    - **Latest Year**: Most recent data available
    """)
    cols = st.columns(4)
    cols[0].metric("Years", f"{len(summary_stats['years'])}")
    cols[1].metric("Age Groups", f"{len(summary_stats['age_groups'])}")
    cols[2].metric("Total Records", f"{summary_stats['total_records']:,}")
    cols[3].metric("Latest Year", df['year'].max())
    
    # Filters
    st.subheader("⚙️ Data Filters")
    st.markdown("""
    *How to use*:
    - Select specific years or age groups to focus your analysis
    - Leave all selected for complete overview
    """)
    col1, col2 = st.columns(2)
    with col1:
        selected_years = st.multiselect(
            "Select Years", 
            sorted(df['year'].unique()),
            default=sorted(df['year'].unique())
        )
    with col2:
        selected_ages = st.multiselect(
            "Select Age Groups",
            df['age_group'].unique(),
            default=df['age_group'].unique()
        )
    
    filtered_df = df[(df['year'].isin(selected_years)) & (df['age_group'].isin(selected_ages))]
    
    # Metrics
    st.subheader("Key Metrics")
    st.markdown("""
    **Metric Definitions**:
    - **Subscribers**: New workforce entrants
    - **Exits**: Workforce departures
    - **Rejoins**: Returning workers
    - **Net Payroll**: Subscribers - Exits + Rejoins
    """)
    metric_cols = st.columns(4)
    metrics = {
        "Subscribers": filtered_df['new_subscribers'].sum(),
        "Exits": filtered_df['exited'].sum(),
        "Rejoins": filtered_df['rejoined'].sum(),
        "Net Payroll": filtered_df['net_payroll'].sum()
    }
    for col, (name, value) in zip(metric_cols, metrics.items()):
        col.metric(f"Total {name}", f"{value:,}")
    
    # Visualization - ENLARGED TO FULL WIDTH
    st.subheader("Trend Analysis")
    st.markdown("""
    **How to interpret**:
    - Compare trends across age groups
    - Switch between line/bar views
    - Hover for exact values
    """)
    metric_options = {
        "New Subscribers": "new_subscribers",
        "Exits": "exited",
        "Rejoins": "rejoined",
        "Net Payroll": "net_payroll",
        "Churn Rate": "churn_rate",
        "Rejoin Rate": "rejoin_rate"
    }
    
    # Configuration options
    col1, col2 = st.columns(2)
    with col1:
        selected_metric = st.selectbox(
            "Select Metric",
            list(metric_options.keys()),
            key="metric_select"
        )
    with col2:
        chart_type = st.radio(
            "Chart Type",
            ["Line", "Bar"],
            horizontal=True
        )
    
    # Create the visualization - NOW LARGER
    plt.figure(figsize=(18, 9))  # Significantly larger figure
    
    metric_col = metric_options[selected_metric]
    
    if chart_type == "Line":
        for age in selected_ages:
            age_data = filtered_df[filtered_df['age_group'] == age]
            plt.plot(age_data['year'], age_data[metric_col], 
                    label=age, linewidth=3, marker='o', markersize=8)
    else:
        pivot_data = filtered_df.pivot_table(
            index='year',
            columns='age_group',
            values=metric_col,
            aggfunc='sum'
        )
        pivot_data.plot(kind='bar', width=0.8)
    
    plt.title(f"{selected_metric} Trend", fontsize=18, pad=20)
    plt.xlabel("Year", fontsize=14)
    plt.ylabel(selected_metric, fontsize=14)
    plt.legend(title="Age Group", bbox_to_anchor=(1.05, 1), fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Display the plot
    st.pyplot(plt)
    
    # Data table
    st.subheader("Detailed Data View")
    st.markdown("""
    *Usage tips*:
    - Sort by clicking column headers
    - Scroll horizontally to see all fields
    - Export using download button below
    """)
    st.dataframe(filtered_df.sort_values(['year', 'age_group']))
    
    # Download
    st.download_button(
        "Download Current Data",
        filtered_df.to_csv(index=False),
        "epfo_data.csv",
        "text/csv"
    )

# Forecasting Page
elif page == "Forecasting":
    st.title("🔮 Workforce Forecasting")
    st.markdown("""
    ## Predictive Workforce Analytics
    
    **Purpose**: Project future workforce trends using Facebook's Prophet algorithm.
    
    **Key Features**:
    - 10-year projections for key metrics
    - Confidence interval visualization
    - Trend decomposition
    
    *Why this matters*: Enables proactive workforce planning and policy development.
    """)
    st.markdown("""
    ### 10-Year Workforce Projections
    Generate forecasts using Facebook's Prophet algorithm.
    """)
    
    # Controls
    st.subheader("⚙️ Forecast Parameters")
    st.markdown("""
    *How to configure*:
    - Select specific age group to forecast
    - Choose metric to project
    - Set forecast horizon (1-10 years)
    """)
    col1, col2 = st.columns(2)
    with col1:
        selected_age = st.selectbox(
            "Age Group",
            df['age_group'].unique(),
            index=2
        )
    with col2:
        selected_metric = st.selectbox(
            "Metric",
            ["net_payroll", "new_subscribers", "exited", "rejoined"],
            format_func=lambda x: x.replace('_', ' ').title()
        )
    
    forecast_years = st.slider("Forecast Period (years)", 1, 10, 5)
    
    # Run forecast
    if st.button("Generate Forecast"):
        with st.spinner(f"Creating {forecast_years}-year forecast..."):
            try:
                # Prepare data
                forecast_data = df[df['age_group'] == selected_age][['year_end', selected_metric]]
                forecast_data.columns = ['ds', 'y']
                
                # Model and predict
                model = Prophet(yearly_seasonality=True)
                model.fit(forecast_data)
                future = model.make_future_dataframe(periods=forecast_years, freq='Y')
                forecast = model.predict(future)
                
                # Plot forecast - LARGE VISUALIZATION
                st.subheader("Forecast Results")
                st.markdown("""
                **Understanding the forecast**:
                - Black dots: Historical data
                - Blue line: Forecast trend
                - Shaded area: Confidence interval (80%)
                """)
                fig1 = model.plot(forecast, figsize=(18, 8))
                plt.title(f"{selected_metric.replace('_', ' ').title()} Forecast for {selected_age}", 
                         fontsize=16)
                plt.xlabel("Year", fontsize=14)
                plt.ylabel(selected_metric.replace('_', ' '), fontsize=14)
                st.pyplot(fig1)
                
                # Components
                st.subheader("Trend Components")
                st.markdown("""
                **What these show**:
                - **Trend**: Long-term direction
                - **Yearly**: Seasonal patterns
                """)
                fig2 = model.plot_components(forecast, figsize=(18, 8))
                st.pyplot(fig2)
                
                # Forecast data
                st.subheader("Forecast Values")
                st.markdown("""
                *Key columns*:
                - **Forecast**: Predicted value
                - **Bounds**: Range of likely outcomes
                """)
                forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_years)
                forecast_df['Year'] = forecast_df['ds'].dt.year
                st.dataframe(
                    forecast_df[['Year', 'yhat', 'yhat_lower', 'yhat_upper']]
                    .rename(columns={
                        'yhat': 'Forecast',
                        'yhat_lower': 'Lower Bound',
                        'yhat_upper': 'Upper Bound'
                    })
                    .style.format("{:,.0f}")
                )
                
            except Exception as e:
                st.error(f"Forecasting error: {str(e)}")
                st.markdown("""
                **Troubleshooting tips**:
                - Ensure sufficient historical data exists
                - Try different age groups or metrics
                - Reduce forecast period if needed
                """)

# Attrition Analysis Page
elif page == "Attrition Analysis":
    st.title("📊 Workforce Attrition Analyzer")
    st.markdown("""
    ## Understanding Workforce Attrition Patterns
    
    This page helps analyze workforce churn rates across different age groups, identifying high-risk segments 
    that need retention focus. Key features:
    
    - **Risk Categorization**: Groups age segments into High/Moderate/Low risk based on churn rates
    - **Trend Analysis**: Shows how churn patterns evolve over time for selected groups
    - **Benchmarking**: Compares groups against average and high-risk thresholds
    
    *Why this matters*: Identifying vulnerable age groups helps target retention policies and understand 
    workforce stability trends.
    """)
    
    # Year selection with default to most recent year
    selected_year = st.selectbox(
        "Select Year for Analysis", 
        sorted(df['year'].unique(), reverse=True),
        index=0
    )
    
    # Filter data
    year_df = df[df['year'] == selected_year].copy()
    
    # Calculate risk thresholds
    mean_churn = year_df['churn_rate'].mean()
    std_churn = year_df['churn_rate'].std()
    
    # Identify risk groups
    year_df['risk_category'] = np.where(
        year_df['churn_rate'] > mean_churn + std_churn, 'High Risk',
        np.where(
            year_df['churn_rate'] > mean_churn, 'Moderate Risk',
            'Low Risk'
        )
    )
    
    # Display risk overview
    st.subheader("🔍 Risk Overview")
    st.markdown("""
    *Key benchmarks for interpreting risk levels*:
    - **High Risk**: Above average + 1 standard deviation
    - **Moderate Risk**: Above average but below high risk threshold
    - **Low Risk**: Below average churn
    """)
    cols = st.columns(3)
    with cols[0]:
        st.metric("Highest Churn Rate", 
                f"{year_df['churn_rate'].max():.1%}",
                year_df.loc[year_df['churn_rate'].idxmax(), 'age_group'])
    with cols[1]:
        st.metric("Average Churn Rate", f"{mean_churn:.1%}")
    with cols[2]:
        st.metric("Risk Threshold", f"{(mean_churn + std_churn):.1%}")
    
    # Visual risk distribution
    st.subheader("📈 Risk Distribution by Age Group")
    st.markdown("""
    **How to read this chart**:
    - Colors show risk classification
    - Dashed lines indicate benchmark values
    - Hover for exact churn rates
    """)
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Create color mapping for risk categories
    palette = {'High Risk': '#ff6b6b', 'Moderate Risk': '#ffd166', 'Low Risk': '#06d6a0'}
    
    sns.barplot(
        data=year_df.sort_values('churn_rate', ascending=False),
        x='age_group',
        y='churn_rate',
        hue='risk_category',
        palette=palette,
        dodge=False,
        ax=ax
    )
    
    # Add threshold line
    ax.axhline(y=mean_churn + std_churn, color='#ef476f', linestyle='--', label='High Risk Threshold')
    ax.axhline(y=mean_churn, color='#ffd166', linestyle=':', label='Average')
    
    ax.set_title(f"Churn Rate Analysis ({selected_year})", fontsize=16, pad=20)
    ax.set_xlabel("Age Group", fontsize=12)
    ax.set_ylabel("Churn Rate", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.legend(title="Risk Level", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    st.pyplot(fig)
    
    # Trend analysis with larger visualization
    st.subheader("📉 Multi-Year Trend Analysis")
    st.markdown("""
    **Purpose**: Track how churn patterns evolve over time for selected groups.
    Compare if problematic groups are improving or worsening.
    """)
    trend_ages = st.multiselect(
        "Select Age Groups to Compare", 
        year_df['age_group'].unique(),
        default=year_df.nlargest(3, 'churn_rate')['age_group'].tolist()
    )
    
    if trend_ages:
        trend_df = df[df['age_group'].isin(trend_ages)]
        
        fig, ax = plt.subplots(figsize=(16, 6))
        
        for age in trend_ages:
            age_data = trend_df[trend_df['age_group'] == age]
            ax.plot(
                age_data['year'], 
                age_data['churn_rate'], 
                label=f"{age} (Latest: {age_data[age_data['year'] == selected_year]['churn_rate'].values[0]:.1%})",
                linewidth=2.5,
                marker='o'
            )
        
        ax.set_title("Churn Rate Trends Over Time", fontsize=16, pad=20)
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Churn Rate", fontsize=12)
        ax.legend(title="Age Group (Latest Year Value)", bbox_to_anchor=(1.05, 1))
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    
    # Detailed metrics table with formatting
    st.subheader("📋 Detailed Metrics")
    st.markdown("""
    **Data Dictionary**:
    - **Churn Rate**: % of workforce that exited (Exited/Total Workforce)
    - **Rejoin Rate**: % of workforce that rejoined (Rejoined/Total Workforce)
    - **Net Payroll**: New Subscribers - Exited + Rejoined
    """)
    display_cols = [
        'age_group', 'new_subscribers', 'exited', 'rejoined',
        'churn_rate', 'rejoin_rate', 'net_payroll'
    ]
    
    styled_df = year_df[display_cols].style.format({
        'new_subscribers': '{:,.0f}',
        'exited': '{:,.0f}',
        'rejoined': '{:,.0f}',
        'net_payroll': '{:,.0f}',
        'churn_rate': '{:.1%}',
        'rejoin_rate': '{:.1%}'
    }).background_gradient(
        subset=['churn_rate'], 
        cmap='RdYlGn_r', 
        vmin=year_df['churn_rate'].min(),
        vmax=year_df['churn_rate'].max()
    )
    
    st.dataframe(styled_df)

# Policy simulator page
elif page == "Policy Simulator":
    st.title("🔄 Workforce Policy Simulator")
    st.markdown("""
    ## Testing Workforce Policy Scenarios
    
    This interactive tool lets you simulate how changes in workforce policies might affect payroll dynamics.
    
    **Key Features**:
    - Adjust recruitment, retention, and re-engagement parameters
    - See immediate impact on net payroll
    - Compare effects across age groups
    
    *Why this matters*: Helps policymakers understand potential outcomes before implementing real changes.
    """)
    
    # Base year selection with default to most recent year
    base_year = st.selectbox(
        "Select Base Year for Simulation",
        sorted(df['year'].unique(), reverse=True),
        index=0
    )
    
    # Get base year data
    base_df = df[df['year'] == base_year].copy()
    
    # Policy controls with better organization
    st.subheader("Policy Levers")
    st.markdown("""
    Adjust these parameters to simulate different policy scenarios:
    - **Recruitment**: Changes in new worker acquisition
    - **Retention**: Changes in workforce exits
    - **Re-engagement**: Changes in workers returning
    """)
    with st.expander("Adjust Workforce Parameters"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Recruitment**")
            sub_change = st.slider(
                "New Subscribers Change (%)",
                -50, 100, 0,
                help="Percentage change in new EPF subscribers"
            )
        
        with col2:
            st.markdown("**Retention**")
            exit_change = st.slider(
                "Exits Change (%)",
                -50, 50, 0,
                help="Percentage change in workforce exits"
            )
        
        with col3:
            st.markdown("**Re-engagement**")
            rejoin_change = st.slider(
                "Rejoins Change (%)",
                -50, 100, 0,
                help="Percentage change in workforce re-entries"
            )
    
    # Apply changes to create simulated scenario
    simulated_df = base_df.copy()
    simulated_df['new_subscribers'] *= (1 + sub_change / 100)
    simulated_df['exited'] *= (1 + exit_change / 100)
    simulated_df['rejoined'] *= (1 + rejoin_change / 100)
    simulated_df['net_payroll'] = simulated_df['new_subscribers'] - simulated_df['exited'] + simulated_df['rejoined']
    
    # Results visualization
    st.subheader("📊 Simulation Results")
    st.markdown("""
    **Understanding the Results**:
    - Positive values (green) indicate improvements over baseline
    - Negative values (red) show potential worsening
    - Hover over metrics for baseline values
    """)
    # Summary metrics comparison
    base_totals = base_df[['new_subscribers', 'exited', 'rejoined', 'net_payroll']].sum()
    sim_totals = simulated_df[['new_subscribers', 'exited', 'rejoined', 'net_payroll']].sum()
    
    cols = st.columns(4)
    metrics = [
        ('New Subscribers', '📈', 'person-add'),
        ('Exited', '📉', 'person-remove'),
        ('Rejoined', '🔄', 'refresh'),
        ('Net Payroll', '💰', 'cash')
    ]
    
    for col, (name, emoji, _) in zip(cols, metrics):
        # Explicit column name mapping
        if name == 'New Subscribers':
            col_name = 'new_subscribers'
        elif name == 'Exited':
            col_name = 'exited'
        elif name == 'Rejoined':
            col_name = 'rejoined'
        else:  # Net Payroll
            col_name = 'net_payroll'
        
        base_val = base_totals[col_name]
        sim_val = sim_totals[col_name]
        change = (sim_val - base_val) / base_val * 100 if base_val != 0 else 0
        
        col.metric(
            f"{emoji} {name}",
            f"{sim_val:,.0f}",
            f"{change:+.1f}%",
            help=f"Base value: {base_val:,.0f}"
        )
    
    # Age group impact analysis
    st.subheader("🧑‍💼 Impact by Age Group")
    st.markdown("""
    **Key Insights**:
    - Which age groups benefit most from your policy changes?
    - Are there groups that respond differently to the same policies?
    """)
    # Merge base and simulated data
    comparison_df = pd.merge(
        base_df[['age_group', 'new_subscribers', 'exited', 'rejoined', 'net_payroll']],
        simulated_df[['age_group', 'new_subscribers', 'exited', 'rejoined', 'net_payroll']],
        on='age_group',
        suffixes=('_base', '_sim')
    )
    
    # Calculate changes
    for metric in ['new_subscribers', 'exited', 'rejoined', 'net_payroll']:
        comparison_df[f'{metric}_change'] = comparison_df[f'{metric}_sim'] - comparison_df[f'{metric}_base']
        comparison_df[f'{metric}_change_pct'] = (comparison_df[f'{metric}_change'] / comparison_df[f'{metric}_base']) * 100
    
    # Visualization
    tab1, tab2 = st.tabs(["Net Payroll Impact", "Component Changes"])
    
    with tab1:
        fig, ax = plt.subplots(figsize=(16, 6))
        
        x = np.arange(len(comparison_df))
        width = 0.35
        
        ax.bar(x - width/2, comparison_df['net_payroll_base'], width, label='Base Scenario', color='#3498db')
        ax.bar(x + width/2, comparison_df['net_payroll_sim'], width, label='Simulated', color='#2ecc71')
        
        ax.set_title("Net Payroll Comparison by Age Group", fontsize=16, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['age_group'])
        ax.set_ylabel("Net Payroll")
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with tab2:
        fig, ax = plt.subplots(figsize=(16, 6))
        
        metrics = ['new_subscribers', 'exited', 'rejoined']
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        bottom = np.zeros(len(comparison_df))
        
        for metric, color in zip(metrics, colors):
            ax.bar(
                comparison_df['age_group'],
                comparison_df[f'{metric}_change'],
                label=metric.replace('_', ' ').title(),
                color=color,
                bottom=bottom
            )
            bottom += comparison_df[f'{metric}_change']
        
        ax.set_title("Component Changes by Age Group", fontsize=16, pad=20)
        ax.set_ylabel("Change from Base Scenario")
        ax.legend(bbox_to_anchor=(1.05, 1))
        ax.grid(True, axis='y', alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    # Detailed data table
    st.subheader("📋 Detailed Comparison")
    st.markdown("""
    **Interpreting the Table**:
    - Δ% columns show percentage change from baseline
    - Color gradient highlights strongest impacts
    """)
    display_cols = [
        'age_group',
        'new_subscribers_base', 'new_subscribers_sim', 'new_subscribers_change_pct',
        'exited_base', 'exited_sim', 'exited_change_pct',
        'rejoined_base', 'rejoined_sim', 'rejoined_change_pct',
        'net_payroll_base', 'net_payroll_sim', 'net_payroll_change_pct'
    ]
    
    styled_df = comparison_df[display_cols].rename(columns={
        'new_subscribers_base': 'Subscribers (Base)',
        'new_subscribers_sim': 'Subscribers (Sim)',
        'new_subscribers_change_pct': 'Subscribers Δ%',
        'exited_base': 'Exited (Base)',
        'exited_sim': 'Exited (Sim)',
        'exited_change_pct': 'Exited Δ%',
        'rejoined_base': 'Rejoined (Base)',
        'rejoined_sim': 'Rejoined (Sim)',
        'rejoined_change_pct': 'Rejoined Δ%',
        'net_payroll_base': 'Net Payroll (Base)',
        'net_payroll_sim': 'Net Payroll (Sim)',
        'net_payroll_change_pct': 'Net Payroll Δ%'
    }).style.format({
        'Subscribers (Base)': '{:,.0f}',
        'Subscribers (Sim)': '{:,.0f}',
        'Subscribers Δ%': '{:+.1f}%',
        'Exited (Base)': '{:,.0f}',
        'Exited (Sim)': '{:,.0f}',
        'Exited Δ%': '{:+.1f}%',
        'Rejoined (Base)': '{:,.0f}',
        'Rejoined (Sim)': '{:,.0f}',
        'Rejoined Δ%': '{:+.1f}%',
        'Net Payroll (Base)': '{:,.0f}',
        'Net Payroll (Sim)': '{:,.0f}',
        'Net Payroll Δ%': '{:+.1f}%'
    }).background_gradient(
        subset=['Net Payroll Δ%'],
        cmap='RdYlGn',
        vmin=-50,
        vmax=50
    )
    
    st.dataframe(styled_df)