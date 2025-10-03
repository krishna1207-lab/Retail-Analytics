import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

def format_number(num, prefix='$'):
    """Format numbers with K/M notation and commas"""
    if pd.isna(num) or num == 0:
        return f"{prefix}0"
    
    if abs(num) >= 1_000_000:
        return f"{prefix}{num/1_000_000:.1f}M"
    elif abs(num) >= 1_000:
        return f"{prefix}{num/1_000:.1f}K"
    else:
        return f"{prefix}{num:,.0f}"

def format_percentage(num):
    """Format percentage with 1 decimal place"""
    return f"{num:.1f}%"

# Page configuration
st.set_page_config(
    page_title="Retail Analytics Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .kpi-container {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .insight-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin-bottom: 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    """Load and preprocess the data"""
    try:
        df = pd.read_csv('customer_shopping.csv')
        
        # Convert date column
        df['invoice_date'] = pd.to_datetime(df['invoice_date'], format='%d/%m/%Y', errors='coerce')
        # Fill any NaT values with alternative format
        mask = df['invoice_date'].isna()
        if mask.any():
            df.loc[mask, 'invoice_date'] = pd.to_datetime(df.loc[mask, 'invoice_date'], format='%m/%d/%Y', errors='coerce')
        
        # Remove rows with invalid dates
        df = df.dropna(subset=['invoice_date'])
        
        # Add derived columns
        df['month'] = df['invoice_date'].dt.month
        df['quarter'] = df['invoice_date'].dt.quarter
        df['year'] = df['invoice_date'].dt.year
        df['day_of_week'] = df['invoice_date'].dt.day_name()
        df['month_year'] = df['invoice_date'].dt.to_period('M')
        df['quarter_year'] = df['invoice_date'].dt.to_period('Q')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def store_performance_analysis(df):
    """Store vs Region Performance Analysis"""
    st.header("🏪 Store vs Region Performance")
    
    if df is None or df.empty:
        st.error("No data available for analysis")
        return
    
    # Calculate store performance metrics
    store_performance = df.groupby('shopping_mall').agg({
        'price': ['sum', 'count', 'mean'],
        'customer_id': 'nunique',
        'quantity': 'sum'
    }).round(2)
    
    store_performance.columns = ['total_revenue', 'transaction_count', 'avg_order_value', 'unique_customers', 'total_quantity']
    store_performance = store_performance.sort_values('total_revenue', ascending=False)
    
    # Calculate percentage shares
    total_revenue = store_performance['total_revenue'].sum()
    total_transactions = store_performance['transaction_count'].sum()
    total_customers = store_performance['unique_customers'].sum()
    
    store_performance['revenue_share'] = (store_performance['total_revenue'] / total_revenue * 100).round(2)
    store_performance['transaction_share'] = (store_performance['transaction_count'] / total_transactions * 100).round(2)
    store_performance['customer_share'] = (store_performance['unique_customers'] / total_customers * 100).round(2)
    
    # Top KPIs
    st.markdown('<div class="kpi-container">', unsafe_allow_html=True)
    st.subheader("📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Total Stores", 
            f"{len(store_performance):,}",
            help="Number of unique stores in the dataset"
        )
    with col2:
        st.metric(
            "Total Revenue", 
            format_number(total_revenue),
            help="Sum of all revenue across all stores"
        )
    with col3:
        st.metric(
            "Total Transactions", 
            f"{total_transactions:,}",
            help="Total number of transactions across all stores"
        )
    with col4:
        st.metric(
            "Avg Revenue per Store", 
            format_number(store_performance['total_revenue'].mean()),
            help="Average revenue per store"
        )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Visualizations
    st.subheader("📈 Performance Visualizations")
    
    # Revenue by Store
    col1, col2 = st.columns(2)
    
    with col1:
        fig_revenue = px.bar(
            x=store_performance.index,
            y=store_performance['total_revenue'],
            title="Revenue by Store",
            labels={'x': 'Store', 'y': 'Revenue ($)'},
            color=store_performance['total_revenue'],
            color_continuous_scale='Blues'
        )
        fig_revenue.update_layout(
            xaxis_tickangle=45, 
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig_revenue, use_container_width=True)
    
    with col2:
        fig_transactions = px.bar(
            x=store_performance.index,
            y=store_performance['transaction_count'],
            title="Transaction Volume by Store",
            labels={'x': 'Store', 'y': 'Transactions'},
            color=store_performance['transaction_count'],
            color_continuous_scale='Greens'
        )
        fig_transactions.update_layout(
            xaxis_tickangle=45, 
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig_transactions, use_container_width=True)
    
    # Average Order Value and Customer Count
    col1, col2 = st.columns(2)
    
    with col1:
        fig_aov = px.bar(
            x=store_performance.index,
            y=store_performance['avg_order_value'],
            title="Average Order Value by Store",
            labels={'x': 'Store', 'y': 'AOV ($)'},
            color=store_performance['avg_order_value'],
            color_continuous_scale='Oranges'
        )
        fig_aov.update_layout(
            xaxis_tickangle=45, 
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig_aov, use_container_width=True)
    
    with col2:
        fig_customers = px.bar(
            x=store_performance.index,
            y=store_performance['unique_customers'],
            title="Customer Count by Store",
            labels={'x': 'Store', 'y': 'Customers'},
            color=store_performance['unique_customers'],
            color_continuous_scale='Purples'
        )
        fig_customers.update_layout(
            xaxis_tickangle=45, 
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig_customers, use_container_width=True)
    
    # Store Performance Summary Table
    st.subheader("📊 Store Performance Summary")
    
    # Add rankings
    store_performance['revenue_rank'] = store_performance['total_revenue'].rank(ascending=False, method='dense').astype(int)
    store_performance['transaction_rank'] = store_performance['transaction_count'].rank(ascending=False, method='dense').astype(int)
    store_performance['aov_rank'] = store_performance['avg_order_value'].rank(ascending=False, method='dense').astype(int)
    store_performance['customer_rank'] = store_performance['unique_customers'].rank(ascending=False, method='dense').astype(int)
    
    # Prepare display dataframe
    display_df = store_performance.reset_index()
    display_df = display_df[[
        'shopping_mall', 'total_revenue', 'revenue_share', 'revenue_rank',
        'transaction_count', 'transaction_share', 'transaction_rank',
        'avg_order_value', 'aov_rank', 'unique_customers', 'customer_share', 'customer_rank'
    ]]
    
    display_df.columns = [
        'Store', 'Revenue ($)', 'Revenue %', 'Revenue Rank',
        'Transactions', 'Transaction %', 'Transaction Rank',
        'Avg Order Value ($)', 'AOV Rank', 'Customers', 'Customer %', 'Customer Rank'
    ]
    
    # Format the dataframe for better display
    display_df['Revenue ($)'] = display_df['Revenue ($)'].apply(lambda x: format_number(x))
    display_df['Revenue %'] = display_df['Revenue %'].apply(lambda x: format_percentage(x))
    display_df['Transaction %'] = display_df['Transaction %'].apply(lambda x: format_percentage(x))
    display_df['Avg Order Value ($)'] = display_df['Avg Order Value ($)'].apply(lambda x: format_number(x))
    display_df['Customer %'] = display_df['Customer %'].apply(lambda x: format_percentage(x))
    
    st.dataframe(display_df, use_container_width=True)
    
    # Performance Insights
    st.subheader("💡 Performance Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        best_store = store_performance.index[0]
        best_revenue = store_performance.iloc[0]['total_revenue']
        best_share = store_performance.iloc[0]['revenue_share']
        st.metric(
            "Best Performing Store", 
            f"{best_store}", 
            f"{format_number(best_revenue)} ({format_percentage(best_share)})"
        )
    
    with col2:
        revenue_gap = store_performance.iloc[0]['total_revenue'] - store_performance.iloc[-1]['total_revenue']
        st.metric("Revenue Gap (Best vs Worst)", format_number(revenue_gap))
    
    with col3:
        avg_transactions = store_performance['transaction_count'].mean()
        st.metric("Average Transactions per Store", f"{avg_transactions:,.0f}")
    
    # Top and Bottom Performers
    st.subheader("🏆 Top & Bottom Performers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.subheader("🏆 Top 3 Performers")
        top_3 = store_performance.head(3)
        for idx, (store, row) in enumerate(top_3.iterrows(), 1):
            st.write(f"**{idx}. {store}**")
            st.write(f"   Revenue: {format_number(row['total_revenue'])} ({format_percentage(row['revenue_share'])})")
            st.write(f"   Transactions: {row['transaction_count']:,} ({format_percentage(row['transaction_share'])})")
            st.write(f"   Customers: {row['unique_customers']:,} ({format_percentage(row['customer_share'])})")
            st.write("---")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="insight-box">', unsafe_allow_html=True)
        st.subheader("📉 Bottom 3 Performers")
        bottom_3 = store_performance.tail(3)
        for idx, (store, row) in enumerate(bottom_3.iterrows(), 1):
            st.write(f"**{idx}. {store}**")
            st.write(f"   Revenue: {format_number(row['total_revenue'])} ({format_percentage(row['revenue_share'])})")
            st.write(f"   Transactions: {row['transaction_count']:,} ({format_percentage(row['transaction_share'])})")
            st.write(f"   Customers: {row['unique_customers']:,} ({format_percentage(row['customer_share'])})")
            st.write("---")
        st.markdown('</div>', unsafe_allow_html=True)

def business_intelligence(df):
    """Business Intelligence Analysis"""
    st.header("📊 Business Intelligence")
    
    if df is None or df.empty:
        st.error("No data available for analysis")
        return
    
    # 1. Monthly and Quarterly Sales Analysis
    st.subheader("📅 Monthly & Quarterly Sales Analysis")
    
    # Monthly sales data
    monthly_sales = df.groupby('month_year').agg({
        'price': ['sum', 'count'],
        'customer_id': 'nunique',
        'quantity': 'sum'
    }).round(2)
    
    monthly_sales.columns = ['revenue', 'transactions', 'customers', 'quantity']
    monthly_sales['month_name'] = monthly_sales.index.strftime('%b %Y')
    
    # Quarterly sales data
    quarterly_sales = df.groupby('quarter_year').agg({
        'price': ['sum', 'count'],
        'customer_id': 'nunique',
        'quantity': 'sum'
    }).round(2)
    
    quarterly_sales.columns = ['revenue', 'transactions', 'customers', 'quantity']
    quarterly_sales['quarter_name'] = quarterly_sales.index.astype(str)
    
    # Monthly sales visualization
    col1, col2 = st.columns(2)
    
    with col1:
        fig_monthly = px.line(
            x=monthly_sales['month_name'],
            y=monthly_sales['revenue'],
            title="Monthly Revenue Trend",
            labels={'x': 'Month', 'y': 'Revenue ($)'},
            markers=True
        )
        fig_monthly.update_layout(xaxis_tickangle=45, height=400)
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    with col2:
        fig_quarterly = px.bar(
            x=quarterly_sales['quarter_name'],
            y=quarterly_sales['revenue'],
            title="Quarterly Revenue",
            labels={'x': 'Quarter', 'y': 'Revenue ($)'},
            color=quarterly_sales['revenue'],
            color_continuous_scale='Viridis'
        )
        fig_quarterly.update_layout(height=400)
        st.plotly_chart(fig_quarterly, use_container_width=True)
    
    # Seasonal pattern analysis
    st.subheader("🌊 Seasonal Pattern Analysis")
    
    # Extract month names for seasonal analysis
    df['month_name'] = df['invoice_date'].dt.strftime('%b')
    monthly_pattern = df.groupby('month_name').agg({
        'price': ['sum', 'count', 'mean'],
        'customer_id': 'nunique'
    }).round(2)
    
    monthly_pattern.columns = ['revenue', 'transactions', 'avg_order_value', 'customers']
    
    # Order months properly
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_pattern = monthly_pattern.reindex(month_order)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_seasonal = px.bar(
            x=monthly_pattern.index,
            y=monthly_pattern['revenue'],
            title="Seasonal Revenue Pattern",
            labels={'x': 'Month', 'y': 'Revenue ($)'},
            color=monthly_pattern['revenue'],
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_seasonal, use_container_width=True)
    
    with col2:
        fig_seasonal_customers = px.line(
            x=monthly_pattern.index,
            y=monthly_pattern['customers'],
            title="Seasonal Customer Pattern",
            labels={'x': 'Month', 'y': 'Number of Customers'},
            markers=True
        )
        st.plotly_chart(fig_seasonal_customers, use_container_width=True)
    
    # Seasonal insights
    st.subheader("💡 Seasonal Insights")
    
    best_month = monthly_pattern['revenue'].idxmax()
    worst_month = monthly_pattern['revenue'].idxmin()
    best_revenue = monthly_pattern.loc[best_month, 'revenue']
    worst_revenue = monthly_pattern.loc[worst_month, 'revenue']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Best Performing Month", best_month, format_number(best_revenue))
    with col2:
        st.metric("Worst Performing Month", worst_month, format_number(worst_revenue))
    with col3:
        seasonal_variation = ((best_revenue - worst_revenue) / worst_revenue * 100)
        st.metric("Seasonal Variation", format_percentage(seasonal_variation))
    
    # 2. Payment Method Analysis
    st.subheader("💳 Payment Method Analysis")
    
    # Payment method distribution
    payment_analysis = df.groupby('payment_method').agg({
        'price': ['sum', 'count', 'mean'],
        'customer_id': 'nunique'
    }).round(2)
    
    payment_analysis.columns = ['revenue', 'transactions', 'avg_order_value', 'customers']
    payment_analysis['revenue_share'] = (payment_analysis['revenue'] / payment_analysis['revenue'].sum() * 100).round(2)
    payment_analysis['transaction_share'] = (payment_analysis['transactions'] / payment_analysis['transactions'].sum() * 100).round(2)
    
    # Payment method visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig_payment_pie = px.pie(
            values=payment_analysis['revenue'],
            names=payment_analysis.index,
            title="Revenue Distribution by Payment Method",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_payment_pie, use_container_width=True)
    
    with col2:
        fig_payment_bar = px.bar(
            x=payment_analysis.index,
            y=payment_analysis['transactions'],
            title="Transaction Count by Payment Method",
            labels={'x': 'Payment Method', 'y': 'Number of Transactions'},
            color=payment_analysis['transactions'],
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig_payment_bar, use_container_width=True)
    
    # Payment method by region
    st.subheader("🌍 Payment Method by Region")
    
    payment_region = df.groupby(['shopping_mall', 'payment_method']).agg({
        'price': 'sum',
        'customer_id': 'nunique'
    }).round(2)
    
    payment_region.columns = ['revenue', 'customers']
    payment_region = payment_region.reset_index()
    
    fig_payment_region = px.bar(
        payment_region,
        x='shopping_mall',
        y='revenue',
        color='payment_method',
        title="Payment Method Usage by Store",
        labels={'x': 'Store', 'y': 'Revenue ($)'},
        barmode='group'
    )
    fig_payment_region.update_layout(xaxis_tickangle=45, height=500)
    st.plotly_chart(fig_payment_region, use_container_width=True)
    
    # Payment method insights
    st.subheader("💡 Payment Method Insights")
    
    most_popular = payment_analysis['revenue'].idxmax()
    most_revenue = payment_analysis.loc[most_popular, 'revenue']
    most_share = payment_analysis.loc[most_popular, 'revenue_share']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Most Popular Payment", most_popular, format_number(most_revenue))
    with col2:
        st.metric("Revenue Share", format_percentage(most_share))
    with col3:
        avg_aov = payment_analysis['avg_order_value'].mean()
        st.metric("Average AOV", format_number(avg_aov))
    
    # 3. Category-wise Insights
    st.subheader("🏷️ Category-wise Insights")
    
    # Category analysis
    category_analysis = df.groupby('category').agg({
        'price': ['sum', 'count', 'mean'],
        'customer_id': 'nunique',
        'quantity': 'sum'
    }).round(2)
    
    category_analysis.columns = ['revenue', 'transactions', 'avg_order_value', 'customers', 'quantity']
    category_analysis['revenue_share'] = (category_analysis['revenue'] / category_analysis['revenue'].sum() * 100).round(2)
    category_analysis['customer_share'] = (category_analysis['customers'] / category_analysis['customers'].sum() * 100).round(2)
    category_analysis['profitability_score'] = (category_analysis['revenue'] * category_analysis['customers']).round(2)
    
    # Sort by profitability
    category_analysis = category_analysis.sort_values('profitability_score', ascending=False)
    
    # Category visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig_category_revenue = px.bar(
            x=category_analysis.index,
            y=category_analysis['revenue'],
            title="Revenue by Category",
            labels={'x': 'Category', 'y': 'Revenue ($)'},
            color=category_analysis['revenue'],
            color_continuous_scale='Viridis'
        )
        fig_category_revenue.update_layout(xaxis_tickangle=45, height=400)
        st.plotly_chart(fig_category_revenue, use_container_width=True)
    
    with col2:
        fig_category_customers = px.bar(
            x=category_analysis.index,
            y=category_analysis['customers'],
            title="Customer Count by Category",
            labels={'x': 'Category', 'y': 'Number of Customers'},
            color=category_analysis['customers'],
            color_continuous_scale='Plasma'
        )
        fig_category_customers.update_layout(xaxis_tickangle=45, height=400)
        st.plotly_chart(fig_category_customers, use_container_width=True)
    
    # Category profitability scatter
    fig_profitability = px.scatter(
        category_analysis,
        x='customers',
        y='revenue',
        size='avg_order_value',
        color='revenue_share',
        title="Category Profitability Analysis",
        labels={'x': 'Number of Customers', 'y': 'Revenue ($)'},
        hover_data=['transactions', 'avg_order_value']
    )
    st.plotly_chart(fig_profitability, use_container_width=True)
    
    # Category insights
    st.subheader("💡 Category Insights")
    
    top_category = category_analysis.index[0]
    top_revenue = category_analysis.iloc[0]['revenue']
    top_customers = category_analysis.iloc[0]['customers']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Most Profitable Category", top_category, format_number(top_revenue))
    with col2:
        st.metric("Customer Count", f"{top_customers:,}")
    with col3:
        top_share = category_analysis.iloc[0]['revenue_share']
        st.metric("Revenue Share", format_percentage(top_share))
    
    # Summary tables
    st.subheader("📋 Business Intelligence Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Payment Method Summary**")
        payment_summary = payment_analysis.reset_index()
        payment_summary = payment_summary[['payment_method', 'revenue', 'revenue_share', 'transactions', 'transaction_share', 'avg_order_value']]
        payment_summary.columns = ['Payment Method', 'Revenue ($)', 'Revenue Share (%)', 'Transactions', 'Transaction Share (%)', 'Avg Order Value ($)']
        
        # Format numbers
        payment_summary['Revenue ($)'] = payment_summary['Revenue ($)'].apply(lambda x: format_number(x))
        payment_summary['Revenue Share (%)'] = payment_summary['Revenue Share (%)'].apply(lambda x: format_percentage(x))
        payment_summary['Transaction Share (%)'] = payment_summary['Transaction Share (%)'].apply(lambda x: format_percentage(x))
        payment_summary['Avg Order Value ($)'] = payment_summary['Avg Order Value ($)'].apply(lambda x: format_number(x))
        
        st.dataframe(payment_summary, use_container_width=True)
    
    with col2:
        st.write("**Category Performance Summary**")
        category_summary = category_analysis.reset_index()
        category_summary = category_summary[['category', 'revenue', 'revenue_share', 'customers', 'customer_share', 'avg_order_value']]
        category_summary.columns = ['Category', 'Revenue ($)', 'Revenue Share (%)', 'Customers', 'Customer Share (%)', 'Avg Order Value ($)']
        
        # Format numbers
        category_summary['Revenue ($)'] = category_summary['Revenue ($)'].apply(lambda x: format_number(x))
        category_summary['Revenue Share (%)'] = category_summary['Revenue Share (%)'].apply(lambda x: format_percentage(x))
        category_summary['Customer Share (%)'] = category_summary['Customer Share (%)'].apply(lambda x: format_percentage(x))
        category_summary['Avg Order Value ($)'] = category_summary['Avg Order Value ($)'].apply(lambda x: format_number(x))
        
        st.dataframe(category_summary, use_container_width=True)

def customer_analysis(df):
    """Comprehensive Customer Analysis"""
    st.header("👥 Customer Analysis")
    
    if df is None or df.empty:
        st.error("No data available for analysis")
        return
    
    # Calculate customer metrics
    customer_metrics = df.groupby('customer_id').agg({
        'price': ['sum', 'count', 'mean'],
        'quantity': 'sum',
        'invoice_date': ['min', 'max']
    }).round(2)
    
    customer_metrics.columns = ['total_spend', 'transaction_count', 'avg_order_value', 'total_quantity', 'first_purchase', 'last_purchase']
    
    # Calculate recency (days since last purchase)
    current_date = df['invoice_date'].max()
    customer_metrics['recency_days'] = (current_date - customer_metrics['last_purchase']).dt.days
    
    # Calculate frequency (number of transactions)
    customer_metrics['frequency'] = customer_metrics['transaction_count']
    
    # Calculate monetary value (total spend)
    customer_metrics['monetary'] = customer_metrics['total_spend']
    
    # 1. High vs Low-value Segmentation
    st.subheader("💎 High vs Low-value Customer Segmentation")
    
    # Create value segments based on total spend
    high_value_threshold = customer_metrics['total_spend'].quantile(0.8)
    medium_value_threshold = customer_metrics['total_spend'].quantile(0.5)
    
    def assign_value_segment(spend):
        if spend >= high_value_threshold:
            return 'High Value'
        elif spend >= medium_value_threshold:
            return 'Medium Value'
        else:
            return 'Low Value'
    
    customer_metrics['value_segment'] = customer_metrics['total_spend'].apply(assign_value_segment)
    
    # Value segment analysis
    segment_analysis = customer_metrics.groupby('value_segment').agg({
        'total_spend': ['sum', 'count', 'mean'],
        'transaction_count': 'mean',
        'avg_order_value': 'mean',
        'recency_days': 'mean'
    }).round(2)
    
    segment_analysis.columns = ['total_spend', 'customer_count', 'avg_spend', 'avg_transactions', 'avg_order_value', 'avg_recency']
    segment_analysis['percentage'] = (segment_analysis['customer_count'] / len(customer_metrics) * 100).round(1)
    segment_analysis['spend_share'] = (segment_analysis['total_spend'] / customer_metrics['total_spend'].sum() * 100).round(1)
    
    # Key metrics for value segmentation
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        high_value_count = segment_analysis.loc['High Value', 'customer_count']
        high_value_pct = segment_analysis.loc['High Value', 'percentage']
        st.metric("High Value Customers", f"{high_value_count:,.0f}", f"{high_value_pct:.1f}%")
    with col2:
        medium_value_count = segment_analysis.loc['Medium Value', 'customer_count']
        medium_value_pct = segment_analysis.loc['Medium Value', 'percentage']
        st.metric("Medium Value Customers", f"{medium_value_count:,.0f}", f"{medium_value_pct:.1f}%")
    with col3:
        low_value_count = segment_analysis.loc['Low Value', 'customer_count']
        low_value_pct = segment_analysis.loc['Low Value', 'percentage']
        st.metric("Low Value Customers", f"{low_value_count:,.0f}", f"{low_value_pct:.1f}%")
    with col4:
        total_customers = len(customer_metrics)
        st.metric("Total Customers", f"{total_customers:,}")
    
    # Value segmentation pie chart
    col1, col2 = st.columns(2)
    
    with col1:
        fig_value_pie = px.pie(
            values=segment_analysis['customer_count'],
            names=segment_analysis.index,
            title="Customer Distribution by Value Segment",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_value_pie, use_container_width=True)
    
    with col2:
        fig_value_revenue = px.pie(
            values=segment_analysis['total_spend'],
            names=segment_analysis.index,
            title="Revenue Distribution by Value Segment",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_value_revenue, use_container_width=True)
    
    # 2. Top 10% Customers Analysis
    st.subheader("👑 Top 10% Customers Analysis")
    
    # Identify top 10% customers
    top_10_percent = int(len(customer_metrics) * 0.1)
    top_customers = customer_metrics.nlargest(top_10_percent, 'total_spend')
    
    # Top customers metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Top 10% Customers", f"{len(top_customers):,}")
    with col2:
        top_revenue = top_customers['total_spend'].sum()
        total_revenue = customer_metrics['total_spend'].sum()
        revenue_share = (top_revenue / total_revenue * 100)
        st.metric("Revenue Share", f"{revenue_share:.1f}%")
    with col3:
        avg_top_spend = top_customers['total_spend'].mean()
        st.metric("Avg Spend (Top 10%)", format_number(avg_top_spend))
    with col4:
        avg_all_spend = customer_metrics['total_spend'].mean()
        multiplier = avg_top_spend / avg_all_spend
        st.metric("Spend Multiplier", f"{multiplier:.1f}x")
    
    # Top customers visualization
    col1, col2 = st.columns(2)
    
    with col1:
        fig_top_customers = px.bar(
            x=top_customers.head(20).index,
            y=top_customers.head(20)['total_spend'],
            title="Top 20 Customers by Spend",
            labels={'x': 'Customer ID', 'y': 'Total Spend ($)'},
            color=top_customers.head(20)['total_spend'],
            color_continuous_scale='Blues'
        )
        fig_top_customers.update_layout(xaxis_tickangle=45, showlegend=False)
        st.plotly_chart(fig_top_customers, use_container_width=True)
    
    with col2:
        # Customer spend distribution
        fig_spend_dist = px.histogram(
            customer_metrics,
            x='total_spend',
            nbins=50,
            title="Customer Spend Distribution",
            labels={'x': 'Total Spend ($)', 'y': 'Number of Customers'},
            color_discrete_sequence=['#1f77b4']
        )
        # Add vertical line for top 10% threshold
        threshold = top_customers['total_spend'].min()
        fig_spend_dist.add_vline(x=threshold, line_dash="dash", line_color="red", 
                                annotation_text=f"Top 10% threshold: {format_number(threshold)}")
        st.plotly_chart(fig_spend_dist, use_container_width=True)
    
    # 3. RFM Analysis
    st.subheader("📊 RFM Analysis (Recency, Frequency, Monetary)")
    
    # Calculate RFM scores (1-5 scale)
    # Handle duplicates by using rank method
    customer_metrics['R_score'] = pd.qcut(customer_metrics['recency_days'].rank(method='first'), 5, labels=[5,4,3,2,1], duplicates='drop')
    customer_metrics['F_score'] = pd.qcut(customer_metrics['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop')
    customer_metrics['M_score'] = pd.qcut(customer_metrics['monetary'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop')
    
    # Convert to numeric
    customer_metrics['R_score'] = customer_metrics['R_score'].astype(int)
    customer_metrics['F_score'] = customer_metrics['F_score'].astype(int)
    customer_metrics['M_score'] = customer_metrics['M_score'].astype(int)
    
    # Create RFM segments
    def assign_rfm_segment(row):
        r, f, m = row['R_score'], row['F_score'], row['M_score']
        
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        elif r >= 3 and f >= 3 and m >= 3:
            return 'Loyal Customers'
        elif r >= 3 and f >= 2 and m >= 2:
            return 'Potential Loyalists'
        elif r >= 4 and f <= 2 and m <= 2:
            return 'New Customers'
        elif r >= 3 and f <= 2 and m >= 3:
            return 'At Risk'
        elif r <= 2 and f >= 3 and m >= 3:
            return 'Cannot Lose Them'
        elif r <= 2 and f >= 2 and m >= 2:
            return 'Hibernating'
        else:
            return 'Lost'
    
    customer_metrics['RFM_segment'] = customer_metrics.apply(assign_rfm_segment, axis=1)
    
    # RFM segment analysis
    rfm_analysis = customer_metrics.groupby('RFM_segment').agg({
        'total_spend': ['sum', 'count', 'mean'],
        'transaction_count': 'mean',
        'recency_days': 'mean'
    }).round(2)
    
    rfm_analysis.columns = ['total_spend', 'customer_count', 'avg_spend', 'avg_transactions', 'avg_recency']
    rfm_analysis['percentage'] = (rfm_analysis['customer_count'] / len(customer_metrics) * 100).round(1)
    rfm_analysis['spend_share'] = (rfm_analysis['total_spend'] / customer_metrics['total_spend'].sum() * 100).round(1)
    
    # RFM visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig_rfm_pie = px.pie(
            values=rfm_analysis['customer_count'],
            names=rfm_analysis.index,
            title="Customer Distribution by RFM Segment",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_rfm_pie, use_container_width=True)
    
    with col2:
        fig_rfm_revenue = px.bar(
            x=rfm_analysis.index,
            y=rfm_analysis['total_spend'],
            title="Revenue by RFM Segment",
            labels={'x': 'RFM Segment', 'y': 'Total Spend ($)'},
            color=rfm_analysis['total_spend'],
            color_continuous_scale='Greens'
        )
        fig_rfm_revenue.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_rfm_revenue, use_container_width=True)
    
    # 4. Repeat vs One-time Customers
    st.subheader("🔄 Repeat vs One-time Customers")
    
    # Identify repeat vs one-time customers
    repeat_customers = customer_metrics[customer_metrics['transaction_count'] > 1]
    one_time_customers = customer_metrics[customer_metrics['transaction_count'] == 1]
    
    # Calculate metrics
    repeat_count = len(repeat_customers)
    one_time_count = len(one_time_customers)
    total_customers = len(customer_metrics)
    
    repeat_revenue = repeat_customers['total_spend'].sum()
    one_time_revenue = one_time_customers['total_spend'].sum()
    total_revenue = customer_metrics['total_spend'].sum()
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Repeat Customers", f"{repeat_count:,}", f"{(repeat_count/total_customers*100):.1f}%")
    with col2:
        st.metric("One-time Customers", f"{one_time_count:,}", f"{(one_time_count/total_customers*100):.1f}%")
    with col3:
        st.metric("Repeat Revenue Share", f"{(repeat_revenue/total_revenue*100):.1f}%")
    with col4:
        st.metric("One-time Revenue Share", f"{(one_time_revenue/total_revenue*100):.1f}%")
    
    # Repeat vs One-time visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Customer count comparison
        fig_customer_type = px.pie(
            values=[repeat_count, one_time_count],
            names=['Repeat Customers', 'One-time Customers'],
            title="Customer Type Distribution",
            color_discrete_sequence=['#2E8B57', '#FF6B6B']
        )
        st.plotly_chart(fig_customer_type, use_container_width=True)
    
    with col2:
        # Revenue comparison
        fig_revenue_type = px.bar(
            x=['Repeat Customers', 'One-time Customers'],
            y=[repeat_revenue, one_time_revenue],
            title="Revenue Contribution by Customer Type",
            labels={'x': 'Customer Type', 'y': 'Total Revenue ($)'},
            color=['Repeat Customers', 'One-time Customers'],
            color_discrete_sequence=['#2E8B57', '#FF6B6B']
        )
        st.plotly_chart(fig_revenue_type, use_container_width=True)
    
    # 5. Additional Customer Insights
    st.subheader("💡 Additional Customer Insights")
    
    # Customer lifetime value analysis
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_customer_value = customer_metrics['total_spend'].mean()
        st.metric("Average Customer Value", format_number(avg_customer_value))
    
    with col2:
        median_customer_value = customer_metrics['total_spend'].median()
        st.metric("Median Customer Value", format_number(median_customer_value))
    
    with col3:
        avg_transactions_per_customer = customer_metrics['transaction_count'].mean()
        st.metric("Avg Transactions per Customer", f"{avg_transactions_per_customer:.1f}")
    
    # Customer retention analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Customer acquisition over time
        monthly_customers = df.groupby(df['invoice_date'].dt.to_period('M'))['customer_id'].nunique()
        fig_monthly_customers = px.line(
            x=monthly_customers.index.astype(str),
            y=monthly_customers.values,
            title="New Customers per Month",
            labels={'x': 'Month', 'y': 'Number of New Customers'}
        )
        st.plotly_chart(fig_monthly_customers, use_container_width=True)
    
    with col2:
        # Customer spend vs frequency scatter
        fig_scatter = px.scatter(
            customer_metrics,
            x='transaction_count',
            y='total_spend',
            title="Customer Spend vs Transaction Frequency",
            labels={'x': 'Number of Transactions', 'y': 'Total Spend ($)'},
            color='value_segment',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Customer behavior insights
    st.subheader("📈 Customer Behavior Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Average days between purchases
        repeat_customers_with_dates = customer_metrics[customer_metrics['transaction_count'] > 1]
        if len(repeat_customers_with_dates) > 0:
            avg_days_between = (repeat_customers_with_dates['last_purchase'] - repeat_customers_with_dates['first_purchase']).dt.days / (repeat_customers_with_dates['transaction_count'] - 1)
            avg_days = avg_days_between.mean()
            st.metric("Avg Days Between Purchases", f"{avg_days:.0f} days")
        else:
            st.metric("Avg Days Between Purchases", "N/A")
    
    with col2:
        # Customer retention rate
        retention_rate = (repeat_count / total_customers * 100)
        st.metric("Customer Retention Rate", f"{retention_rate:.1f}%")
    
    with col3:
        # Average customer lifetime
        customer_lifetime = (customer_metrics['last_purchase'] - customer_metrics['first_purchase']).dt.days
        avg_lifetime = customer_lifetime.mean()
        st.metric("Avg Customer Lifetime", f"{avg_lifetime:.0f} days")
    
    # Summary tables
    st.subheader("📋 Customer Analysis Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Value Segment Summary**")
        display_segment = segment_analysis.reset_index()
        display_segment = display_segment[['value_segment', 'customer_count', 'percentage', 'total_spend', 'spend_share', 'avg_spend']]
        display_segment.columns = ['Value Segment', 'Customer Count', 'Percentage (%)', 'Total Spend ($)', 'Spend Share (%)', 'Avg Spend ($)']
        
        # Format numbers
        display_segment['Total Spend ($)'] = display_segment['Total Spend ($)'].apply(lambda x: format_number(x))
        display_segment['Spend Share (%)'] = display_segment['Spend Share (%)'].apply(lambda x: format_percentage(x))
        display_segment['Avg Spend ($)'] = display_segment['Avg Spend ($)'].apply(lambda x: format_number(x))
        
        st.dataframe(display_segment, use_container_width=True)
    
    with col2:
        st.write("**RFM Segment Summary**")
        display_rfm = rfm_analysis.reset_index()
        display_rfm = display_rfm[['RFM_segment', 'customer_count', 'percentage', 'total_spend', 'spend_share', 'avg_spend']]
        display_rfm.columns = ['RFM Segment', 'Customer Count', 'Percentage (%)', 'Total Spend ($)', 'Spend Share (%)', 'Avg Spend ($)']
        
        # Format numbers
        display_rfm['Total Spend ($)'] = display_rfm['Total Spend ($)'].apply(lambda x: format_number(x))
        display_rfm['Spend Share (%)'] = display_rfm['Spend Share (%)'].apply(lambda x: format_percentage(x))
        display_rfm['Avg Spend ($)'] = display_rfm['Avg Spend ($)'].apply(lambda x: format_number(x))
        
        st.dataframe(display_rfm, use_container_width=True)

def main():
    st.markdown('<h1 class="main-header">🛍️ Retail Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    if df is None:
        st.error("Failed to load data. Please check if 'customer_shopping.csv' exists.")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏪 Store Performance", "👥 Customer Analysis", "📊 Business Intelligence", "🔬 Advanced Analytics"])
    
    with tab1:
        store_performance_analysis(df)
    
    with tab2:
        customer_analysis(df)
    
    with tab3:
        business_intelligence(df)
    
    with tab4:
        advanced_analytics(df)

def advanced_analytics(df):
    """Advanced Analytics with Discount Impact and Campaign Simulation"""
    st.header("🔬 Advanced Analytics")
    
    if df is None or df.empty:
        st.error("No data available for analysis")
        return
    
    # 1. Discount Impact on Profitability
    st.subheader("💰 Discount Impact on Profitability")
    
    # Calculate effective margin (price - discount) per product
    # For this analysis, we'll assume discount is a percentage of price
    # Since we don't have actual discount data, we'll simulate different discount scenarios
    
    # Get unique products and their base prices
    product_analysis = df.groupby(['category', 'shopping_mall']).agg({
        'price': ['mean', 'count', 'sum'],
        'quantity': 'sum'
    }).round(2)
    
    product_analysis.columns = ['avg_price', 'transaction_count', 'total_revenue', 'total_quantity']
    product_analysis = product_analysis.reset_index()
    
    # Simulate different discount scenarios
    discount_scenarios = [0, 5, 10, 15, 20, 25, 30]
    
    # Calculate effective margin for each discount scenario
    discount_impact_data = []
    for discount in discount_scenarios:
        product_analysis[f'discount_{discount}'] = product_analysis['avg_price'] * (discount / 100)
        product_analysis[f'effective_price_{discount}'] = product_analysis['avg_price'] - product_analysis[f'discount_{discount}']
        product_analysis[f'effective_margin_{discount}'] = product_analysis[f'effective_price_{discount}'] - (product_analysis['avg_price'] * 0.6)  # Assuming 60% cost
        product_analysis[f'projected_revenue_{discount}'] = product_analysis[f'effective_price_{discount}'] * product_analysis['total_quantity']
        
        # Calculate total impact
        total_original_revenue = product_analysis['total_revenue'].sum()
        total_projected_revenue = product_analysis[f'projected_revenue_{discount}'].sum()
        revenue_impact = ((total_projected_revenue - total_original_revenue) / total_original_revenue) * 100
        
        discount_impact_data.append({
            'discount_percent': discount,
            'total_revenue': total_projected_revenue,
            'revenue_impact': revenue_impact,
            'avg_effective_margin': product_analysis[f'effective_margin_{discount}'].mean()
        })
    
    discount_df = pd.DataFrame(discount_impact_data)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Avg Price", format_number(product_analysis['avg_price'].mean()))
    with col2:
        st.metric("Current Total Revenue", format_number(product_analysis['total_revenue'].sum()))
    with col3:
        st.metric("Current Avg Margin", format_number(product_analysis['avg_price'].mean() * 0.4))  # 40% margin
    with col4:
        st.metric("Total Products", f"{len(product_analysis):,}")
    
    # Discount impact visualization
    col1, col2 = st.columns(2)
    
    with col1:
        fig_discount_revenue = px.line(
            discount_df,
            x='discount_percent',
            y='total_revenue',
            title="Revenue Impact by Discount Level",
            labels={'x': 'Discount (%)', 'y': 'Total Revenue ($)'},
            markers=True
        )
        fig_discount_revenue.add_hline(y=product_analysis['total_revenue'].sum(), line_dash="dash", 
                                     annotation_text="Current Revenue")
        st.plotly_chart(fig_discount_revenue, use_container_width=True)
    
    with col2:
        fig_discount_margin = px.line(
            discount_df,
            x='discount_percent',
            y='avg_effective_margin',
            title="Average Effective Margin by Discount Level",
            labels={'x': 'Discount (%)', 'y': 'Avg Effective Margin ($)'},
            markers=True,
            color_discrete_sequence=['#FF6B6B']
        )
        st.plotly_chart(fig_discount_margin, use_container_width=True)
    
    # Revenue impact table
    st.subheader("📊 Discount Impact Analysis")
    display_discount = discount_df.copy()
    display_discount['total_revenue'] = display_discount['total_revenue'].apply(lambda x: format_number(x))
    display_discount['revenue_impact'] = display_discount['revenue_impact'].apply(lambda x: f"{x:+.1f}%")
    display_discount['avg_effective_margin'] = display_discount['avg_effective_margin'].apply(lambda x: format_number(x))
    display_discount.columns = ['Discount (%)', 'Total Revenue', 'Revenue Impact (%)', 'Avg Effective Margin']
    
    st.dataframe(display_discount, use_container_width=True)
    
    # 2. Campaign Simulation
    st.subheader("🎯 Campaign Simulation: High-Value Customer Targeting")
    
    # Calculate customer metrics for campaign simulation
    customer_metrics = df.groupby('customer_id').agg({
        'price': ['sum', 'count', 'mean'],
        'quantity': 'sum',
        'invoice_date': ['min', 'max']
    }).round(2)
    
    customer_metrics.columns = ['total_spend', 'transaction_count', 'avg_order_value', 'total_quantity', 'first_purchase', 'last_purchase']
    
    # Identify high-value customers (top 20% by spend)
    high_value_threshold = customer_metrics['total_spend'].quantile(0.8)
    high_value_customers = customer_metrics[customer_metrics['total_spend'] >= high_value_threshold]
    
    # Campaign simulation controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        discount_percent = st.slider("Campaign Discount (%)", 0, 50, 10, 1)
    with col2:
        campaign_reach = st.slider("Campaign Reach (% of High-Value Customers)", 10, 100, 50, 5)
    with col3:
        response_rate = st.slider("Expected Response Rate (%)", 10, 100, 30, 5)
    
    # Calculate campaign metrics
    target_customers = int(len(high_value_customers) * (campaign_reach / 100))
    responding_customers = int(target_customers * (response_rate / 100))
    
    # Calculate campaign costs and returns
    avg_order_value = high_value_customers['avg_order_value'].mean()
    discount_amount = avg_order_value * (discount_percent / 100)
    new_avg_order_value = avg_order_value - discount_amount
    
    # Campaign costs
    campaign_cost = responding_customers * discount_amount
    
    # Campaign returns
    additional_orders_per_customer = 1.5  # Assume 1.5 additional orders per responding customer
    campaign_revenue = responding_customers * new_avg_order_value * additional_orders_per_customer
    
    # ROI calculation
    campaign_profit = campaign_revenue - campaign_cost
    roi_percentage = (campaign_profit / campaign_cost * 100) if campaign_cost > 0 else 0
    
    # Display campaign metrics
    st.subheader("📈 Campaign Performance Projection")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Target Customers", f"{target_customers:,}")
    with col2:
        st.metric("Expected Responders", f"{responding_customers:,}")
    with col3:
        st.metric("Campaign Cost", format_number(campaign_cost))
    with col4:
        st.metric("Projected Revenue", format_number(campaign_revenue))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Campaign Profit", format_number(campaign_profit))
    with col2:
        st.metric("ROI", f"{roi_percentage:.1f}%")
    with col3:
        st.metric("Cost per Customer", format_number(campaign_cost / responding_customers) if responding_customers > 0 else "$0")
    with col4:
        st.metric("Revenue per Customer", format_number(campaign_revenue / responding_customers) if responding_customers > 0 else "$0")
    
    # Campaign simulation visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # ROI by discount level
        discount_levels = range(0, 51, 5)
        roi_data = []
        
        for disc in discount_levels:
            disc_amount = avg_order_value * (disc / 100)
            new_avg_ov = avg_order_value - disc_amount
            cost = responding_customers * disc_amount
            revenue = responding_customers * new_avg_ov * additional_orders_per_customer
            profit = revenue - cost
            roi = (profit / cost * 100) if cost > 0 else 0
            roi_data.append({'discount': disc, 'roi': roi, 'profit': profit})
        
        roi_df = pd.DataFrame(roi_data)
        
        fig_roi = px.line(
            roi_df,
            x='discount',
            y='roi',
            title="ROI by Discount Level",
            labels={'x': 'Discount (%)', 'y': 'ROI (%)'},
            markers=True
        )
        fig_roi.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
        fig_roi.add_vline(x=discount_percent, line_dash="dash", line_color="green", 
                         annotation_text=f"Selected: {discount_percent}%")
        st.plotly_chart(fig_roi, use_container_width=True)
    
    with col2:
        # Profit by discount level
        fig_profit = px.line(
            roi_df,
            x='discount',
            y='profit',
            title="Campaign Profit by Discount Level",
            labels={'x': 'Discount (%)', 'y': 'Campaign Profit ($)'},
            markers=True,
            color_discrete_sequence=['#2E8B57']
        )
        fig_profit.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
        fig_profit.add_vline(x=discount_percent, line_dash="dash", line_color="green", 
                            annotation_text=f"Selected: {discount_percent}%")
        st.plotly_chart(fig_profit, use_container_width=True)
    
    # Campaign insights
    st.subheader("💡 Campaign Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Optimal Discount Analysis**")
        optimal_discount = roi_df.loc[roi_df['roi'].idxmax(), 'discount']
        max_roi = roi_df['roi'].max()
        st.write(f"• **Optimal Discount**: {optimal_discount}% (ROI: {max_roi:.1f}%)")
        
        if roi_percentage > 0:
            st.write(f"• **Current Selection**: {discount_percent}% (ROI: {roi_percentage:.1f}%)")
            if discount_percent < optimal_discount:
                st.write(f"• **Recommendation**: Consider increasing discount to {optimal_discount}% for better ROI")
            elif discount_percent > optimal_discount:
                st.write(f"• **Recommendation**: Consider reducing discount to {optimal_discount}% for better ROI")
        else:
            st.write("• **Warning**: Current discount level results in negative ROI")
    
    with col2:
        st.write("**Campaign Effectiveness**")
        if roi_percentage > 50:
            st.write("• **Excellent**: High ROI campaign")
        elif roi_percentage > 20:
            st.write("• **Good**: Profitable campaign")
        elif roi_percentage > 0:
            st.write("• **Marginal**: Low but positive ROI")
        else:
            st.write("• **Poor**: Negative ROI - reconsider strategy")
        
        st.write(f"• **Break-even point**: {roi_df[roi_df['roi'] >= 0]['discount'].min():.0f}% discount")
        st.write(f"• **Max profit**: {format_number(roi_df['profit'].max())} at {roi_df.loc[roi_df['profit'].idxmax(), 'discount']:.0f}% discount")

if __name__ == "__main__":
    main()
