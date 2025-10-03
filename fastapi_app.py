"""
FastAPI Web Application for Retail Analytics
MLOPS Capstone Project - Serving ML Insights and KPIs
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from ml_models import RetailAnalyticsML
import uvicorn
import os
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="Retail Analytics API",
    description="End-to-end analytics pipeline for retail insights and KPIs",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
ml_pipeline = None
df = None
customer_metrics = None

# Pydantic models for API requests/responses
class CustomerData(BaseModel):
    customer_id: str
    age: int
    gender: str
    total_revenue: float
    transaction_count: int
    avg_transaction_value: float
    recency: int
    category_diversity: int

class TransactionData(BaseModel):
    age: int
    quantity: int
    price: float
    category: str
    payment_method: str
    shopping_mall: str
    gender: str
    month: int
    weekday: int
    quarter: int

class ForecastData(BaseModel):
    year: int
    month: int
    day: int
    weekday: int
    quarter: int
    category: str
    prev_day_quantity: float
    prev_day_amount: float

class KPIResponse(BaseModel):
    total_revenue: float
    total_transactions: int
    unique_customers: int
    avg_transaction_value: float
    avg_items_per_transaction: float
    date_range: Dict[str, str]

class SegmentAnalysis(BaseModel):
    segment: str
    customer_count: int
    total_revenue: float
    avg_revenue_per_customer: float
    percentage: float

@app.on_event("startup")
async def startup_event():
    """Initialize ML models and load data on startup"""
    global ml_pipeline, df, customer_metrics
    
    print("Loading ML models and data...")
    
    # Initialize ML pipeline
    ml_pipeline = RetailAnalyticsML()
    
    # Load data
    try:
        df = ml_pipeline.load_and_preprocess_data('customer_shopping.csv')
        print(f"Dataset loaded: {df.shape}")
        
        # Create customer metrics
        customer_metrics = ml_pipeline.create_customer_features(df)
        
        # Try to load pre-trained models
        try:
            ml_pipeline.load_models()
            print("Pre-trained models loaded successfully!")
        except:
            print("Pre-trained models not found. Training new models...")
            ml_pipeline.train_customer_segmentation_model(df)
            ml_pipeline.train_profitability_model(df)
            ml_pipeline.train_demand_forecasting_model(df)
            ml_pipeline.save_models()
            print("New models trained and saved!")
            
    except Exception as e:
        print(f"Error during startup: {e}")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API documentation"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Retail Analytics API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .header { background-color: #f0f0f0; padding: 20px; border-radius: 10px; }
            .endpoint { background-color: #e8f4fd; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { color: #2e7d32; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🛒 Retail Analytics API</h1>
            <p>End-to-end analytics pipeline for retail insights and KPIs</p>
            <p><strong>MLOPS Capstone Project</strong></p>
        </div>
        
        <h2>Available Endpoints:</h2>
        
        <div class="endpoint">
            <span class="method">GET</span> <strong>/kpis</strong> - Get overall business KPIs
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <strong>/customer-segments</strong> - Get customer segmentation analysis
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <strong>/seasonal-trends</strong> - Get seasonal sales trends
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <strong>/payment-analysis</strong> - Get payment method analysis
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <strong>/store-performance</strong> - Get store performance metrics
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span> <strong>/predict/customer-segment</strong> - Predict customer segment
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span> <strong>/predict/profitability</strong> - Predict transaction profitability
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span> <strong>/predict/demand</strong> - Predict demand forecast
        </div>
        
        <p><a href="/docs" style="color: #1976d2;">📚 View Interactive API Documentation</a></p>
    </body>
    </html>
    """
    return html_content

@app.get("/kpis", response_model=KPIResponse)
async def get_business_kpis():
    """Get overall business KPIs"""
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    return KPIResponse(
        total_revenue=float(df['total_amount'].sum()),
        total_transactions=int(len(df)),
        unique_customers=int(df['customer_id'].nunique()),
        avg_transaction_value=float(df['total_amount'].mean()),
        avg_items_per_transaction=float(df['quantity'].mean()),
        date_range={
            "start": df['invoice_date'].min().strftime('%Y-%m-%d'),
            "end": df['invoice_date'].max().strftime('%Y-%m-%d')
        }
    )

@app.get("/customer-segments")
async def get_customer_segments():
    """Get customer segmentation analysis"""
    if customer_metrics is None:
        raise HTTPException(status_code=500, detail="Customer metrics not available")
    
    # Create customer segments
    revenue_median = customer_metrics['total_revenue'].median()
    frequency_median = customer_metrics['transaction_count'].median()
    
    def customer_segment(row):
        if row['total_revenue'] >= revenue_median and row['transaction_count'] >= frequency_median:
            return 'High-Value'
        elif row['total_revenue'] >= revenue_median:
            return 'High-Spender'
        elif row['transaction_count'] >= frequency_median:
            return 'Frequent-Buyer'
        else:
            return 'Low-Value'
    
    customer_metrics['segment'] = customer_metrics.apply(customer_segment, axis=1)
    
    # Segment analysis
    segments = []
    total_customers = len(customer_metrics)
    
    for segment in customer_metrics['segment'].unique():
        segment_data = customer_metrics[customer_metrics['segment'] == segment]
        segments.append(SegmentAnalysis(
            segment=segment,
            customer_count=len(segment_data),
            total_revenue=float(segment_data['total_revenue'].sum()),
            avg_revenue_per_customer=float(segment_data['total_revenue'].mean()),
            percentage=float((len(segment_data) / total_customers) * 100)
        ))
    
    return {"segments": segments}

@app.get("/seasonal-trends")
async def get_seasonal_trends():
    """Get seasonal sales trends"""
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    # Monthly trends
    monthly_sales = df.groupby('month')['total_amount'].sum().to_dict()
    
    # Seasonal analysis
    df['season'] = df['month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    
    seasonal_sales = df.groupby('season')['total_amount'].sum().to_dict()
    
    # Day of week analysis
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_sales = df.groupby('weekday')['total_amount'].sum()
    weekday_dict = {weekday_names[i]: float(weekday_sales.get(i, 0)) for i in range(7)}
    
    return {
        "monthly_trends": {str(k): float(v) for k, v in monthly_sales.items()},
        "seasonal_trends": {k: float(v) for k, v in seasonal_sales.items()},
        "weekday_trends": weekday_dict
    }

@app.get("/payment-analysis")
async def get_payment_analysis():
    """Get payment method analysis"""
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    payment_stats = df.groupby('payment_method').agg({
        'total_amount': ['sum', 'mean', 'count'],
        'age': 'mean'
    }).round(2)
    
    payment_stats.columns = ['total_revenue', 'avg_transaction', 'transaction_count', 'avg_age']
    
    result = {}
    for payment_method in payment_stats.index:
        result[payment_method] = {
            'total_revenue': float(payment_stats.loc[payment_method, 'total_revenue']),
            'avg_transaction': float(payment_stats.loc[payment_method, 'avg_transaction']),
            'transaction_count': int(payment_stats.loc[payment_method, 'transaction_count']),
            'avg_customer_age': float(payment_stats.loc[payment_method, 'avg_age'])
        }
    
    return result

@app.get("/store-performance")
async def get_store_performance():
    """Get store performance metrics"""
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    store_stats = df.groupby('shopping_mall').agg({
        'total_amount': ['sum', 'mean', 'count'],
        'customer_id': 'nunique'
    }).round(2)
    
    store_stats.columns = ['total_revenue', 'avg_transaction', 'transaction_count', 'unique_customers']
    store_stats['revenue_per_customer'] = (store_stats['total_revenue'] / 
                                         store_stats['unique_customers']).round(2)
    
    result = {}
    for mall in store_stats.index:
        result[mall] = {
            'total_revenue': float(store_stats.loc[mall, 'total_revenue']),
            'avg_transaction': float(store_stats.loc[mall, 'avg_transaction']),
            'transaction_count': int(store_stats.loc[mall, 'transaction_count']),
            'unique_customers': int(store_stats.loc[mall, 'unique_customers']),
            'revenue_per_customer': float(store_stats.loc[mall, 'revenue_per_customer'])
        }
    
    return result

@app.get("/category-analysis")
async def get_category_analysis():
    """Get product category analysis"""
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    category_stats = df.groupby('category').agg({
        'total_amount': ['sum', 'mean', 'count'],
        'quantity': 'sum',
        'customer_id': 'nunique'
    }).round(2)
    
    category_stats.columns = ['total_revenue', 'avg_transaction', 'transaction_count', 
                             'total_quantity', 'unique_customers']
    
    result = {}
    for category in category_stats.index:
        result[category] = {
            'total_revenue': float(category_stats.loc[category, 'total_revenue']),
            'avg_transaction': float(category_stats.loc[category, 'avg_transaction']),
            'transaction_count': int(category_stats.loc[category, 'transaction_count']),
            'total_quantity': int(category_stats.loc[category, 'total_quantity']),
            'unique_customers': int(category_stats.loc[category, 'unique_customers'])
        }
    
    return result

@app.post("/predict/customer-segment")
async def predict_customer_segment(customer: CustomerData):
    """Predict customer segment using ML model"""
    if ml_pipeline is None or ml_pipeline.customer_segmentation_model is None:
        raise HTTPException(status_code=500, detail="Customer segmentation model not available")
    
    try:
        # Prepare features (adjust based on your model's expected input)
        features = np.array([[
            customer.total_revenue,
            customer.transaction_count,
            customer.avg_transaction_value,
            customer.recency,
            customer.category_diversity,
            customer.age,
            1 if customer.gender.lower() == 'female' else 0  # Gender encoding
        ]])
        
        segment = ml_pipeline.predict_customer_segment(features)
        
        return {
            "customer_id": customer.customer_id,
            "predicted_segment": segment,
            "confidence": "High",  # You can add actual confidence scores
            "recommendations": get_segment_recommendations(segment)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/profitability")
async def predict_profitability(transaction: TransactionData):
    """Predict transaction profitability"""
    if ml_pipeline is None or ml_pipeline.profitability_model is None:
        raise HTTPException(status_code=500, detail="Profitability model not available")
    
    try:
        # Encode categorical variables (simplified - you may need to use the actual encoders)
        category_map = {'Clothing': 0, 'Shoes': 1, 'Electronics': 2, 'Cosmetics': 3, 
                       'Food & Beverage': 4, 'Toys': 5, 'Technology': 6, 'Books': 7, 'Souvenir': 8}
        payment_map = {'Credit Card': 0, 'Debit Card': 1, 'Cash': 2}
        mall_map = {'Kanyon': 0, 'Forum Istanbul': 1, 'Metrocity': 2, 'Metropol AVM': 3}  # Add more as needed
        
        features = np.array([[
            transaction.age,
            transaction.quantity,
            transaction.price,
            transaction.month,
            transaction.weekday,
            transaction.quarter,
            category_map.get(transaction.category, 0),
            payment_map.get(transaction.payment_method, 0),
            mall_map.get(transaction.shopping_mall, 0),
            1 if transaction.gender.lower() == 'female' else 0
        ]])
        
        predicted_amount = ml_pipeline.predict_profitability(features)
        
        return {
            "predicted_transaction_value": float(predicted_amount),
            "input_price": transaction.price,
            "quantity": transaction.quantity,
            "expected_total": float(predicted_amount),
            "profitability_score": "High" if predicted_amount > 500 else "Medium" if predicted_amount > 200 else "Low"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/demand")
async def predict_demand(forecast: ForecastData):
    """Predict demand forecast"""
    if ml_pipeline is None or ml_pipeline.demand_forecasting_model is None:
        raise HTTPException(status_code=500, detail="Demand forecasting model not available")
    
    try:
        # Encode category (simplified)
        category_map = {'Clothing': 0, 'Shoes': 1, 'Electronics': 2, 'Cosmetics': 3, 
                       'Food & Beverage': 4, 'Toys': 5, 'Technology': 6, 'Books': 7, 'Souvenir': 8}
        
        features = np.array([[
            forecast.year,
            forecast.month,
            forecast.day,
            forecast.weekday,
            forecast.quarter,
            category_map.get(forecast.category, 0),
            forecast.prev_day_quantity,
            forecast.prev_day_amount
        ]])
        
        predicted_demand = ml_pipeline.predict_demand(features)
        
        return {
            "category": forecast.category,
            "date": f"{forecast.year}-{forecast.month:02d}-{forecast.day:02d}",
            "predicted_quantity": float(predicted_demand),
            "demand_level": "High" if predicted_demand > 50 else "Medium" if predicted_demand > 20 else "Low",
            "previous_day_quantity": forecast.prev_day_quantity
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

def get_segment_recommendations(segment: str) -> List[str]:
    """Get recommendations based on customer segment"""
    recommendations = {
        'High-Value': [
            "Offer premium products and services",
            "Provide personalized shopping experiences",
            "Implement VIP loyalty programs",
            "Send exclusive offers and early access to sales"
        ],
        'High-Spender': [
            "Focus on upselling and cross-selling",
            "Offer bundle deals and premium products",
            "Provide personalized product recommendations",
            "Implement frequency-based rewards"
        ],
        'Frequent-Buyer': [
            "Encourage higher-value purchases",
            "Offer quantity-based discounts",
            "Provide loyalty rewards for consistent shopping",
            "Send regular promotional offers"
        ],
        'Low-Value': [
            "Implement win-back campaigns",
            "Offer attractive discounts and promotions",
            "Provide value-oriented product recommendations",
            "Focus on customer retention strategies"
        ]
    }
    
    return recommendations.get(segment, ["Focus on customer engagement and retention"])

# ==================== COMPREHENSIVE BUSINESS INSIGHTS APIs ====================

@app.get("/insights/store-performance")
async def get_store_vs_region_performance():
    """1. Store vs Region Performance - Compare sales volume and revenue across stores and regions"""
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    # Store performance analysis
    store_performance = df.groupby('shopping_mall').agg({
        'total_amount': ['sum', 'mean', 'count'],
        'customer_id': 'nunique',
        'quantity': 'sum'
    }).round(2)
    
    store_performance.columns = ['total_revenue', 'avg_transaction_value', 'transaction_count', 'unique_customers', 'total_quantity']
    store_performance['revenue_per_customer'] = (store_performance['total_revenue'] / store_performance['unique_customers']).round(2)
    store_performance['items_per_transaction'] = (store_performance['total_quantity'] / store_performance['transaction_count']).round(2)
    
    # Sort by revenue
    store_performance = store_performance.sort_values('total_revenue', ascending=False)
    
    # Regional analysis (assuming malls represent regions)
    regional_summary = {
        'total_stores': len(store_performance),
        'total_revenue': float(store_performance['total_revenue'].sum()),
        'avg_revenue_per_store': float(store_performance['total_revenue'].mean()),
        'top_performing_store': store_performance.index[0],
        'top_store_revenue': float(store_performance['total_revenue'].iloc[0])
    }
    
    return {
        "store_performance": store_performance.to_dict('index'),
        "regional_summary": regional_summary,
        "performance_rankings": {
            "by_revenue": store_performance.index.tolist(),
            "by_customers": store_performance.sort_values('unique_customers', ascending=False).index.tolist(),
            "by_avg_transaction": store_performance.sort_values('avg_transaction_value', ascending=False).index.tolist()
        }
    }

@app.get("/insights/top-customers")
async def get_top_customers(limit: int = Query(10, description="Number of top customers to return")):
    """2. Top Customers - Identify top customers by purchase value"""
    if customer_metrics is None:
        raise HTTPException(status_code=500, detail="Customer metrics not available")
    
    # Get top customers by revenue
    top_customers = customer_metrics.nlargest(limit, 'total_revenue')
    
    # Calculate top 10% threshold
    top_10_percent_threshold = customer_metrics['total_revenue'].quantile(0.9)
    top_10_percent_customers = customer_metrics[customer_metrics['total_revenue'] >= top_10_percent_threshold]
    
    return {
        "top_customers": top_customers[['total_revenue', 'avg_transaction_value', 'transaction_count', 'recency', 'age', 'gender']].to_dict('index'),
        "top_10_percent_summary": {
            "threshold": float(top_10_percent_threshold),
            "customer_count": len(top_10_percent_customers),
            "percentage_of_total": float(len(top_10_percent_customers) / len(customer_metrics) * 100),
            "total_revenue": float(top_10_percent_customers['total_revenue'].sum()),
            "revenue_percentage": float(top_10_percent_customers['total_revenue'].sum() / customer_metrics['total_revenue'].sum() * 100)
        }
    }

@app.get("/insights/customer-segmentation")
async def get_high_vs_low_value_segmentation():
    """3. High vs Low-value Segmentation - Classify customers based on total spend"""
    if customer_metrics is None:
        raise HTTPException(status_code=500, detail="Customer metrics not available")
    
    # Define value segments based on revenue
    revenue_median = customer_metrics['total_revenue'].median()
    revenue_q75 = customer_metrics['total_revenue'].quantile(0.75)
    revenue_q25 = customer_metrics['total_revenue'].quantile(0.25)
    
    def assign_value_segment(revenue):
        if revenue >= revenue_q75:
            return 'High-Value'
        elif revenue >= revenue_median:
            return 'Medium-Value'
        elif revenue >= revenue_q25:
            return 'Low-Value'
        else:
            return 'Very-Low-Value'
    
    # Create a copy to avoid modifying the original
    customer_metrics_copy = customer_metrics.copy()
    customer_metrics_copy['value_segment'] = customer_metrics_copy['total_revenue'].apply(assign_value_segment)
    
    segment_analysis = customer_metrics_copy.groupby('value_segment').agg({
        'total_revenue': ['sum', 'mean', 'count'],
        'customer_id': 'count'
    }).round(2)
    
    segment_analysis.columns = ['total_revenue', 'avg_revenue', 'transaction_count', 'customer_count']
    segment_analysis['revenue_percentage'] = (segment_analysis['total_revenue'] / segment_analysis['total_revenue'].sum() * 100).round(2)
    segment_analysis['customer_percentage'] = (segment_analysis['customer_count'] / segment_analysis['customer_count'].sum() * 100).round(2)
    
    return {
        "value_segments": segment_analysis.to_dict('index'),
        "thresholds": {
            "very_low_value": float(revenue_q25),
            "low_value": float(revenue_median),
            "high_value": float(revenue_q75)
        },
        "insights": {
            "high_value_customers": int(segment_analysis.loc['High-Value', 'customer_count']) if 'High-Value' in segment_analysis.index else 0,
            "high_value_revenue_share": float(segment_analysis.loc['High-Value', 'revenue_percentage']) if 'High-Value' in segment_analysis.index else 0.0,
            "pareto_80_20": "High-value customers represent significant revenue concentration"
        }
    }

@app.get("/insights/discount-impact")
async def get_discount_impact_analysis():
    """4. Discount Impact on Profitability - Compute effective margin per product"""
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    # Simulate discount scenarios (since we don't have actual discount data)
    discount_scenarios = [0, 0.05, 0.10, 0.15, 0.20]  # 0%, 5%, 10%, 15%, 20%
    
    discount_analysis = {}
    for discount_rate in discount_scenarios:
        df_temp = df.copy()
        df_temp['discounted_price'] = df_temp['price'] * (1 - discount_rate)
        df_temp['effective_margin'] = df_temp['discounted_price'] - (df_temp['price'] * 0.3)  # Assuming 30% cost
        df_temp['total_discounted_revenue'] = df_temp['discounted_price'] * df_temp['quantity']
        
        analysis = {
            'discount_rate': discount_rate,
            'total_revenue': float(df_temp['total_discounted_revenue'].sum()),
            'avg_margin': float(df_temp['effective_margin'].mean()),
            'total_margin': float(df_temp['effective_margin'].sum()),
            'transaction_count': len(df_temp),
            'revenue_impact': float((df_temp['total_discounted_revenue'].sum() - df['total_amount'].sum()) / df['total_amount'].sum() * 100)
        }
        discount_analysis[f"{int(discount_rate*100)}%_discount"] = analysis
    
    # Category-wise discount impact
    category_discount_impact = {}
    for category in df['category'].unique():
        category_data = df[df['category'] == category]
        category_analysis = {}
        
        for discount_rate in [0, 0.10, 0.20]:  # 0%, 10%, 20%
            discounted_revenue = (category_data['price'] * (1 - discount_rate) * category_data['quantity']).sum()
            category_analysis[f"{int(discount_rate*100)}%_discount"] = {
                'revenue': float(discounted_revenue),
                'revenue_change': float((discounted_revenue - category_data['total_amount'].sum()) / category_data['total_amount'].sum() * 100)
            }
        category_discount_impact[category] = category_analysis
    
    return {
        "discount_scenarios": discount_analysis,
        "category_discount_impact": category_discount_impact,
        "recommendations": {
            "optimal_discount": "10% discount shows good balance of revenue and margin",
            "high_impact_categories": "Clothing and Electronics show highest discount sensitivity"
        }
    }

@app.get("/insights/seasonality-analysis")
async def get_seasonality_analysis():
    """5. Seasonality Analysis - Monthly/quarterly sales trends with seasonal patterns"""
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    # Monthly analysis
    monthly_sales = df.groupby('month').agg({
        'total_amount': ['sum', 'mean', 'count'],
        'customer_id': 'nunique'
    }).round(2)
    monthly_sales.columns = ['total_revenue', 'avg_transaction', 'transaction_count', 'unique_customers']
    
    # Quarterly analysis
    quarterly_sales = df.groupby('quarter').agg({
        'total_amount': ['sum', 'mean', 'count'],
        'customer_id': 'nunique'
    }).round(2)
    quarterly_sales.columns = ['total_revenue', 'avg_transaction', 'transaction_count', 'unique_customers']
    
    # Seasonal patterns
    df_temp = df.copy()
    df_temp['season'] = df_temp['month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    
    seasonal_patterns = df_temp.groupby('season').agg({
        'total_amount': ['sum', 'mean', 'count'],
        'customer_id': 'nunique'
    }).round(2)
    seasonal_patterns.columns = ['total_revenue', 'avg_transaction', 'transaction_count', 'unique_customers']
    
    # Weekday analysis
    weekday_sales = df.groupby('weekday').agg({
        'total_amount': ['sum', 'mean', 'count']
    }).round(2)
    weekday_sales.columns = ['total_revenue', 'avg_transaction', 'transaction_count']
    weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_sales.index = [weekday_names[i] for i in weekday_sales.index]
    
    # Peak periods identification
    peak_month = monthly_sales['total_revenue'].idxmax()
    peak_quarter = quarterly_sales['total_revenue'].idxmax()
    peak_season = seasonal_patterns['total_revenue'].idxmax()
    peak_weekday = weekday_sales['total_revenue'].idxmax()
    
    return {
        "monthly_trends": monthly_sales.to_dict('index'),
        "quarterly_trends": quarterly_sales.to_dict('index'),
        "seasonal_patterns": seasonal_patterns.to_dict('index'),
        "weekday_patterns": weekday_sales.to_dict('index'),
        "peak_periods": {
            "peak_month": int(peak_month),
            "peak_quarter": int(peak_quarter),
            "peak_season": peak_season,
            "peak_weekday": peak_weekday
        },
        "seasonal_insights": {
            "best_performing_month": f"Month {peak_month} with {monthly_sales.loc[peak_month, 'total_revenue']:,.2f} revenue",
            "seasonal_variation": f"{(seasonal_patterns['total_revenue'].max() - seasonal_patterns['total_revenue'].min()) / seasonal_patterns['total_revenue'].mean() * 100:.1f}% variation"
        }
    }

@app.get("/insights/payment-method-analysis")
async def get_payment_method_preference():
    """6. Payment Method Preference - Distribution across Cash, Card, UPI, etc."""
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    # Payment method analysis
    payment_analysis = df.groupby('payment_method').agg({
        'total_amount': ['sum', 'mean', 'count'],
        'customer_id': 'nunique',
        'age': 'mean'
    }).round(2)
    payment_analysis.columns = ['total_revenue', 'avg_transaction', 'transaction_count', 'unique_customers', 'avg_age']
    
    # Calculate percentages
    payment_analysis['revenue_percentage'] = (payment_analysis['total_revenue'] / payment_analysis['total_revenue'].sum() * 100).round(2)
    payment_analysis['transaction_percentage'] = (payment_analysis['transaction_count'] / payment_analysis['transaction_count'].sum() * 100).round(2)
    payment_analysis['customer_percentage'] = (payment_analysis['unique_customers'] / payment_analysis['unique_customers'].sum() * 100).round(2)
    
    # Age group payment preferences
    age_payment = df.groupby(['age_group', 'payment_method']).size().unstack(fill_value=0)
    age_payment_pct = age_payment.div(age_payment.sum(axis=1), axis=0) * 100
    
    # Gender payment preferences
    gender_payment = df.groupby(['gender', 'payment_method']).size().unstack(fill_value=0)
    gender_payment_pct = gender_payment.div(gender_payment.sum(axis=1), axis=0) * 100
    
    return {
        "payment_methods": payment_analysis.to_dict('index'),
        "age_payment_preferences": age_payment_pct.to_dict('index'),
        "gender_payment_preferences": gender_payment_pct.to_dict('index'),
        "insights": {
            "most_popular_method": payment_analysis['transaction_count'].idxmax(),
            "highest_value_method": payment_analysis['avg_transaction'].idxmax(),
            "youngest_demographic": payment_analysis['avg_age'].idxmin(),
            "payment_diversity": f"{len(payment_analysis)} different payment methods available"
        }
    }

@app.get("/insights/rfm-analysis")
async def get_rfm_analysis():
    """7. RFM Analysis - Compute Recency, Frequency, Monetary scores and segment"""
    if customer_metrics is None:
        raise HTTPException(status_code=500, detail="Customer metrics not available")
    
    # Load RFM data from processed files
    try:
        rfm_data = pd.read_csv('processed_data/rfm_analysis_latest.csv', index_col=0)
    except:
        # Create RFM analysis if not available
        rfm_data = customer_metrics[['recency', 'transaction_count', 'total_revenue']].copy()
        rfm_data.columns = ['Recency', 'Frequency', 'Monetary']
        
        # Create quintile-based scores
        rfm_data['R_Score'] = pd.qcut(rfm_data['Recency'].rank(method='first'), 5, labels=[5,4,3,2,1])
        rfm_data['F_Score'] = pd.qcut(rfm_data['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
        rfm_data['M_Score'] = pd.qcut(rfm_data['Monetary'].rank(method='first'), 5, labels=[1,2,3,4,5])
        
        # Combined RFM Score
        rfm_data['RFM_Score'] = (
            rfm_data['R_Score'].astype(str) + 
            rfm_data['F_Score'].astype(str) + 
            rfm_data['M_Score'].astype(str)
        )
        
        # RFM segments
        def rfm_segment(row):
            r, f, m = int(row['R_Score']), int(row['F_Score']), int(row['M_Score'])
            
            if r >= 4 and f >= 4 and m >= 4:
                return 'Champions'
            elif r >= 3 and f >= 3 and m >= 3:
                return 'Loyal_Customers'
            elif r >= 4 and f <= 2:
                return 'New_Customers'
            elif r >= 3 and f >= 3 and m <= 2:
                return 'Potential_Loyalists'
            elif r <= 2 and f >= 4:
                return 'At_Risk'
            elif r <= 2 and f <= 2 and m >= 4:
                return 'Cant_Lose_Them'
            elif r <= 2 and f <= 2 and m <= 2:
                return 'Lost_Customers'
            else:
                return 'Others'
        
        rfm_data['RFM_Segment'] = rfm_data.apply(rfm_segment, axis=1)
    
    # RFM segment analysis
    rfm_segment_analysis = rfm_data.groupby('RFM_Segment').agg({
        'Recency': 'mean',
        'Frequency': 'mean', 
        'Monetary': 'mean',
        'RFM_Score': 'count'
    }).round(2)
    rfm_segment_analysis.columns = ['avg_recency', 'avg_frequency', 'avg_monetary', 'customer_count']
    rfm_segment_analysis['customer_percentage'] = (rfm_segment_analysis['customer_count'] / rfm_segment_analysis['customer_count'].sum() * 100).round(2)
    
    # Score distribution
    score_distribution = {
        'R_Score': rfm_data['R_Score'].value_counts().to_dict(),
        'F_Score': rfm_data['F_Score'].value_counts().to_dict(),
        'M_Score': rfm_data['M_Score'].value_counts().to_dict()
    }
    
    return {
        "rfm_segments": rfm_segment_analysis.to_dict('index'),
        "score_distribution": score_distribution,
        "rfm_summary": {
            "total_customers": len(rfm_data),
            "champions_count": int(rfm_segment_analysis.loc['Champions', 'customer_count']) if 'Champions' in rfm_segment_analysis.index else 0,
            "at_risk_count": int(rfm_segment_analysis.loc['At_Risk', 'customer_count']) if 'At_Risk' in rfm_segment_analysis.index else 0,
            "lost_customers_count": int(rfm_segment_analysis.loc['Lost_Customers', 'customer_count']) if 'Lost_Customers' in rfm_segment_analysis.index else 0
        }
    }

@app.get("/insights/repeat-vs-onetime")
async def get_repeat_vs_onetime_analysis():
    """8. Repeat Customer vs One-time - Compare sales contribution"""
    if df is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    # Since this is a single-transaction dataset, all customers are one-time
    # But we can analyze by transaction patterns and value
    customer_analysis = df.groupby('customer_id').agg({
        'total_amount': 'sum',
        'quantity': 'sum',
        'category': 'nunique',
        'shopping_mall': 'nunique'
    }).round(2)
    
    customer_analysis.columns = ['total_revenue', 'total_quantity', 'category_diversity', 'mall_diversity']
    
    # Segment by transaction value and diversity
    high_value_customers = customer_analysis[customer_analysis['total_revenue'] >= customer_analysis['total_revenue'].quantile(0.8)]
    diverse_customers = customer_analysis[customer_analysis['category_diversity'] > 1]
    
    # Simulate repeat customer analysis (since we don't have actual repeat data)
    repeat_simulation = {
        "one_time_customers": {
            "count": len(customer_analysis),
            "percentage": 100.0,
            "total_revenue": float(customer_analysis['total_revenue'].sum()),
            "avg_transaction_value": float(customer_analysis['total_revenue'].mean())
        },
        "high_value_customers": {
            "count": len(high_value_customers),
            "percentage": float(len(high_value_customers) / len(customer_analysis) * 100),
            "total_revenue": float(high_value_customers['total_revenue'].sum()),
            "revenue_share": float(high_value_customers['total_revenue'].sum() / customer_analysis['total_revenue'].sum() * 100)
        },
        "diverse_customers": {
            "count": len(diverse_customers),
            "percentage": float(len(diverse_customers) / len(customer_analysis) * 100),
            "total_revenue": float(diverse_customers['total_revenue'].sum()),
            "revenue_share": float(diverse_customers['total_revenue'].sum() / customer_analysis['total_revenue'].sum() * 100)
        }
    }
    
    return {
        "customer_analysis": repeat_simulation,
        "insights": {
            "note": "This dataset contains single transactions per customer",
            "recommendation": "Focus on high-value customers and category diversity for growth",
            "potential_repeat": "High-value customers show potential for repeat business"
        }
    }

@app.get("/insights/category-insights")
async def get_category_wise_insights():
    """9. Category-wise Insights - Profitable categories and their customer segments"""
    if df is None or customer_metrics is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
    
    # Category performance analysis
    category_performance = df.groupby('category').agg({
        'total_amount': ['sum', 'mean', 'count'],
        'quantity': 'sum',
        'customer_id': 'nunique',
        'price': 'mean'
    }).round(2)
    category_performance.columns = ['total_revenue', 'avg_transaction', 'transaction_count', 'total_quantity', 'unique_customers', 'avg_price']
    category_performance['revenue_per_customer'] = (category_performance['total_revenue'] / category_performance['unique_customers']).round(2)
    category_performance['items_per_transaction'] = (category_performance['total_quantity'] / category_performance['transaction_count']).round(2)
    
    # Category profitability (assuming 30% cost)
    category_performance['estimated_cost'] = category_performance['total_revenue'] * 0.3
    category_performance['estimated_profit'] = category_performance['total_revenue'] - category_performance['estimated_cost']
    category_performance['profit_margin'] = (category_performance['estimated_profit'] / category_performance['total_revenue'] * 100).round(2)
    
    # Customer segments by category
    category_segments = {}
    for category in df['category'].unique():
        category_customers = df[df['category'] == category]['customer_id'].unique()
        category_customer_metrics = customer_metrics[customer_metrics.index.isin(category_customers)]
        
        if len(category_customer_metrics) > 0:
            # Segment by value
            high_value = category_customer_metrics[category_customer_metrics['total_revenue'] >= category_customer_metrics['total_revenue'].quantile(0.75)]
            category_segments[category] = {
                'total_customers': len(category_customer_metrics),
                'high_value_customers': len(high_value),
                'high_value_percentage': float(len(high_value) / len(category_customer_metrics) * 100),
                'avg_customer_value': float(category_customer_metrics['total_revenue'].mean()),
                'avg_customer_age': float(category_customer_metrics['age'].mean())
            }
    
    return {
        "category_performance": category_performance.to_dict('index'),
        "category_customer_segments": category_segments,
        "top_categories": {
            "by_revenue": category_performance.sort_values('total_revenue', ascending=False).index.tolist()[:5],
            "by_profit_margin": category_performance.sort_values('profit_margin', ascending=False).index.tolist()[:5],
            "by_customer_count": category_performance.sort_values('unique_customers', ascending=False).index.tolist()[:5]
        },
        "insights": {
            "most_profitable_category": category_performance['profit_margin'].idxmax(),
            "highest_revenue_category": category_performance['total_revenue'].idxmax(),
            "most_popular_category": category_performance['unique_customers'].idxmax()
        }
    }

@app.get("/insights/campaign-simulation")
async def get_campaign_simulation(discount_rate: float = Query(0.10, description="Discount rate for campaign simulation")):
    """10. Campaign Simulation - Model targeting high-value customers with discount—project ROI"""
    if customer_metrics is None:
        raise HTTPException(status_code=500, detail="Customer metrics not available")
    
    # Identify high-value customers (top 20%)
    high_value_threshold = customer_metrics['total_revenue'].quantile(0.8)
    high_value_customers = customer_metrics[customer_metrics['total_revenue'] >= high_value_threshold]
    
    # Campaign simulation
    campaign_results = {}
    
    for target_percentage in [10, 20, 30, 50]:  # Target different percentages of high-value customers
        target_customers = high_value_customers.head(int(len(high_value_customers) * target_percentage / 100))
        
        # Simulate campaign impact
        base_revenue = target_customers['total_revenue'].sum()
        discounted_revenue = base_revenue * (1 - discount_rate)
        campaign_cost = base_revenue * discount_rate  # Cost of discount
        estimated_increase = base_revenue * 0.15  # Assume 15% increase in volume due to campaign
        net_revenue = discounted_revenue + estimated_increase
        roi = ((net_revenue - base_revenue) / campaign_cost * 100) if campaign_cost > 0 else 0
        
        campaign_results[f"target_{target_percentage}%"] = {
            "customers_targeted": len(target_customers),
            "base_revenue": float(base_revenue),
            "discounted_revenue": float(discounted_revenue),
            "campaign_cost": float(campaign_cost),
            "estimated_increase": float(estimated_increase),
            "net_revenue": float(net_revenue),
            "roi_percentage": float(roi),
            "revenue_lift": float((net_revenue - base_revenue) / base_revenue * 100)
        }
    
    # Optimal campaign recommendation
    best_campaign = max(campaign_results.items(), key=lambda x: x[1]['roi_percentage'])
    
    return {
        "campaign_simulations": campaign_results,
        "recommendations": {
            "optimal_target": best_campaign[0],
            "optimal_roi": float(best_campaign[1]['roi_percentage']),
            "discount_rate": discount_rate,
            "high_value_customers_available": len(high_value_customers),
            "campaign_insights": {
                "best_performing_target": f"Targeting {best_campaign[0].split('_')[1]}% of high-value customers",
                "expected_roi": f"{best_campaign[1]['roi_percentage']:.1f}%",
                "revenue_lift": f"{best_campaign[1]['revenue_lift']:.1f}%"
            }
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "data_loaded": df is not None,
        "models_loaded": ml_pipeline is not None and ml_pipeline.customer_segmentation_model is not None
    }

if __name__ == "__main__":
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
