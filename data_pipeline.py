"""
Automated Data Ingestion and Processing Pipeline
MLOPS Capstone Project - Retail Analytics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
from pathlib import Path
import json
from typing import Dict, List, Optional
import schedule
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RetailDataPipeline:
    """Automated data processing pipeline for retail analytics"""
    
    def __init__(self, config_path: str = "pipeline_config.json"):
        self.config = self.load_config(config_path)
        self.processed_data_path = Path(self.config.get("processed_data_path", "processed_data/"))
        self.processed_data_path.mkdir(exist_ok=True)
        
    def load_config(self, config_path: str) -> Dict:
        """Load pipeline configuration"""
        default_config = {
            "raw_data_path": "customer_shopping.csv",
            "processed_data_path": "processed_data/",
            "backup_path": "backups/",
            "quality_thresholds": {
                "min_transaction_value": 0.01,
                "max_transaction_value": 10000,
                "min_age": 18,
                "max_age": 100,
                "min_quantity": 1,
                "max_quantity": 50
            },
            "feature_engineering": {
                "create_age_groups": True,
                "create_seasonal_features": True,
                "create_customer_metrics": True,
                "create_rfm_features": True
            }
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Could not load config from {config_path}: {e}")
                logger.info("Using default configuration")
        
        return default_config
    
    def validate_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean data based on quality thresholds"""
        logger.info("Starting data quality validation...")
        initial_rows = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(df)} duplicate rows")
        
        # Validate transaction values
        thresholds = self.config["quality_thresholds"]
        
        # Filter by transaction value
        df = df[
            (df['price'] >= thresholds["min_transaction_value"]) &
            (df['price'] <= thresholds["max_transaction_value"])
        ]
        
        # Filter by age
        df = df[
            (df['age'] >= thresholds["min_age"]) &
            (df['age'] <= thresholds["max_age"])
        ]
        
        # Filter by quantity
        df = df[
            (df['quantity'] >= thresholds["min_quantity"]) &
            (df['quantity'] <= thresholds["max_quantity"])
        ]
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            logger.warning(f"Missing values found: {missing_counts[missing_counts > 0].to_dict()}")
            # Handle missing values (you can customize this based on business rules)
            df = df.dropna()
        
        final_rows = len(df)
        logger.info(f"Data quality validation completed. Rows: {initial_rows} -> {final_rows}")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features"""
        logger.info("Starting feature engineering...")
        
        config = self.config["feature_engineering"]
        
        # Convert date
        df['invoice_date'] = pd.to_datetime(df['invoice_date'], format='%d/%m/%Y')
        
        # Basic features
        df['total_amount'] = df['quantity'] * df['price']
        df['year'] = df['invoice_date'].dt.year
        df['month'] = df['invoice_date'].dt.month
        df['day'] = df['invoice_date'].dt.day
        df['weekday'] = df['invoice_date'].dt.dayofweek
        df['quarter'] = df['invoice_date'].dt.quarter
        
        # Age groups
        if config.get("create_age_groups", True):
            df['age_group'] = pd.cut(df['age'], 
                                   bins=[0, 25, 35, 50, 65, 100], 
                                   labels=['18-25', '26-35', '36-50', '51-65', '65+'])
        
        # Seasonal features
        if config.get("create_seasonal_features", True):
            df['season'] = df['month'].map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Fall', 10: 'Fall', 11: 'Fall'
            })
            
            # Weekend indicator
            df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
            
            # Holiday proximity (simplified - you can enhance this)
            df['is_holiday_season'] = df['month'].isin([11, 12, 1]).astype(int)
        
        logger.info(f"Feature engineering completed. New shape: {df.shape}")
        return df
    
    def create_customer_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create customer-level aggregated metrics"""
        logger.info("Creating customer metrics...")
        
        customer_metrics = df.groupby('customer_id').agg({
            'total_amount': ['sum', 'mean', 'count', 'std'],
            'quantity': ['sum', 'mean'],
            'invoice_date': ['min', 'max'],
            'category': lambda x: x.nunique(),
            'shopping_mall': lambda x: x.nunique(),
            'age': 'first',
            'gender': 'first',
            'payment_method': lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]
        }).round(2)
        
        # Flatten column names
        customer_metrics.columns = [
            'total_revenue', 'avg_transaction_value', 'transaction_count', 'transaction_std',
            'total_items', 'avg_items_per_transaction',
            'first_purchase', 'last_purchase',
            'category_diversity', 'mall_diversity',
            'age', 'gender', 'preferred_payment_method'
        ]
        
        # Calculate additional metrics
        reference_date = df['invoice_date'].max()
        customer_metrics['recency'] = (reference_date - customer_metrics['last_purchase']).dt.days
        customer_metrics['customer_lifetime_days'] = (
            customer_metrics['last_purchase'] - customer_metrics['first_purchase']
        ).dt.days + 1
        customer_metrics['purchase_frequency'] = (
            customer_metrics['transaction_count'] / customer_metrics['customer_lifetime_days'] * 30
        ).round(2)  # Purchases per month
        
        # Customer value segments
        revenue_q75 = customer_metrics['total_revenue'].quantile(0.75)
        frequency_q75 = customer_metrics['transaction_count'].quantile(0.75)
        
        def customer_segment(row):
            if row['total_revenue'] >= revenue_q75 and row['transaction_count'] >= frequency_q75:
                return 'Champions'
            elif row['total_revenue'] >= revenue_q75:
                return 'High-Spenders'
            elif row['transaction_count'] >= frequency_q75:
                return 'Frequent-Buyers'
            elif row['recency'] <= 30:
                return 'New-Customers'
            elif row['recency'] <= 90:
                return 'Active-Customers'
            else:
                return 'At-Risk'
        
        customer_metrics['customer_segment'] = customer_metrics.apply(customer_segment, axis=1)
        
        logger.info(f"Customer metrics created for {len(customer_metrics)} customers")
        return customer_metrics
    
    def create_rfm_analysis(self, customer_metrics: pd.DataFrame) -> pd.DataFrame:
        """Create RFM (Recency, Frequency, Monetary) analysis"""
        logger.info("Creating RFM analysis...")
        
        # RFM scores (1-5 scale)
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
        
        logger.info("RFM analysis completed")
        return rfm_data
    
    def create_product_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create product/category-level metrics"""
        logger.info("Creating product metrics...")
        
        product_metrics = df.groupby(['category', 'shopping_mall']).agg({
            'total_amount': ['sum', 'mean', 'count'],
            'quantity': ['sum', 'mean'],
            'customer_id': 'nunique',
            'price': ['mean', 'std']
        }).round(2)
        
        product_metrics.columns = [
            'total_revenue', 'avg_transaction_value', 'transaction_count',
            'total_quantity', 'avg_quantity_per_transaction',
            'unique_customers', 'avg_price', 'price_std'
        ]
        
        # Calculate additional metrics
        product_metrics['revenue_per_customer'] = (
            product_metrics['total_revenue'] / product_metrics['unique_customers']
        ).round(2)
        
        logger.info(f"Product metrics created for {len(product_metrics)} category-mall combinations")
        return product_metrics
    
    def generate_insights_summary(self, df: pd.DataFrame, customer_metrics: pd.DataFrame, 
                                rfm_data: pd.DataFrame) -> Dict:
        """Generate automated insights summary"""
        logger.info("Generating insights summary...")
        
        insights = {
            'data_summary': {
                'total_transactions': len(df),
                'unique_customers': df['customer_id'].nunique(),
                'total_revenue': float(df['total_amount'].sum()),
                'avg_transaction_value': float(df['total_amount'].mean()),
                'date_range': {
                    'start': df['invoice_date'].min().strftime('%Y-%m-%d'),
                    'end': df['invoice_date'].max().strftime('%Y-%m-%d')
                }
            },
            'top_categories': df.groupby('category')['total_amount'].sum().sort_values(ascending=False).head(5).to_dict(),
            'top_stores': df.groupby('shopping_mall')['total_amount'].sum().sort_values(ascending=False).head(5).to_dict(),
            'customer_segments': customer_metrics['customer_segment'].value_counts().to_dict(),
            'rfm_segments': rfm_data['RFM_Segment'].value_counts().to_dict(),
            'seasonal_trends': df.groupby('season')['total_amount'].sum().to_dict(),
            'payment_preferences': df['payment_method'].value_counts(normalize=True).to_dict(),
            'generated_at': datetime.now().isoformat()
        }
        
        return insights
    
    def save_processed_data(self, df: pd.DataFrame, customer_metrics: pd.DataFrame, 
                          rfm_data: pd.DataFrame, product_metrics: pd.DataFrame, 
                          insights: Dict) -> None:
        """Save all processed data and insights"""
        logger.info("Saving processed data...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main datasets
        df.to_csv(self.processed_data_path / f"transactions_processed_{timestamp}.csv", index=False)
        customer_metrics.to_csv(self.processed_data_path / f"customer_metrics_{timestamp}.csv")
        rfm_data.to_csv(self.processed_data_path / f"rfm_analysis_{timestamp}.csv")
        product_metrics.to_csv(self.processed_data_path / f"product_metrics_{timestamp}.csv")
        
        # Save latest versions (for API consumption)
        df.to_csv(self.processed_data_path / "transactions_latest.csv", index=False)
        customer_metrics.to_csv(self.processed_data_path / "customer_metrics_latest.csv")
        rfm_data.to_csv(self.processed_data_path / "rfm_analysis_latest.csv")
        product_metrics.to_csv(self.processed_data_path / "product_metrics_latest.csv")
        
        # Save insights
        with open(self.processed_data_path / f"insights_{timestamp}.json", 'w') as f:
            json.dump(insights, f, indent=2, default=str)
        
        with open(self.processed_data_path / "insights_latest.json", 'w') as f:
            json.dump(insights, f, indent=2, default=str)
        
        logger.info(f"Data saved to {self.processed_data_path}")
    
    def run_pipeline(self, data_path: Optional[str] = None) -> Dict:
        """Run the complete data processing pipeline"""
        logger.info("=== STARTING DATA PIPELINE ===")
        
        try:
            # Load raw data
            data_path = data_path or self.config["raw_data_path"]
            logger.info(f"Loading data from {data_path}")
            df = pd.read_csv(data_path)
            logger.info(f"Raw data loaded: {df.shape}")
            
            # Data quality validation
            df = self.validate_data_quality(df)
            
            # Feature engineering
            df = self.engineer_features(df)
            
            # Create customer metrics
            customer_metrics = self.create_customer_metrics(df)
            
            # RFM analysis
            rfm_data = self.create_rfm_analysis(customer_metrics)
            
            # Product metrics
            product_metrics = self.create_product_metrics(df)
            
            # Generate insights
            insights = self.generate_insights_summary(df, customer_metrics, rfm_data)
            
            # Save processed data
            self.save_processed_data(df, customer_metrics, rfm_data, product_metrics, insights)
            
            logger.info("=== DATA PIPELINE COMPLETED SUCCESSFULLY ===")
            
            return {
                'status': 'success',
                'processed_transactions': len(df),
                'unique_customers': len(customer_metrics),
                'insights': insights,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def schedule_pipeline(self, frequency: str = "daily"):
        """Schedule the pipeline to run automatically"""
        logger.info(f"Scheduling pipeline to run {frequency}")
        
        if frequency == "daily":
            schedule.every().day.at("02:00").do(self.run_pipeline)
        elif frequency == "weekly":
            schedule.every().monday.at("02:00").do(self.run_pipeline)
        elif frequency == "hourly":
            schedule.every().hour.do(self.run_pipeline)
        
        logger.info("Pipeline scheduled. Starting scheduler...")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute


def create_sample_config():
    """Create a sample configuration file"""
    config = {
        "raw_data_path": "customer_shopping.csv",
        "processed_data_path": "processed_data/",
        "backup_path": "backups/",
        "quality_thresholds": {
            "min_transaction_value": 0.01,
            "max_transaction_value": 10000,
            "min_age": 18,
            "max_age": 100,
            "min_quantity": 1,
            "max_quantity": 50
        },
        "feature_engineering": {
            "create_age_groups": True,
            "create_seasonal_features": True,
            "create_customer_metrics": True,
            "create_rfm_features": True
        },
        "scheduling": {
            "enabled": False,
            "frequency": "daily"  # options: daily, weekly, hourly
        }
    }
    
    with open("pipeline_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Sample configuration created: pipeline_config.json")


def main():
    """Main function to run the data pipeline"""
    print("=== RETAIL ANALYTICS DATA PIPELINE ===")
    
    # Create sample config if it doesn't exist
    if not os.path.exists("pipeline_config.json"):
        create_sample_config()
    
    # Initialize and run pipeline
    pipeline = RetailDataPipeline()
    result = pipeline.run_pipeline()
    
    print(f"\nPipeline Result: {result['status']}")
    if result['status'] == 'success':
        print(f"Processed {result['processed_transactions']} transactions")
        print(f"Analyzed {result['unique_customers']} customers")
    
    return pipeline


if __name__ == "__main__":
    pipeline = main()
