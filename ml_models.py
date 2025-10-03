"""
ML Models for Retail Analytics - Customer Segmentation, Profitability, and Forecasting
MLOPS Capstone Project
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class RetailAnalyticsML:
    def __init__(self):
        self.scaler = StandardScaler()
        self.customer_segmentation_model = None
        self.profitability_model = None
        self.demand_forecasting_model = None
        self.kmeans_model = None
        self.label_encoders = {}
        
    def load_and_preprocess_data(self, csv_path):
        """Load and preprocess the retail dataset"""
        df = pd.read_csv(csv_path)
        
        # Convert date
        df['invoice_date'] = pd.to_datetime(df['invoice_date'], format='%d/%m/%Y')
        
        # Feature engineering
        df['total_amount'] = df['quantity'] * df['price']
        df['year'] = df['invoice_date'].dt.year
        df['month'] = df['invoice_date'].dt.month
        df['weekday'] = df['invoice_date'].dt.dayofweek
        df['quarter'] = df['invoice_date'].dt.quarter
        
        # Age groups
        df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 65, 100], 
                                labels=['18-25', '26-35', '36-50', '51-65', '65+'])
        
        return df
    
    def create_customer_features(self, df):
        """Create customer-level features for segmentation (adapted for single-transaction dataset)"""
        # Since each customer has only 1 transaction, we'll use transaction-level features
        customer_features = df.copy()
        
        # Rename columns to match expected format
        customer_features = customer_features.rename(columns={
            'total_amount': 'total_revenue',
            'quantity': 'total_items'
        })
        
        # Add derived features
        customer_features['transaction_count'] = 1  # All customers have 1 transaction
        customer_features['avg_transaction_value'] = customer_features['total_revenue']
        customer_features['first_purchase'] = customer_features['invoice_date']
        customer_features['last_purchase'] = customer_features['invoice_date']
        customer_features['category_diversity'] = 1  # Each transaction has 1 category
        customer_features['mall_diversity'] = 1  # Each transaction has 1 mall
        
        # Calculate recency (days since transaction)
        reference_date = df['invoice_date'].max()
        customer_features['recency'] = (reference_date - customer_features['invoice_date']).dt.days
        customer_features['customer_lifetime_days'] = 1  # Single transaction = 1 day
        
        # Set index to customer_id for consistency
        customer_features = customer_features.set_index('customer_id')
        
        return customer_features
    
    def train_customer_segmentation_model(self, df):
        """Train customer segmentation model using K-Means and Random Forest"""
        print("Training Customer Segmentation Model...")
        
        # Create customer features
        customer_features = self.create_customer_features(df)
        
        # Prepare features for clustering
        clustering_features = ['total_revenue', 'transaction_count', 'avg_transaction_value', 
                             'recency', 'category_diversity']
        X_cluster = customer_features[clustering_features].fillna(0)
        X_cluster_scaled = self.scaler.fit_transform(X_cluster)
        
        # K-Means clustering
        self.kmeans_model = KMeans(n_clusters=4, random_state=42)
        customer_features['cluster'] = self.kmeans_model.fit_predict(X_cluster_scaled)
        
        # Create segment labels based on transaction value and recency (adapted for single transactions)
        revenue_q75 = customer_features['total_revenue'].quantile(0.75)
        revenue_q50 = customer_features['total_revenue'].quantile(0.50)
        revenue_q25 = customer_features['total_revenue'].quantile(0.25)
        recency_q25 = customer_features['recency'].quantile(0.25)
        recency_q75 = customer_features['recency'].quantile(0.75)
        
        def assign_segment(row):
            if row['total_revenue'] >= revenue_q75 and row['recency'] <= recency_q25:
                return 'High-Value-Recent'
            elif row['total_revenue'] >= revenue_q75:
                return 'High-Value'
            elif row['total_revenue'] >= revenue_q50 and row['recency'] <= recency_q25:
                return 'Medium-Value-Recent'
            elif row['total_revenue'] >= revenue_q50:
                return 'Medium-Value'
            elif row['recency'] <= recency_q25:
                return 'Low-Value-Recent'
            else:
                return 'Low-Value'
        
        customer_features['segment'] = customer_features.apply(assign_segment, axis=1)
        
        # Train Random Forest classifier
        X_features = customer_features[clustering_features + ['age']].fillna(0)
        
        # Encode gender
        if 'gender' not in self.label_encoders:
            self.label_encoders['gender'] = LabelEncoder()
        customer_features['gender_encoded'] = self.label_encoders['gender'].fit_transform(
            customer_features['gender']
        )
        
        X_features['gender_encoded'] = customer_features['gender_encoded']
        y = customer_features['segment']
        
        # Debug: Print segment distribution
        print(f"Segment distribution:\n{y.value_counts()}")
        
        # Encode target variable
        if 'segment' not in self.label_encoders:
            self.label_encoders['segment'] = LabelEncoder()
        y_encoded = self.label_encoders['segment'].fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Use more conservative parameters to prevent overfitting
        self.customer_segmentation_model = RandomForestClassifier(
            n_estimators=50, 
            max_depth=10, 
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        self.customer_segmentation_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.customer_segmentation_model.predict(X_test)
        print("Customer Segmentation Model Performance:")
        print(classification_report(y_test, y_pred, 
                                  target_names=self.label_encoders['segment'].classes_))
        
        return customer_features
    
    def train_profitability_model(self, df):
        """Train profitability prediction model"""
        print("\nTraining Profitability Model...")
        
        # Create features
        features = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['category', 'payment_method', 'shopping_mall', 'gender']
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            features[f'{col}_encoded'] = self.label_encoders[col].fit_transform(features[col])
        
        # Select features for profitability prediction
        feature_cols = ['age', 'quantity', 'price', 'month', 'weekday', 'quarter'] + \
                      [f'{col}_encoded' for col in categorical_cols]
        
        X = features[feature_cols]
        y = features['total_amount']  # Target: transaction profitability
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Use more conservative parameters to prevent overfitting
        self.profitability_model = RandomForestRegressor(
            n_estimators=50, 
            max_depth=10, 
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        self.profitability_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.profitability_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Profitability Model Performance:")
        print(f"MSE: {mse:.2f}")
        print(f"R2 Score: {r2:.4f}")
        print(f"RMSE: {np.sqrt(mse):.2f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.profitability_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 5 Features for Profitability:")
        print(feature_importance.head())
        
    def train_demand_forecasting_model(self, df):
        """Train demand forecasting model"""
        print("\nTraining Demand Forecasting Model...")
        
        # Aggregate daily sales
        daily_sales = df.groupby(['invoice_date', 'category']).agg({
            'quantity': 'sum',
            'total_amount': 'sum'
        }).reset_index()
        
        # Create time-based features
        daily_sales['year'] = daily_sales['invoice_date'].dt.year
        daily_sales['month'] = daily_sales['invoice_date'].dt.month
        daily_sales['day'] = daily_sales['invoice_date'].dt.day
        daily_sales['weekday'] = daily_sales['invoice_date'].dt.dayofweek
        daily_sales['quarter'] = daily_sales['invoice_date'].dt.quarter
        
        # Encode category
        if 'category_forecast' not in self.label_encoders:
            self.label_encoders['category_forecast'] = LabelEncoder()
        daily_sales['category_encoded'] = self.label_encoders['category_forecast'].fit_transform(
            daily_sales['category']
        )
        
        # Create lag features (previous day sales)
        daily_sales = daily_sales.sort_values(['category', 'invoice_date'])
        daily_sales['prev_day_quantity'] = daily_sales.groupby('category')['quantity'].shift(1)
        daily_sales['prev_day_amount'] = daily_sales.groupby('category')['total_amount'].shift(1)
        
        # Remove rows with NaN (first day for each category)
        daily_sales = daily_sales.dropna()
        
        # Features for forecasting
        forecast_features = ['year', 'month', 'day', 'weekday', 'quarter', 'category_encoded',
                           'prev_day_quantity', 'prev_day_amount']
        
        X = daily_sales[forecast_features]
        y = daily_sales['quantity']  # Predict next day quantity
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Use more conservative parameters to prevent overfitting
        self.demand_forecasting_model = RandomForestRegressor(
            n_estimators=50, 
            max_depth=10, 
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        self.demand_forecasting_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.demand_forecasting_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Demand Forecasting Model Performance:")
        print(f"MSE: {mse:.2f}")
        print(f"R2 Score: {r2:.4f}")
        print(f"RMSE: {np.sqrt(mse):.2f}")
        
    def predict_customer_segment(self, customer_data):
        """Predict customer segment for new data"""
        if self.customer_segmentation_model is None:
            raise ValueError("Customer segmentation model not trained yet!")
        
        prediction = self.customer_segmentation_model.predict(customer_data)
        segment = self.label_encoders['segment'].inverse_transform(prediction)
        return segment[0]
    
    def predict_profitability(self, transaction_data):
        """Predict transaction profitability"""
        if self.profitability_model is None:
            raise ValueError("Profitability model not trained yet!")
        
        prediction = self.profitability_model.predict(transaction_data)
        return prediction[0]
    
    def predict_demand(self, forecast_data):
        """Predict demand for next period"""
        if self.demand_forecasting_model is None:
            raise ValueError("Demand forecasting model not trained yet!")
        
        prediction = self.demand_forecasting_model.predict(forecast_data)
        return prediction[0]
    
    def save_models(self, model_path="models/"):
        """Save all trained models"""
        import os
        os.makedirs(model_path, exist_ok=True)
        
        if self.customer_segmentation_model:
            joblib.dump(self.customer_segmentation_model, f"{model_path}customer_segmentation_model.pkl")
        if self.profitability_model:
            joblib.dump(self.profitability_model, f"{model_path}profitability_model.pkl")
        if self.demand_forecasting_model:
            joblib.dump(self.demand_forecasting_model, f"{model_path}demand_forecasting_model.pkl")
        if self.kmeans_model:
            joblib.dump(self.kmeans_model, f"{model_path}kmeans_model.pkl")
        
        joblib.dump(self.scaler, f"{model_path}scaler.pkl")
        joblib.dump(self.label_encoders, f"{model_path}label_encoders.pkl")
        
        print(f"Models saved to {model_path}")
    
    def load_models(self, model_path="models/"):
        """Load pre-trained models"""
        try:
            self.customer_segmentation_model = joblib.load(f"{model_path}customer_segmentation_model.pkl")
            self.profitability_model = joblib.load(f"{model_path}profitability_model.pkl")
            self.demand_forecasting_model = joblib.load(f"{model_path}demand_forecasting_model.pkl")
            self.kmeans_model = joblib.load(f"{model_path}kmeans_model.pkl")
            self.scaler = joblib.load(f"{model_path}scaler.pkl")
            self.label_encoders = joblib.load(f"{model_path}label_encoders.pkl")
            print(f"Models loaded from {model_path}")
        except FileNotFoundError as e:
            print(f"Model file not found: {e}")


def main():
    """Main function to train all models"""
    print("=== RETAIL ANALYTICS ML PIPELINE ===")
    
    # Initialize ML pipeline
    ml_pipeline = RetailAnalyticsML()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = ml_pipeline.load_and_preprocess_data('customer_shopping.csv')
    print(f"Dataset loaded: {df.shape}")
    
    # Train models
    customer_features = ml_pipeline.train_customer_segmentation_model(df)
    ml_pipeline.train_profitability_model(df)
    ml_pipeline.train_demand_forecasting_model(df)
    
    # Save models
    ml_pipeline.save_models()
    
    print("\n=== MODEL TRAINING COMPLETED ===")
    print("All models have been trained and saved successfully!")
    
    # Example predictions
    print("\n=== EXAMPLE PREDICTIONS ===")
    
    # Example customer data for segmentation
    sample_customer = np.array([[5000, 10, 500, 30, 3, 25, 1]]).reshape(1, -1)  # Example features
    try:
        segment = ml_pipeline.predict_customer_segment(sample_customer)
        print(f"Sample customer segment: {segment}")
    except Exception as e:
        print(f"Customer segmentation prediction error: {e}")
    
    return ml_pipeline


if __name__ == "__main__":
    ml_pipeline = main()
