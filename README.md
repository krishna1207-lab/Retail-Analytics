# 📊 Retail Analytics Dashboard

A comprehensive retail analytics dashboard built with Streamlit and FastAPI, providing insights into customer behavior, sales performance, and business metrics.

## 🚀 Features

### 📈 **7 Main Analysis Tabs:**

1. **Store/Region Performance** - Store performance metrics and top customers
2. **Customer Segmentation & RFM** - Customer segmentation and RFM analysis
3. **Profitability Analysis** - Discount impact and profitability insights
4. **Seasonal Trend Analysis** - Seasonal patterns and customer type analysis
5. **Payment Method Insights** - Payment method distribution and analysis
6. **Category Insights** - Product category performance and profitability
7. **Campaign Simulation** - Marketing campaign ROI and effectiveness

### 🎯 **Key Capabilities:**

- **Interactive Visualizations** - Plotly charts and graphs
- **Real-time Data** - FastAPI backend integration
- **Customer Analytics** - Segmentation, RFM analysis, and behavior insights
- **Business Intelligence** - Revenue analysis, profitability metrics
- **Marketing Insights** - Campaign simulation and ROI analysis
- **Seasonal Analysis** - Trend analysis and pattern recognition

## 🛠️ Installation & Setup

### **Option 1: Quick Start (Recommended)**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/krishna1207-lab/retail-analytics-dashboard.git
   cd retail-analytics-dashboard
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard:**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Access the dashboard:**
   - Open your browser to `http://localhost:8501`

### **Option 2: With FastAPI Backend**

1. **Start the FastAPI server:**
   ```bash
   python fastapi_app.py
   ```

2. **Start the Streamlit dashboard:**
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Access the services:**
   - Streamlit Dashboard: `http://localhost:8501`
   - FastAPI Server: `http://localhost:8000`
   - API Documentation: `http://localhost:8000/docs`

## 📁 Repository Structure

```
retail-analytics-dashboard/
├── 📊 streamlit_app.py              # Main Streamlit dashboard
├── 🚀 fastapi_app.py                # FastAPI backend server
├── 🤖 ml_models.py                  # Machine learning models
├── 🔄 data_pipeline.py              # Data processing pipeline
├── 📈 customer_shopping.csv         # Main dataset
├── 📋 requirements.txt              # Python dependencies
├── 📖 README.md                     # This file
├── 🗂️ models/                       # Pre-trained ML models
│   ├── customer_segmentation_model.pkl
│   ├── demand_forecasting_model.pkl
│   ├── kmeans_model.pkl
│   ├── label_encoders.pkl
│   ├── profitability_model.pkl
│   └── scaler.pkl
└── 📊 processed_data/               # Processed datasets
    ├── customer_metrics_latest.csv
    ├── rfm_analysis_latest.csv
    ├── product_metrics_latest.csv
    ├── insights_latest.json
    └── transactions_latest.csv
```

## 📊 Data Requirements

The dashboard expects the following data files:

- `customer_shopping.csv` - Main customer transaction data
- `processed_data/customer_metrics_latest.csv` - Processed customer metrics
- `processed_data/rfm_analysis_latest.csv` - RFM analysis results

### **Data Format:**
```csv
customer_id,age,gender,category,quantity,price,payment_method,purchase_date
C100001,25,Male,Clothing,2,50.00,Credit Card,2023-01-15
```

## 🎨 Dashboard Sections

### **1. Store/Region Performance**
- Store revenue analysis
- Transaction volume by store
- Top customers by revenue
- Customer distribution analysis

### **2. Customer Segmentation & RFM**
- Value-based customer segmentation
- ML model customer segments
- RFM analysis (Recency, Frequency, Monetary)
- Customer segment performance metrics

### **3. Profitability Analysis**
- Discount impact on revenue
- Category profitability analysis
- High impact categories identification
- Revenue share analysis

### **4. Seasonal Trend Analysis**
- Monthly revenue trends
- Seasonal pattern analysis
- Customer type distribution
- Repeat vs one-time customer analysis

### **5. Payment Method Insights**
- Payment method distribution
- Revenue by payment method
- Payment preference analysis
- Transaction value insights

### **6. Category Insights**
- Top categories by revenue
- Category profitability metrics
- Customer count by category
- Performance summary

### **7. Campaign Simulation**
- Marketing campaign ROI analysis
- Customer targeting effectiveness
- Expected revenue projections
- Campaign cost analysis

## 🔧 Configuration

### **API Configuration**
The dashboard connects to a FastAPI backend. To configure:

1. **Update API URL** in `streamlit_app.py`:
   ```python
   API_BASE_URL = "http://localhost:8000"  # Change as needed
   ```

2. **Data File Paths** - Update file paths if needed:
   ```python
   customer_data = pd.read_csv('customer_shopping.csv')
   ```

## 📈 Usage

### **Navigation**
- Use the sidebar to switch between different analysis tabs
- Each tab provides specific insights and visualizations
- Interactive charts allow zooming and filtering

### **Data Interaction**
- Hover over charts for detailed information
- Use filters and controls where available
- Export data using Streamlit's built-in features

## 🚀 Deployment

### **Streamlit Cloud (Recommended)**

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Deploy the app

3. **Access your live dashboard:**
   - Your dashboard will be available at `https://your-app-name.streamlit.app`

### **Other Deployment Options**

- **Heroku** - Use the included `Procfile`
- **AWS** - Deploy using AWS App Runner or EC2
- **Google Cloud** - Use Cloud Run or App Engine
- **Docker** - Use the included `Dockerfile`

## 📋 Requirements

- Python 3.8+
- Streamlit 1.28.0+
- Pandas 1.5.0+
- Plotly 5.15.0+
- Requests 2.31.0+

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the GitHub repository
- Check the documentation
- Review the API documentation at `/docs`

## 🎯 Future Enhancements

- Real-time data updates
- Advanced filtering options
- Export functionality
- Mobile-responsive design
- Additional visualization types
- Machine learning predictions

---

**Built with ❤️ using Streamlit and FastAPI**