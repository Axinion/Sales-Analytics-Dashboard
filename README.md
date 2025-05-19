📊 Sales Analytics Dashboard

An interactive and comprehensive Sales Analytics Dashboard built using Streamlit, Plotly, Prophet, and various data science libraries. This tool helps businesses analyze sales performance, forecast future revenue, understand customer behavior, detect anomalies, and more — all in a modern, interactive web application.
🚀 Features
🏠 Overview

    Total Sales, Revenue, Profit, Quantity KPIs

    Month-over-Month (MoM) trends and changes

    Sales trends with 7-day moving averages

    Sales breakdown by Region

    Paginated raw data table exportable to CSV

📈 Forecasting

    30-day sales forecasting using Facebook Prophet

    Forecast plot with upper/lower confidence intervals

    Delta comparison between forecasted and last actual sales

👥 Customer Analytics

    RFM (Recency, Frequency, Monetary) Analysis

    Customer segmentation: Champions, Loyalists, At-Risk, etc.

    Interactive pie chart of customer segments

🔍 Advanced Analytics

    Seasonal Decomposition (Trend, Seasonality, Residual)

    Customer Segmentation using K-Means on RFM features

    Statistical Analysis:

        Correlation heatmaps

        Histograms and Boxplots

        Summary statistics

🌍 Advanced Visualizations

    Sales Heat Map by Region and Date

    Animated Time Series chart of Sales/Profit/Quantity

    Year-over-Year comparison of Sales and Profit

🧠 Predictive Analytics

    Churn Prediction based on days since last purchase

    Price Elasticity Analysis (Unit Price vs Quantity sold)

    Product Recommendations using correlation of products bought together

    Anomaly Detection of abnormal sales using rolling statistics

📂 File Structure

├── app.py                  # Main Streamlit app file
├── Amazon 2_Raw.xlsx       # Primary dataset file (Excel)
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies

📊 Dataset Requirements

Your dataset should include the following columns:
Column Name	Description
Order Date	Date of order placement
Ship Date	Date of shipping
Region	Sales region or geography
Category	Product category
Sales	Total sales amount
Profit	Profit earned
Quantity	Number of units sold
EmailID	Customer's email address (for RFM)
Product Name	Name of the product (for analysis)
Order ID	Unique identifier for each order
💻 Installation & Setup
1. Clone the Repository

git clone https://github.com/yourusername/sales-analytics-dashboard.git
cd sales-analytics-dashboard

2. Install Dependencies

pip install -r requirements.txt

3. Run the Streamlit App

streamlit run app.py

    The dashboard will launch in your browser at http://localhost:8501

🧪 Technologies Used

    Streamlit – Web framework for interactive dashboards

    Pandas / NumPy – Data wrangling and manipulation

    Plotly – Interactive charts and visualizations

    Prophet – Forecasting library developed by Facebook

    Scikit-learn – Machine learning for clustering and preprocessing

    Statsmodels – Seasonal decomposition of time series

    mlxtend – Frequent pattern mining and association rules

📌 Key Highlights

    Fast filtering by date, region, category

    Responsive UI and modern styling with custom CSS

    Modular layout with tabs, sections, and charts

    Multiple caching layers to optimize performance

    Export functionality for filtered datasets

📝 To-Do / Future Enhancements

    Add authentication for secure usage

    Integrate external databases (e.g., PostgreSQL, Snowflake)

    Add support for real-time data ingestion via APIs

    Build dynamic recommendation engine with NLP

🙌 Acknowledgements

    Facebook Prophet for time series forecasting

    Streamlit community for layout and widget inspiration

    Open source libraries enabling rapid dashboard development

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
