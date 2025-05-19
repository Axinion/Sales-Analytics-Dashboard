import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from prophet import Prophet
#from streamlit_folium import folium_static
#import folium
from streamlit_option_menu import option_menu
from streamlit_extras.metric_cards import style_metric_cards
import base64
from io import BytesIO
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings('ignore')

# Cache configuration
CACHE_TTL = 3600  # 1 hour cache time

# Required columns for data validation
REQUIRED_COLUMNS = ['Order Date', 'Ship Date', 'Region', 'Category', 'Sales', 'Profit', 'Quantity']

# Page config
st.set_page_config(
    page_title="Sales Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    .stSelectbox {
        margin-bottom: 1rem;
    }
    @media (max-width: 768px) {
        .main {
            padding: 1rem;
        }
        .stMetric {
            margin-bottom: 1rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Optimized data loading with caching
@st.cache_data(ttl=CACHE_TTL)
def load_data():
    """
    Load and cache the data with optimized data types.
    Returns a tuple of (dataframe, error_message)
    """
    try:
        # Try loading Excel file first
        df = pd.read_excel('Amazon 2_Raw.xlsx', engine='openpyxl')
        
        # Map column names to match our required structure
        column_mapping = {
            'Order Date': 'Order Date',
            'Ship Date': 'Ship Date',
            'Geography': 'Region',  # Map Geography to Region
            'Category': 'Category',
            'Sales': 'Sales',
            'Profit': 'Profit',
            'Quantity': 'Quantity'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Validate required columns
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_columns:
            return None, f"Missing required columns: {', '.join(missing_columns)}"
        
        # Optimize data types during loading
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        df['Ship Date'] = pd.to_datetime(df['Ship Date'])
        
        # Convert numeric columns to appropriate types
        numeric_cols = ['Sales', 'Profit', 'Quantity']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Convert categorical columns
        categorical_cols = ['Region', 'Category']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        return df, None
    except Exception as e:
        try:
            # Fallback to CSV if Excel fails
            df = pd.read_csv('sales_data.csv')
            # Apply the same column mapping for CSV
            df = df.rename(columns=column_mapping)
            return df, None
        except Exception as e:
            return None, f"Error loading data files: {str(e)}"

# Cache expensive computations
@st.cache_data(ttl=CACHE_TTL)
def calculate_metrics(df):
    """Calculate and cache common metrics"""
    return {
        'total_sales': df['Sales'].sum(),
        'total_profit': df['Profit'].sum(),
        'total_quantity': df['Quantity'].sum(),
        'avg_order_value': df['Sales'].mean(),
        'profit_margin': (df['Profit'].sum() / df['Sales'].sum()) * 100
    }

@st.cache_data(ttl=CACHE_TTL)
def get_filtered_data(df, date_range, region, category):
    """Cache filtered data based on user selections"""
    mask = (
        (df['Order Date'].dt.date >= date_range[0]) &
        (df['Order Date'].dt.date <= date_range[1])
    )
    
    if region != 'All Regions':
        mask = mask & (df['Region'] == region)
    
    if category != 'All Categories':
        mask = mask & (df['Category'] == category)
    
    return df[mask]

# Pagination helper function
def paginate_dataframe(df, page_size=10, page_num=1):
    """Helper function to paginate dataframe"""
    total_pages = len(df) // page_size + (1 if len(df) % page_size > 0 else 0)
    start_idx = (page_num - 1) * page_size
    end_idx = start_idx + page_size
    return df.iloc[start_idx:end_idx], total_pages

# Load and process data
df, error = load_data()
if error:
    st.error(error)
    st.stop()
if df is None or df.empty:
    st.error("No data loaded. Please check your data files.")
    st.stop()

# Navigation with icons
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Overview", "Forecasting", "Customer Analytics", "Advanced Analytics", "Advanced Visualizations", "Predictive Analytics"],
        icons=["house", "graph-up", "people", "chart-line", "map", "brain"],
        menu_icon="cast",
        default_index=0,
    )

# Sidebar filters with help text
st.sidebar.title("Filters")

# Date range filter with help
st.sidebar.markdown("### Date Range")
st.sidebar.markdown("Select the time period for analysis")
min_date = df['Order Date'].min()
max_date = df['Order Date'].max()
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    help="Choose the date range for your analysis"
)

# Region filter with Select All option and help
st.sidebar.markdown("### Region")
st.sidebar.markdown("Select a region to analyze")
all_regions = ['All Regions'] + sorted(df['Region'].unique().tolist())
selected_region = st.sidebar.selectbox(
    "Select Region",
    options=all_regions,
    index=0,
    help="Choose a specific region or view all regions"
)

# Category filter with Select All option and help
st.sidebar.markdown("### Category")
st.sidebar.markdown("Select a category to analyze")
all_categories = ['All Categories'] + sorted(df['Category'].unique().tolist())
selected_category = st.sidebar.selectbox(
    "Select Category",
    options=all_categories,
    index=0,
    help="Choose a specific category or view all categories"
)

# Apply filters
mask = (
    (df['Order Date'].dt.date >= date_range[0]) &
    (df['Order Date'].dt.date <= date_range[1])
)

if selected_region != 'All Regions':
    mask = mask & (df['Region'] == selected_region)

if selected_category != 'All Categories':
    mask = mask & (df['Category'] == selected_category)

filtered_df = df[mask]

# Export functionality
def get_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'data:file/csv;base64,{b64}'
    return f'<a href="{href}" download="{filename}">{text}</a>'

# Overview Page
if selected == "Overview":
    st.title("📊 Sales Dashboard")
    
    # Export button
    st.markdown(get_download_link(filtered_df, 'sales_data.csv', '📥 Download Filtered Data'), unsafe_allow_html=True)
    
    st.markdown("---")

    # KPI Metrics with cached calculations
    with st.spinner('Calculating metrics...'):
        metrics = calculate_metrics(filtered_df)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Sales", f"{metrics['total_quantity']:,.0f} units", 
                     help="Total number of units sold in the selected period")

        with col2:
            st.metric("Total Revenue", f"${metrics['total_sales']:,.2f}",
                     help="Total revenue generated in the selected period")

        with col3:
            st.metric("Total Profit", f"${metrics['total_profit']:,.2f}",
                     help="Total profit earned in the selected period")

        style_metric_cards()

    st.markdown("---")

    # Month-over-Month Comparison with cached calculations
    st.subheader("📈 Month-over-Month Comparison")
    with st.spinner('Generating comparison...'):
        # Calculate MoM metrics using cached filtered data
        filtered_df['Month'] = filtered_df['Order Date'].dt.to_period('M')
        mom_metrics = filtered_df.groupby('Month').agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Quantity': 'sum'
        }).reset_index()
        
        mom_metrics['Month'] = mom_metrics['Month'].astype(str)
        mom_metrics['Sales_Change'] = mom_metrics['Sales'].pct_change() * 100
        mom_metrics['Profit_Change'] = mom_metrics['Profit'].pct_change() * 100
        mom_metrics['Quantity_Change'] = mom_metrics['Quantity'].pct_change() * 100

        # Display MoM changes
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sales MoM Change", 
                     f"{mom_metrics['Sales_Change'].iloc[-1]:,.1f}%",
                     help="Month-over-Month change in sales")
        with col2:
            st.metric("Profit MoM Change", 
                     f"{mom_metrics['Profit_Change'].iloc[-1]:,.1f}%",
                     help="Month-over-Month change in profit")
        with col3:
            st.metric("Quantity MoM Change", 
                     f"{mom_metrics['Quantity_Change'].iloc[-1]:,.1f}%",
                     help="Month-over-Month change in quantity sold")

    # Charts with enhanced tooltips and caching
    col1, col2 = st.columns(2)

    with col1:
        # Sales over Time with Moving Average (cached)
        with st.spinner('Generating sales trend...'):
            daily_sales = filtered_df.groupby('Order Date')['Sales'].sum().reset_index()
            if not daily_sales.empty:
                daily_sales['MA7'] = daily_sales['Sales'].rolling(window=7).mean()
                
                fig_sales = go.Figure()
                fig_sales.add_trace(go.Scatter(
                    x=daily_sales['Order Date'],
                    y=daily_sales['Sales'],
                    name='Daily Revenue',
                    line=dict(color='rgb(26, 118, 255)'),
                    hovertemplate="Date: %{x}<br>Revenue: $%{y:,.2f}<extra></extra>"
                ))
                fig_sales.add_trace(go.Scatter(
                    x=daily_sales['Order Date'],
                    y=daily_sales['MA7'],
                    name='7-Day Moving Average',
                    line=dict(color='rgb(255, 65, 54)'),
                    hovertemplate="Date: %{x}<br>7-Day MA: $%{y:,.2f}<extra></extra>"
                ))
                
                fig_sales.update_layout(
                    title='Sales Over Time with Moving Average',
                    xaxis_title='Date',
                    yaxis_title='Revenue ($)',
                    hovermode='x unified',
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                st.plotly_chart(fig_sales, use_container_width=True)
            else:
                st.info("No sales data available for the selected period.")

    with col2:
        # Sales by Region with enhanced tooltips (cached)
        with st.spinner('Generating regional analysis...'):
            region_sales = filtered_df.groupby('Region')['Sales'].sum().reset_index()
            if not region_sales.empty:
                fig_region = px.bar(
                    region_sales,
                    x='Region',
                    y='Sales',
                    title='Sales by Region',
                    labels={'Sales': 'Sales ($)', 'Region': 'Region'},
                    color='Sales',
                    color_continuous_scale='Viridis'
                )
                fig_region.update_traces(
                    hovertemplate="Region: %{x}<br>Sales: $%{y:,.2f}<extra></extra>"
                )
                st.plotly_chart(fig_region, use_container_width=True)
            else:
                st.info("No regional sales data available.")

    # Add paginated data table
    st.subheader("📋 Detailed Sales Data")
    
    # Pagination controls
    page_size = st.selectbox("Rows per page", [10, 25, 50, 100])
    page_num = st.number_input("Page", min_value=1, value=1)
    
    # Get paginated data
    paginated_df, total_pages = paginate_dataframe(filtered_df, page_size=page_size, page_num=page_num)
    
    # Display paginated data
    st.dataframe(
        paginated_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Display pagination info
    st.caption(f"Page {page_num} of {total_pages} | Total records: {len(filtered_df)}")

# Forecasting Page
elif selected == "Forecasting":
    st.title("📈 Sales Forecasting")
    
    # Forecasting
    try:
        # Prepare data for forecasting
        daily_sales = filtered_df.groupby('Order Date')['Sales'].sum().reset_index()
        if len(daily_sales) < 30:  # Need at least 30 days of data
            st.warning("⚠️ Not enough data points for forecasting. Please select a wider date range (at least 30 days).")
        else:
            # Prepare data for Prophet
            prophet_df = daily_sales.rename(columns={'Order Date': 'ds', 'Sales': 'y'})
            
            # Initialize and fit Prophet model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True,
                changepoint_prior_scale=0.05
            )
            
            try:
                model.fit(prophet_df)
                
                # Make future predictions
                future = model.make_future_dataframe(periods=30)
                forecast = model.predict(future)
                
                # Plot forecast
                fig_forecast = go.Figure()
                
                # Add actual values
                fig_forecast.add_trace(go.Scatter(
                    x=prophet_df['ds'],
                    y=prophet_df['y'],
                    name='Actual',
                    line=dict(color='rgb(26, 118, 255)')
                ))
                
                # Add forecast
                fig_forecast.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat'],
                    name='Forecast',
                    line=dict(color='rgb(255, 65, 54)')
                ))
                
                # Add confidence intervals
                fig_forecast.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat_upper'],
                    fill=None,
                    mode='lines',
                    line_color='rgba(255, 65, 54, 0.2)',
                    name='Upper Bound'
                ))
                
                fig_forecast.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat_lower'],
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(255, 65, 54, 0.2)',
                    name='Lower Bound'
                ))
                
                fig_forecast.update_layout(
                    title='30-Day Sales Forecast',
                    xaxis_title='Date',
                    yaxis_title='Sales ($)',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                # Display forecast metrics
                last_actual = prophet_df['y'].iloc[-1]
                last_forecast = forecast['yhat'].iloc[-1]
                forecast_change = ((last_forecast - last_actual) / last_actual) * 100
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Last Actual Sales", f"${last_actual:,.2f}")
                with col2:
                    st.metric("30-Day Forecast", f"${last_forecast:,.2f}",
                             delta=f"{forecast_change:.1f}%")
                
            except Exception as e:
                st.error(f"Error in Prophet forecasting: {str(e)}")
                st.warning("⚠️ Could not generate forecast. Please check your data and try again.")

    except Exception as e:
        st.error(f"Error in forecasting section: {str(e)}")
        st.warning("⚠️ Forecasting could not be completed. Please check your data and try again.")

# Customer Analytics Page
elif selected == "Customer Analytics":
    st.title("👥 Customer Analytics")
    
    # RFM Analysis
    try:
        # Calculate RFM metrics using EmailID instead of Customer_ID
        current_date = filtered_df['Order Date'].max()
        rfm = filtered_df.groupby('EmailID').agg({
            'Order Date': lambda x: (current_date - x.max()).days,  # Recency in days
            'EmailID': 'count',  # Frequency
            'Sales': 'sum'  # Monetary
        }).rename(columns={
            'Order Date': 'Recency',
            'EmailID': 'Frequency',
            'Sales': 'Monetary'
        })

        # Handle edge cases
        if len(rfm) < 4:  # Need at least 4 customers for quartiles
            st.warning("⚠️ Not enough customers for RFM analysis. Need at least 4 customers.")
        else:
            # Calculate RFM scores
            r_labels = range(4, 0, -1)
            f_labels = range(1, 5)
            m_labels = range(1, 5)

            # Calculate quartiles with error handling
            try:
                r_quartiles = pd.qcut(rfm['Recency'], q=4, labels=r_labels, duplicates='drop')
                f_quartiles = pd.qcut(rfm['Frequency'], q=4, labels=f_labels, duplicates='drop')
                m_quartiles = pd.qcut(rfm['Monetary'], q=4, labels=m_labels, duplicates='drop')
            except ValueError as e:
                st.error(f"Error calculating RFM quartiles: {str(e)}")
                st.warning("⚠️ Not enough unique values for RFM analysis. Try selecting a different date range.")
                st.stop()

            rfm['R'] = r_quartiles
            rfm['F'] = f_quartiles
            rfm['M'] = m_quartiles

            # Calculate RFM Score
            rfm['RFM_Score'] = rfm['R'].astype(str) + rfm['F'].astype(str) + rfm['M'].astype(str)

            # Define segments
            def segment_customers(row):
                if pd.isna(row['R']) or pd.isna(row['F']) or pd.isna(row['M']):
                    return 'Unknown'
                r = int(row['R'])
                f = int(row['F'])
                m = int(row['M'])
                if r >= 4 and f >= 4 and m >= 4:
                    return 'Champions'
                elif r >= 3 and f >= 3 and m >= 3:
                    return 'Loyal Customers'
                elif r >= 3 and f >= 1 and m >= 1:
                    return 'Potential Loyalists'
                elif r >= 2 and f >= 2 and m >= 2:
                    return 'Recent Customers'
                elif r >= 2 and f >= 1 and m >= 1:
                    return 'Promising'
                elif r >= 1 and f >= 1 and m >= 1:
                    return 'Needs Attention'
                else:
                    return 'Lost Customers'

            rfm['Segment'] = rfm.apply(segment_customers, axis=1)

            # Display RFM Analysis
            st.subheader("📊 RFM Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Customers", len(rfm))
                st.metric("Average Recency", f"{rfm['Recency'].mean():.1f} days")
                st.metric("Average Frequency", f"{rfm['Frequency'].mean():.1f} orders")
                st.metric("Average Monetary", f"${rfm['Monetary'].mean():,.2f}")

            with col2:
                segment_counts = rfm['Segment'].value_counts()
                fig_segments = px.pie(
                    values=segment_counts.values,
                    names=segment_counts.index,
                    title='Customer Segments'
                )
                st.plotly_chart(fig_segments, use_container_width=True)

    except Exception as e:
        st.error(f"Error in RFM analysis: {str(e)}")
        st.warning("⚠️ RFM analysis could not be completed. Please check your data and try again.")

# Add new Advanced Analytics page
elif selected == "Advanced Analytics":
    st.title(" Advanced Analytics")
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs([
        "Seasonal Analysis", 
        "Customer Segmentation", 
        "Statistical Analysis"
    ])
    
    with tab1:
        st.subheader("📈 Seasonal Decomposition Analysis")
        
        # Prepare time series data
        daily_sales = filtered_df.groupby('Order Date')['Sales'].sum().reset_index()
        daily_sales.set_index('Order Date', inplace=True)
        
        if len(daily_sales) >= 30:  # Need enough data points
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(daily_sales['Sales'], period=7)
            
            # Plot decomposition
            fig = go.Figure()
            
            # Trend
            fig.add_trace(go.Scatter(
                x=decomposition.trend.index,
                y=decomposition.trend,
                name='Trend',
                line=dict(color='blue')
            ))
            
            # Seasonal
            fig.add_trace(go.Scatter(
                x=decomposition.seasonal.index,
                y=decomposition.seasonal,
                name='Seasonal',
                line=dict(color='green')
            ))
            
            # Residual
            fig.add_trace(go.Scatter(
                x=decomposition.resid.index,
                y=decomposition.resid,
                name='Residual',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title='Time Series Decomposition',
                xaxis_title='Date',
                yaxis_title='Sales',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display insights
            st.markdown("### Insights")
            st.markdown(f"""
                - **Trend Strength**: {abs(decomposition.trend).mean():.2f}
                - **Seasonal Strength**: {abs(decomposition.seasonal).mean():.2f}
                - **Residual Strength**: {abs(decomposition.resid).mean():.2f}
            """)
        else:
            st.warning("Not enough data points for seasonal decomposition. Please select a wider date range.")
    
    with tab2:
        st.subheader("👥 Customer Segmentation")
        
        # Prepare RFM data
        current_date = filtered_df['Order Date'].max()
        rfm = filtered_df.groupby('EmailID').agg({
            'Order Date': lambda x: (current_date - x.max()).days,
            'EmailID': 'count',
            'Sales': 'sum'
        }).rename(columns={
            'Order Date': 'Recency',
            'EmailID': 'Frequency',
            'Sales': 'Monetary'
        })
        
        if len(rfm) >= 4:  # Need enough customers
            # Scale the data
            scaler = StandardScaler()
            rfm_scaled = scaler.fit_transform(rfm)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=4, random_state=42)
            rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
            
            # Plot clusters
            fig = px.scatter_3d(
                rfm,
                x='Recency',
                y='Frequency',
                z='Monetary',
                color='Cluster',
                title='Customer Segments (3D View)',
                labels={
                    'Recency': 'Recency (days)',
                    'Frequency': 'Purchase Frequency',
                    'Monetary': 'Monetary Value ($)'
                }
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display cluster characteristics
            st.markdown("### Cluster Characteristics")
            cluster_stats = rfm.groupby('Cluster').agg({
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': 'mean'
            }).round(2)
            
            st.dataframe(cluster_stats)
            
            # Provide insights
            st.markdown("### Segment Insights")
            for cluster in rfm['Cluster'].unique():
                cluster_data = rfm[rfm['Cluster'] == cluster]
                st.markdown(f"""
                    **Segment {cluster}**:
                    - Average Recency: {cluster_data['Recency'].mean():.0f} days
                    - Average Frequency: {cluster_data['Frequency'].mean():.1f} purchases
                    - Average Monetary: ${cluster_data['Monetary'].mean():,.2f}
                """)
        else:
            st.warning("Not enough customers for segmentation. Please select a wider date range or more regions.")
    
    with tab3:
        st.subheader("📊 Statistical Analysis")
        
        # Correlation Analysis
        st.markdown("### Correlation Analysis")
        
        # Calculate correlations
        numeric_cols = ['Sales', 'Profit', 'Quantity']
        corr_matrix = filtered_df[numeric_cols].corr()
        
        # Plot correlation heatmap
        fig = px.imshow(
            corr_matrix,
            title='Correlation Matrix',
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary Statistics
        st.markdown("### Summary Statistics")
        summary_stats = filtered_df[numeric_cols].describe()
        st.dataframe(summary_stats)
        
        # Distribution Analysis
        st.markdown("### Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                filtered_df,
                x='Sales',
                title='Sales Distribution',
                nbins=50
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(
                filtered_df,
                y='Sales',
                title='Sales Box Plot'
            )
            st.plotly_chart(fig, use_container_width=True)

# Advanced Visualizations Page
elif selected == "Advanced Visualizations":
    st.title("🗺️ Advanced Visualizations")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs([
        "Sales Heat Map",
        "Animated Time Series",
        "Year-over-Year Analysis"
    ])
    
    with tab1:
        st.subheader("🌡️ Sales Density Heat Map")
        
        # Prepare data for heat map
        daily_sales_by_region = filtered_df.groupby(['Order Date', 'Region'])['Sales'].sum().reset_index()
        
        # Create pivot table for heat map
        heatmap_data = daily_sales_by_region.pivot_table(
            index='Order Date',
            columns='Region',
            values='Sales',
            aggfunc='sum'
        ).fillna(0)
        
        # Create heat map
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='Viridis',
            colorbar=dict(title='Sales ($)')
        ))
        
        fig.update_layout(
            title='Sales Density by Region and Time',
            xaxis_title='Region',
            yaxis_title='Date',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add region-specific insights
        st.subheader("Region Insights")
        region_stats = filtered_df.groupby('Region').agg({
            'Sales': ['sum', 'mean', 'count'],
            'Profit': 'sum'
        }).round(2)
        
        st.dataframe(region_stats)
    
    with tab2:
        st.subheader("📈 Animated Time Series")
        
        # Prepare data for animation
        daily_metrics = filtered_df.groupby('Order Date').agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Quantity': 'sum'
        }).reset_index()
        
        # Create animated line chart
        fig = go.Figure()
        
        for metric in ['Sales', 'Profit', 'Quantity']:
            fig.add_trace(go.Scatter(
                x=daily_metrics['Order Date'],
                y=daily_metrics[metric],
                name=metric,
                mode='lines',
                hovertemplate=f"{metric}: $%{{y:,.2f}}<extra></extra>"
            ))
        
        fig.update_layout(
            title='Animated Sales Metrics Over Time',
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='x unified',
            updatemenus=[{
                'buttons': [
                    {'method': 'animate', 'label': 'Play', 'args': [None, {'frame': {'duration': 500, 'redraw': True}}]},
                    {'method': 'animate', 'label': 'Pause', 'args': [[None], {'frame': {'duration': 0, 'redraw': True}}]}
                ],
                'type': 'buttons',
                'showactive': False,
                'x': 0.1,
                'y': 0,
                'xanchor': 'right',
                'yanchor': 'top'
            }]
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📊 Year-over-Year Analysis")
        
        # Prepare data for YoY comparison
        filtered_df['Year'] = filtered_df['Order Date'].dt.year
        filtered_df['Month'] = filtered_df['Order Date'].dt.month
        
        yoy_data = filtered_df.groupby(['Year', 'Month']).agg({
            'Sales': 'sum',
            'Profit': 'sum',
            'Quantity': 'sum'
        }).reset_index()
        
        # Create synchronized charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig_sales = go.Figure()
            for year in yoy_data['Year'].unique():
                year_data = yoy_data[yoy_data['Year'] == year]
                fig_sales.add_trace(go.Scatter(
                    x=year_data['Month'],
                    y=year_data['Sales'],
                    name=str(year),
                    mode='lines+markers'
                ))
            
            fig_sales.update_layout(
                title='Year-over-Year Sales Comparison',
                xaxis_title='Month',
                yaxis_title='Sales ($)',
                hovermode='x unified'
            )
            st.plotly_chart(fig_sales, use_container_width=True)
        
        with col2:
            fig_profit = go.Figure()
            for year in yoy_data['Year'].unique():
                year_data = yoy_data[yoy_data['Year'] == year]
                fig_profit.add_trace(go.Scatter(
                    x=year_data['Month'],
                    y=year_data['Profit'],
                    name=str(year),
                    mode='lines+markers'
                ))
            
            fig_profit.update_layout(
                title='Year-over-Year Profit Comparison',
                xaxis_title='Month',
                yaxis_title='Profit ($)',
                hovermode='x unified'
            )
            st.plotly_chart(fig_profit, use_container_width=True)

# Predictive Analytics Page
elif selected == "Predictive Analytics":
    st.title("🧠 Predictive Analytics")
    
    # Create tabs for different predictive analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "Churn Prediction",
        "Price Elasticity",
        "Product Recommendations",
        "Anomaly Detection"
    ])
    
    with tab1:
        st.subheader("👥 Customer Churn Analysis")
        
        # Calculate customer activity metrics
        current_date = filtered_df['Order Date'].max()
        customer_activity = filtered_df.groupby('EmailID').agg({
            'Order Date': lambda x: (current_date - x.max()).days,
            'Order ID': 'count',
            'Sales': 'sum'
        }).rename(columns={
            'Order Date': 'Days Since Last Purchase',
            'Order ID': 'Total Orders',
            'Sales': 'Total Spent'
        })
        
        # Define churn threshold (e.g., 90 days)
        churn_threshold = 90
        customer_activity['Churn Risk'] = customer_activity['Days Since Last Purchase'].apply(
            lambda x: 'High' if x > churn_threshold else 'Low'
        )
        
        # Display churn analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Churn Risk Customers",
                f"{len(customer_activity[customer_activity['Churn Risk'] == 'High'])}",
                f"{len(customer_activity[customer_activity['Churn Risk'] == 'High']) / len(customer_activity):.1%} of total"
            )
        
        with col2:
            fig = px.scatter(
                customer_activity,
                x='Days Since Last Purchase',
                y='Total Spent',
                color='Churn Risk',
                title='Customer Churn Risk Analysis',
                labels={
                    'Days Since Last Purchase': 'Days Since Last Purchase',
                    'Total Spent': 'Total Spent ($)'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("💰 Price Elasticity Analysis")
        
        # Calculate price elasticity
        product_metrics = filtered_df.groupby('Product Name').agg({
            'Sales': 'sum',
            'Quantity': 'sum'
        }).reset_index()
        
        # Calculate unit price from Sales and Quantity
        product_metrics['Unit_Price'] = product_metrics['Sales'] / product_metrics['Quantity']
        
        # Calculate price elasticity (simplified version)
        product_metrics['Price Elasticity'] = (
            (product_metrics['Quantity'].pct_change() / product_metrics['Unit_Price'].pct_change())
            .fillna(0)
            .clip(-10, 10)  # Clip extreme values
        )
        
        # Display price elasticity analysis
        fig = px.scatter(
            product_metrics,
            x='Unit_Price',
            y='Quantity',
            size='Sales',
            color='Price Elasticity',
            title='Price Elasticity Analysis',
            labels={
                'Unit_Price': 'Average Price ($)',
                'Quantity': 'Total Quantity Sold',
                'Sales': 'Total Sales ($)'
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display insights
        st.markdown("### Price Elasticity Insights")
        elastic_products = product_metrics[abs(product_metrics['Price Elasticity']) > 1]
        inelastic_products = product_metrics[abs(product_metrics['Price Elasticity']) <= 1]
        
        st.markdown(f"""
            - **Elastic Products** (price sensitive): {len(elastic_products)}
            - **Inelastic Products** (price insensitive): {len(inelastic_products)}
        """)
        
        # Add detailed product analysis
        st.markdown("### Top 5 Most Elastic Products")
        top_elastic = product_metrics.nlargest(5, 'Price Elasticity')
        st.dataframe(top_elastic[['Product Name', 'Unit_Price', 'Quantity', 'Price Elasticity']].round(2))
        
        st.markdown("### Top 5 Most Inelastic Products")
        top_inelastic = product_metrics.nsmallest(5, 'Price Elasticity')
        st.dataframe(top_inelastic[['Product Name', 'Unit_Price', 'Quantity', 'Price Elasticity']].round(2))
    
    with tab3:
        st.subheader("🎯 Product Recommendations")
        
        # Calculate product correlations
        product_pairs = filtered_df.groupby(['Order ID', 'Product Name'])['Quantity'].sum().unstack().fillna(0)
        product_corr = product_pairs.corr()
        
        # Get top recommendations for each product
        def get_top_recommendations(product, n=3):
            if product in product_corr.columns:
                return product_corr[product].sort_values(ascending=False)[1:n+1]
            return pd.Series()
        
        # Display product recommendations
        st.markdown("### Product Recommendations")
        
        # Select a product to get recommendations
        selected_product = st.selectbox(
            "Select a product to see recommendations",
            options=sorted(filtered_df['Product Name'].unique())
        )
        
        recommendations = get_top_recommendations(selected_product)
        if not recommendations.empty:
            st.markdown("#### Top Recommendations")
            for product, correlation in recommendations.items():
                st.markdown(f"- **{product}** (Correlation: {correlation:.2f})")
        else:
            st.warning("No recommendations available for this product.")
    
    with tab4:
        st.subheader("⚠️ Sales Anomaly Detection")
        
        # Calculate daily sales
        daily_sales = filtered_df.groupby('Order Date')['Sales'].sum().reset_index()
        
        # Calculate moving average and standard deviation
        window = 7
        daily_sales['MA'] = daily_sales['Sales'].rolling(window=window).mean()
        daily_sales['STD'] = daily_sales['Sales'].rolling(window=window).std()
        
        # Define anomalies (values outside 2 standard deviations)
        daily_sales['Anomaly'] = abs(daily_sales['Sales'] - daily_sales['MA']) > (2 * daily_sales['STD'])
        
        # Plot anomalies
        fig = go.Figure()
        
        # Add normal sales
        fig.add_trace(go.Scatter(
            x=daily_sales['Order Date'],
            y=daily_sales['Sales'],
            name='Sales',
            mode='lines',
            line=dict(color='blue')
        ))
        
        # Add moving average
        fig.add_trace(go.Scatter(
            x=daily_sales['Order Date'],
            y=daily_sales['MA'],
            name='Moving Average',
            mode='lines',
            line=dict(color='green')
        ))
        
        # Add anomalies
        anomalies = daily_sales[daily_sales['Anomaly']]
        fig.add_trace(go.Scatter(
            x=anomalies['Order Date'],
            y=anomalies['Sales'],
            name='Anomalies',
            mode='markers',
            marker=dict(color='red', size=10)
        ))
        
        fig.update_layout(
            title='Sales Anomaly Detection',
            xaxis_title='Date',
            yaxis_title='Sales ($)',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display anomaly insights
        if not anomalies.empty:
            st.markdown("### Detected Anomalies")
            for _, row in anomalies.iterrows():
                st.markdown(f"""
                    - **Date**: {row['Order Date'].strftime('%Y-%m-%d')}
                    - **Sales**: ${row['Sales']:,.2f}
                    - **Deviation**: {abs(row['Sales'] - row['MA']):,.2f} from moving average
                """)
        else:
            st.info("No significant anomalies detected in the selected period.")

# Footer with additional information
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Dashboard created with Streamlit • Data is from Amazon 2_Raw.xlsx for demonstration purposes</p>
        <p style='font-size: 0.8em; color: #666;'>Last updated: {}</p>
    </div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True) 