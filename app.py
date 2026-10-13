import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

# Set Page Config
st.set_page_config(page_title="Davis Law Master KPI Dashboard", layout="wide")

# --- DATA ---
@st.cache_data
def get_data():
    raw_data = {
        "Month_Year": ["Jan 2024", "Feb 2024", "Mar 2024", "Apr 2024", "May 2024", "Jun 2024", "Jul 2024", "Aug 2024", "Sep 2024", "Oct 2024", "Nov 2024", "Dec 2024", "Jan 2025", "Feb 2025", "Mar 2025", "Apr 2025", "May 2025", "Jun 2025", "Jul 2025", "Aug 2025", "Sep 2025", "Oct 2025", "Nov 2025", "Dec 2025", "Jan 2026", "Feb 2026", "Mar 2026", "Apr 2026", "May 2026", "Jun 2026", "Jul 2026"],
        "Signed_Clients": [31, 30, 51, 33, 47, 50, 80, 99, 61, 62, 63, 52, 63, 50, 56, 66, 71, 55, 56, 51, 62, 69, 70, 117, 114, 87, 76, 71, 78, 89, 95],
        "Revenue": [400000, 410000, 450000, 420000, 460000, 440000, 480000, 500000, 490000, 510000, 629674, 519731, 629674, 499742, 559710, 659472, 709432, 549560, 559552, 509592, 619504, 689448, 699440, 1169064, 1139088, 869304, 759392, 709432, 779376, 889288, 920000],
        "Marketing_Spend": [150000, 150000, 150000, 150000, 150000, 150000, 150000, 150000, 150000, 150000, 196132, 206062, 180443, 208606, 206888, 213348, 162098, 160124, 160623, 211141, 143438, 156950, 123626, 122639, 129196, 129196, 131379, 131379, 135000, 135000, 135000],
    }
    df = pd.DataFrame(raw_data)
    
    # Clean up month names (Standardize to 3 letters)
    df['Month_Year'] = df['Month_Year'].str.replace('Sept', 'Sep').str.replace('July', 'Jul')
    
    # Convert to Date
    df['Date'] = pd.to_datetime(df['Month_Year'], format='%b %Y', errors='coerce')
    
    # Drop any rows where date conversion failed
    df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
    
    # Calculations
    df['ROI'] = df['Revenue'] / df['Marketing_Spend']
    df['CAC'] = df['Marketing_Spend'] / df['Signed_Clients']
    return df

df = get_data()

# --- SIDEBAR ---
st.sidebar.title("Davis Law Navigation")
mode = st.sidebar.radio("View Mode", ["Executive Summary", "Predictive Forecast", "Strategic Spend Planner"])

# --- MODE 1: EXECUTIVE ---
if mode == "Executive Summary":
    st.title("⚖️ Firm Performance Overview")
    latest = df.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue", f"${latest['Revenue']:,.0f}")
    c2.metric("Signed Clients", int(latest['Signed_Clients']))
    c3.metric("Marketing ROI", f"{latest['ROI']:.2f}x")
    c4.metric("Avg CAC", f"${latest['CAC']:,.0f}")
    
    st.plotly_chart(px.line(df, x='Date', y='Revenue', title="Revenue Growth History", markers=True), use_container_width=True)

# --- MODE 2: FORECAST ---
elif mode == "Predictive Forecast":
    st.title("🔮 End-of-Year Projection")
    
    # Prepare data for regression
    df_forecast = df.copy()
    df_forecast['days_since'] = (df_forecast['Date'] - df_forecast['Date'].min()).dt.days
    
    # Train Model
    X = df_forecast[['days_since']].values
    y = df_forecast['Revenue'].values
    model = LinearRegression().fit(X, y)
    
    # Predict 5 months out
    future_dates = [df_forecast['Date'].max() + timedelta(days=30*i) for i in range(1, 6)]
    future_days = np.array([(d - df_forecast['Date'].min()).days for d in future_dates]).reshape(-1,1)
    preds = model.predict(future_days)
    
    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_forecast['Date'], y=df_forecast['Revenue'], name="Actual Revenue", mode='lines+markers'))
    fig.add_trace(go.Scatter(x=future_dates, y=preds, name="AI Projection", line=dict(dash='dash', color='orange')))
    
    fig.update_layout(title="Revenue Forecast through December 2026", xaxis_title="Date", yaxis_title="Revenue ($)")
    st.plotly_chart(fig, use_container_width=True)
    st.write("This model uses linear regression based on your historical growth since Jan 2024.")

# --- MODE 3: STRATEGY ---
elif mode == "Strategic Spend Planner":
    st.title("🛠️ Spend Scenario Lab")
    pct = st.sidebar.slider("Change Monthly Budget %", -50, 100, 0, step=5)
    
    # Calculate baseline from last 6 months
    avg_roi = df.tail(6)['ROI'].mean()
    current_spend = df.iloc[-1]['Marketing_Spend']
    planned_spend = current_spend * (1 + pct/100)
    projected_rev = planned_spend * avg_roi
    
    col1, col2 = st.columns(2)
    col1.metric("New Monthly Budget", f"${planned_spend:,.0f}")
    col2.metric("Projected Monthly Revenue", f"${projected_rev:,.0f}", f"{pct}% shift")
    
    st.info(f"Based on your recent 6-month average ROI of **{avg_roi:.2f}x**, this budget shift would impact your top line as shown above.")
