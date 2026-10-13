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
        "Month_Year": ["Jan 2024", "Feb 2024", "Mar 2024", "Apr 2024", "May 2024", "Jun 2024", "Jul 2024", "Aug 2024", "Sept 2024", "Oct 2024", "Nov 2024", "Dec 2024", "Jan 2025", "Feb 2025", "Mar 2025", "Apr 2025", "May 2025", "Jun 2025", "Jul 2025", "Aug 2025", "Sept 2025", "Oct 2025", "Nov 2025", "Dec 2025", "Jan 2026", "Feb 2026", "Mar 2026", "Apr 2026", "May 2026", "Jun 2026", "July 2026"],
        "Signed_Clients": [31, 30, 51, 33, 47, 50, 80, 99, 61, 62, 63, 52, 63, 50, 56, 66, 71, 55, 56, 51, 62, 69, 70, 117, 114, 87, 76, 71, 78, 89, 95],
        "Revenue": [400000, 410000, 450000, 420000, 460000, 440000, 480000, 500000, 490000, 510000, 629674, 519731, 629674, 499742, 559710, 659472, 709432, 549560, 559552, 509592, 619504, 689448, 699440, 1169064, 1139088, 869304, 759392, 709432, 779376, 889288, 920000],
        "Marketing_Spend": [150000, 150000, 150000, 150000, 150000, 150000, 150000, 150000, 150000, 150000, 196132, 206062, 180443, 208606, 206888, 213348, 162098, 160124, 160623, 211141, 143438, 156950, 123626, 122639, 129196, 129196, 131379, 131379, 135000, 135000, 135000],
        "Total_Leads": [197, 168, 177, 275, 333, 242, 280, 321, 318, 278, 161, 153, 167, 176, 204, 190, 179, 205, 232, 226, 187, 174, 147, 220, 303, 366, 303, 378, 381, 390, 400]
    }
    df = pd.DataFrame(raw_data)
    df['Date'] = pd.to_datetime(df['Month_Year'], format='%b %Y', errors='coerce')
    df.loc[df['Month_Year'] == 'July 2026', 'Date'] = datetime(2026, 7, 1)
    df = df.sort_values('Date').reset_index(drop=True)
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
    
    st.plotly_chart(px.line(df, x='Date', y='Revenue', title="Revenue Growth"), use_container_width=True)

# --- MODE 2: FORECAST ---
elif mode == "Predictive Forecast":
    st.title("🔮 End-of-Year Projection")
    df['days_since'] = (df['Date'] - df['Date'].min()).dt.days
    model = LinearRegression().fit(df[['days_since']], df['Revenue'])
    
    future_dates = [df['Date'].max() + timedelta(days=30*i) for i in range(1, 6)]
    future_days = np.array([(d - df['Date'].min()).days for d in future_dates]).reshape(-1,1)
    preds = model.predict(future_days)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Revenue'], name="Actual"))
    fig.add_trace(go.Scatter(x=future_dates, y=preds, name="AI Projection", line=dict(dash='dash')))
    st.plotly_chart(fig, use_container_width=True)

# --- MODE 3: STRATEGY ---
elif mode == "Strategic Spend Planner":
    st.title("🛠️ Spend Scenario Lab")
    pct = st.sidebar.slider("Change Budget %", -50, 100, 0)
    avg_roi = df.tail(6)['ROI'].mean()
    new_rev = df.iloc[-1]['Marketing_Spend'] * (1 + pct/100) * avg_roi
    
    st.metric("Potential Monthly Revenue", f"${new_rev:,.0f}", f"{pct}% spend change")
    st.info("Adjust the slider in the sidebar to see how changing marketing spend affects your revenue.")
