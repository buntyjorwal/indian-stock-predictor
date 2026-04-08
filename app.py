Python 3.14.4 (tags/v3.14.4:23116f9, Apr  7 2026, 14:10:54) [MSC v.1944 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Indian Stock Predictor", page_icon="📈", layout="wide")
st.title("📊 Indian Stock Market Predictor")
st.write("Real-time data + Future Price Prediction (NSE/BSE)")

# Sidebar
st.sidebar.header("Choose Stock")
stock_symbol = st.sidebar.text_input("Enter Stock Symbol (add .NS)", value="RELIANCE.NS")
days_to_predict = st.sidebar.slider("Predict next how many days?", 7, 60, 15)

# Fetch data
if st.sidebar.button("Get Data & Predict"):
    with st.spinner("Fetching latest market data..."):
        data = yf.download(stock_symbol, period="2y", interval="1d")
        
        if data.empty:
            st.error("Invalid symbol or no data. Try RELIANCE.NS, TCS.NS, HDFCBANK.NS, or ^NSEI")
        else:
            st.success(f"✅ Showing data for **{stock_symbol}**")
            
            # Current price
            current_price = data['Close'][-1]
            st.metric("Current Price (₹)", f"{current_price:.2f}")
            
...             # Chart
...             fig = go.Figure()
...             fig.add_trace(go.Candlestick(x=data.index,
...                                         open=data['Open'], high=data['High'],
...                                         low=data['Low'], close=data['Close'],
...                                         name="Price"))
...             fig.update_layout(title=f"{stock_symbol} - Last 2 Years", xaxis_title="Date", yaxis_title="Price (₹)")
...             st.plotly_chart(fig, use_container_width=True)
...             
...             # Prediction with Prophet
...             df = data[['Close']].reset_index()
...             df.columns = ['ds', 'y']
...             
...             model = Prophet(daily_seasonality=True)
...             model.fit(df)
...             
...             future = model.make_future_dataframe(periods=days_to_predict)
...             forecast = model.predict(future)
...             
...             # Show prediction chart
...             st.subheader(f"📈 Predicted Price for Next {days_to_predict} Days")
...             fig2 = go.Figure()
...             fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Predicted Price", line=dict(color="blue")))
...             fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name="Lower Bound", line=dict(dash="dot")))
...             fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name="Upper Bound", line=dict(dash="dot")))
...             st.plotly_chart(fig2, use_container_width=True)
...             
...             # Show predicted prices in table
...             st.subheader("Predicted Prices (Last 5 days shown)")
...             prediction_table = forecast[['ds', 'yhat']].tail(days_to_predict + 5).round(2)
...             prediction_table.columns = ["Date", "Predicted Price (₹)"]
...             st.dataframe(prediction_table)
... 
