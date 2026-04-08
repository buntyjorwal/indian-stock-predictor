import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta

st.set_page_config(page_title="Indian Stock Predictor", page_icon="📈", layout="wide")

st.title("📊 Indian Stock Market Predictor")
st.markdown("**Real-time NSE/BSE data + Simple Prophet Prediction**")

# Sidebar
st.sidebar.header("Stock Selection")
symbol_input = st.sidebar.text_input("Enter Stock Symbol", value="RELIANCE.NS").upper().strip()
if not symbol_input.endswith(".NS") and symbol_input != "^NSEI":
    symbol = symbol_input + ".NS"
else:
    symbol = symbol_input

days = st.sidebar.slider("Days to Predict", min_value=7, max_value=60, value=15)

if st.sidebar.button("Fetch Data & Predict", type="primary"):
    with st.spinner(f"Downloading data for {symbol}..."):
        try:
            data = yf.download(symbol, period="2y", interval="1d", progress=False)
            
            if data.empty:
                st.error(f"❌ No data found for **{symbol}**. Try RELIANCE.NS, TCS.NS, HDFCBANK.NS, or ^NSEI")
            else:
                current_price = float(data['Close'].iloc[-1])
                st.metric(label="Current Price", value=f"₹ {current_price:,.2f}")
                
                # Candlestick chart
                fig = go.Figure(data=[go.Candlestick(x=data.index,
                                                     open=data['Open'],
                                                     high=data['High'],
                                                     low=data['Low'],
                                                     close=data['Close'])])
                fig.update_layout(title=f"{symbol} - Price History", xaxis_title="Date", yaxis_title="Price (₹)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Prophet Prediction
                df = data['Close'].reset_index()
                df.columns = ['ds', 'y']
                
                model = Prophet(yearly_seasonality=True, daily_seasonality=False)
                model.fit(df)
                
                future = model.make_future_dataframe(periods=days)
                forecast = model.predict(future)
                
                st.subheader(f"📈 Prediction for Next {days} Days")
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Predicted", line=dict(color="#00ff00")))
                fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name="Lower", line=dict(dash="dot")))
                fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name="Upper", line=dict(dash="dot")))
                st.plotly_chart(fig2, use_container_width=True)
                
                st.dataframe(forecast[['ds', 'yhat']].tail(days).round(2).rename(columns={"ds": "Date", "yhat": "Predicted Price (₹)"}))
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Make sure the stock symbol is correct (e.g., RELIANCE.NS)")

st.caption("Note: This is a simple statistical prediction, not financial advice. Past performance ≠ future results.")
