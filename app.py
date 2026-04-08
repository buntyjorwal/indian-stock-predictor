import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
import plotly.graph_objects as go
import time

st.set_page_config(page_title="Indian Stock Market Predictor", page_icon="📈", layout="wide")

st.title("📊 Indian Stock Market Predictor")
st.markdown("**Real-time NSE/BSE data + Simple Prophet Prediction**")

# Sidebar
st.sidebar.header("Stock Selection")
symbol_input = st.sidebar.text_input("Enter Stock Symbol", value="RELIANCE.NS").strip().upper()

if "NIFTY" in symbol_input or symbol_input == "NSEI":
    symbol = "^NSEI"
elif not symbol_input.endswith((".NS", ".BO")):
    symbol = symbol_input + ".NS"
else:
    symbol = symbol_input

days = st.sidebar.slider("Days to Predict", min_value=7, max_value=60, value=15)

if st.sidebar.button("Fetch Data & Predict", type="primary"):
    with st.spinner(f"Fetching data for **{symbol}**... (may take 15-30 seconds due to yfinance limits)"):
        data = pd.DataFrame()
        
        for attempt in range(5):
            try:
                data = yf.download(symbol, period="2y", interval="1d", progress=False, timeout=40, auto_adjust=True)
                if len(data) > 20:   # at least some data
                    break
            except:
                pass
            time.sleep(4)
        
        if len(data) < 20:
            st.error(f"❌ Could not fetch enough data for **{symbol}** right now.")
            st.info("**Why?** yfinance often has temporary blocks for Indian stocks on cloud servers.")
            st.info("**Try these exact symbols:** RELIANCE.NS, HDFCBANK.NS, TCS.NS, INFY.NS, SBIN.NS, ^NSEI")
            st.info("💡 Refresh the page after 1-2 minutes or try on your local laptop (streamlit run app.py)")
        else:
            current_price = float(data['Close'].iloc[-1])
            st.success(f"✅ Data loaded for **{symbol}**")
            st.metric("Current Price", f"₹ {current_price:,.2f}")

            # Charts and prediction (same as before)
            fig = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])])
            fig.update_layout(title=f"{symbol} Price History", xaxis_title="Date", yaxis_title="Price (₹)")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader(f"📈 Prediction for Next {days} Days")
            df = data[['Close']].reset_index()
            df.columns = ['ds', 'y']

            model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
            model.fit(df)
            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Predicted", line=dict(color="#00ff88")))
            fig2.update_layout(title="Forecast", xaxis_title="Date", yaxis_title="Price (₹)")
            st.plotly_chart(fig2, use_container_width=True)

            st.dataframe(forecast[['ds', 'yhat']].tail(days).round(2).rename(columns={"ds":"Date", "yhat":"Predicted ₹"}), use_container_width=True)

st.caption("⚠️ This is an educational demo only. Not financial advice. Data from Yahoo Finance (unreliable for .NS on cloud).")