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

# Auto add .NS if needed
if symbol_input == "^NSEI":
    symbol = "^NSEI"
elif not symbol_input.endswith(".NS"):
    symbol = symbol_input + ".NS"
else:
    symbol = symbol_input

days = st.sidebar.slider("Days to Predict", min_value=7, max_value=60, value=15)

if st.sidebar.button("Fetch Data & Predict", type="primary"):
    with st.spinner(f"Fetching latest data for **{symbol}**..."):
        data = None
        for attempt in range(3):  # Retry up to 3 times
            try:
                data = yf.download(symbol, period="2y", interval="1d", progress=False, timeout=30)
                if not data.empty:
                    break
            except:
                pass
            time.sleep(2)  # Wait before retry

        if data is None or data.empty:
            # Try with shorter period as fallback
            try:
                data = yf.download(symbol, period="1y", interval="1d", progress=False)
            except:
                pass

        if data is None or data.empty:
            st.error(f"❌ Could not fetch data for **{symbol}**. ")
            st.info("**Try these symbols:** RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, SBIN.NS, ^NSEI (Nifty 50)")
            st.info("Tip: Sometimes yfinance has temporary issues with Indian stocks. Refresh after 30 seconds.")
        else:
            current_price = float(data['Close'].iloc[-1])
            st.success(f"✅ Data loaded successfully for **{symbol}**")
            st.metric("Current Price", f"₹ {current_price:,.2f}")

            # Candlestick Chart
            fig = go.Figure(data=[go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close']
            )])
            fig.update_layout(title=f"{symbol} - Price History (Last 2 Years)", xaxis_title="Date", yaxis_title="Price (₹)")
            st.plotly_chart(fig, use_container_width=True)

            # Prophet Prediction
            st.subheader(f"📈 Predicted Price for Next {days} Days")
            df = data[['Close']].reset_index()
            df.columns = ['ds', 'y']

            try:
                model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=True)
                model.fit(df)

                future = model.make_future_dataframe(periods=days)
                forecast = model.predict(future)

                # Prediction Chart
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Predicted Price", line=dict(color="#00ff88")))
                fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name="Lower Bound", line=dict(dash="dot", color="#ffaa00")))
                fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name="Upper Bound", line=dict(dash="dot", color="#ffaa00")))
                fig2.update_layout(title="Price Forecast", xaxis_title="Date", yaxis_title="Price (₹)")
                st.plotly_chart(fig2, use_container_width=True)

                # Table
                pred_table = forecast[['ds', 'yhat']].tail(days).round(2)
                pred_table.columns = ["Date", "Predicted Price (₹)"]
                st.dataframe(pred_table, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

st.caption("⚠️ This is for learning/educational purpose only. Not financial advice. Stock prices are volatile.")