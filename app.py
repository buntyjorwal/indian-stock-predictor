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

if symbol_input == "^NSEI":
    symbol = "^NSEI"
elif not symbol_input.endswith((".NS", ".BO")):
    symbol = symbol_input + ".NS"
else:
    symbol = symbol_input

days = st.sidebar.slider("Days to Predict", min_value=7, max_value=60, value=15)

if st.sidebar.button("Fetch Data & Predict", type="primary"):
    with st.spinner(f"Fetching data for **{symbol}** (this may take 10-20 seconds)..."):
        data = pd.DataFrame()
        
        # Retry logic
        for attempt in range(4):
            try:
                data = yf.download(symbol, period="2y", interval="1d", 
                                 progress=False, timeout=30, auto_adjust=True)
                if not data.empty:
                    break
            except:
                pass
            time.sleep(3)
        
        # Fallback to 1 year if 2y fails
        if data.empty:
            try:
                data = yf.download(symbol, period="1y", interval="1d", 
                                 progress=False, auto_adjust=True)
            except:
                pass

        if data.empty:
            st.error(f"❌ Sorry, could not fetch data for **{symbol}** right now.")
            st.info("**Suggested symbols to try:**")
            st.info("• RELIANCE.NS\n• HDFCBANK.NS\n• TCS.NS\n• INFY.NS\n• SBIN.NS\n• ^NSEI (Nifty 50)")
            st.info("💡 yfinance sometimes has temporary issues with Indian stocks on cloud. Try again in a few minutes or refresh the page.")
        else:
            # Safe price extraction
            try:
                current_price = float(data['Close'].iloc[-1])
            except:
                current_price = float(data['Close'].mean())  # fallback

            st.success(f"✅ Successfully loaded **{symbol}**")
            st.metric("Current Price", f"₹ {current_price:,.2f}")

            # Candlestick Chart
            fig = go.Figure(data=[go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close']
            )])
            fig.update_layout(title=f"{symbol} - Historical Price", xaxis_title="Date", yaxis_title="Price (₹)")
            st.plotly_chart(fig, use_container_width=True)

            # Prophet Prediction
            st.subheader(f"📈 Prediction for Next {days} Days")
            df_prophet = data[['Close']].reset_index()
            df_prophet.columns = ['ds', 'y']

            try:
                model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
                model.fit(df_prophet)

                future = model.make_future_dataframe(periods=days)
                forecast = model.predict(future)

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], 
                                        name="Predicted", line=dict(color="#00ff88", width=2)))
                fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], 
                                        name="Lower Bound", line=dict(dash="dot")))
                fig2.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], 
                                        name="Upper Bound", line=dict(dash="dot")))
                fig2.update_layout(title="Price Forecast", xaxis_title="Date", yaxis_title="Price (₹)")
                st.plotly_chart(fig2, use_container_width=True)

                # Table
                pred_table = forecast[['ds', 'yhat']].tail(days).round(2)
                pred_table = pred_table.rename(columns={"ds": "Date", "yhat": "Predicted Price (₹)"})
                st.dataframe(pred_table, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

st.caption("⚠️ Educational tool only • Not financial advice • Market data via Yahoo Finance")