import time
import pandas as pd
import streamlit as st
import yfinance as yf
from prophet import Prophet
import plotly.graph_objects as go

st.set_page_config(page_title="Indian Stock Market Predictor", page_icon="📈", layout="wide")

st.title("📊 Indian Stock Market Predictor")
st.markdown("**Real-time NSE/BSE data + Simple Prophet Prediction**")


def normalize_symbol(symbol_input: str) -> str:
    s = (symbol_input or "").strip().upper()
    if s in ["NIFTY", "NSEI", "^NSEI"]:
        return "^NSEI"
    if not s.endswith((".NS", ".BO")) and not s.startswith("^"):
        return s + ".NS"
    return s


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x) for x in col if str(x).strip() != ""]).strip("_")
            for col in df.columns.values
        ]
    return df


def find_price_column(df: pd.DataFrame, base_name: str) -> str:
    if base_name in df.columns:
        return base_name
    matches = [c for c in df.columns if str(c).startswith(base_name)]
    return matches[0] if matches else ""


def clean_market_data(df: pd.DataFrame) -> pd.DataFrame:
    df = flatten_columns(df.copy())

    open_col = find_price_column(df, "Open")
    high_col = find_price_column(df, "High")
    low_col = find_price_column(df, "Low")
    close_col = find_price_column(df, "Close")

    required = {
        "Open": open_col,
        "High": high_col,
        "Low": low_col,
        "Close": close_col,
    }

    missing = [k for k, v in required.items() if not v]
    if missing:
        raise ValueError(f"Required columns not found: {', '.join(missing)}. Returned columns: {list(df.columns)}")

    cleaned = pd.DataFrame(index=df.index)
    cleaned["Open"] = pd.to_numeric(df[open_col], errors="coerce")
    cleaned["High"] = pd.to_numeric(df[high_col], errors="coerce")
    cleaned["Low"] = pd.to_numeric(df[low_col], errors="coerce")
    cleaned["Close"] = pd.to_numeric(df[close_col], errors="coerce")
    cleaned = cleaned.dropna(subset=["Open", "High", "Low", "Close"]).copy()

    return cleaned


@st.cache_data(show_spinner=False, ttl=900)
def fetch_stock_data(symbol: str) -> pd.DataFrame:
    last_error = None

    for _ in range(5):
        try:
            raw = yf.download(
                symbol,
                period="2y",
                interval="1d",
                progress=False,
                auto_adjust=True,
                threads=False
            )

            if raw is None or raw.empty:
                time.sleep(3)
                continue

            cleaned = clean_market_data(raw)
            if len(cleaned) >= 20:
                return cleaned

        except Exception as ex:
            last_error = ex

        time.sleep(3)

    if last_error:
        raise last_error

    return pd.DataFrame()


def build_forecast(data: pd.DataFrame, days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = data[["Close"]].reset_index().copy()
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "ds", "Close": "y"})
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["ds", "y"]).copy()

    if len(df) < 20:
        raise ValueError("Not enough clean data for prediction.")

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True
    )
    model.fit(df)

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    return df, forecast


# Sidebar
st.sidebar.header("Stock Selection")

quick_stocks = {
    "RELIANCE": "RELIANCE.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "SBIN": "SBIN.NS",
    "NIFTY": "^NSEI",
}

selected_quick = st.sidebar.selectbox(
    "Quick Select",
    ["Custom"] + list(quick_stocks.keys()),
    index=0
)

default_symbol = "RELIANCE.NS"
if selected_quick != "Custom":
    default_symbol = quick_stocks[selected_quick]

symbol_input = st.sidebar.text_input("Enter Stock Symbol", value=default_symbol).strip().upper()
symbol = normalize_symbol(symbol_input)

days = st.sidebar.slider("Days to Predict", min_value=7, max_value=60, value=15)

run_btn = st.sidebar.button("Fetch Data & Predict", type="primary")

if run_btn:
    with st.spinner(f"Fetching data for **{symbol}**..."):
        try:
            data = fetch_stock_data(symbol)
        except Exception as ex:
            data = pd.DataFrame()
            st.error(f"❌ Error while fetching data for **{symbol}**")
            st.code(str(ex))

    if data.empty or len(data) < 20:
        st.error(f"❌ Could not fetch enough data for **{symbol}** right now.")
        st.info("Try these symbols: RELIANCE.NS, HDFCBANK.NS, TCS.NS, INFY.NS, SBIN.NS, ^NSEI")
    else:
        current_price = float(pd.to_numeric(data["Close"], errors="coerce").dropna().iloc[-1])
        prev_price = float(pd.to_numeric(data["Close"], errors="coerce").dropna().iloc[-2])
        price_change = current_price - prev_price
        pct_change = (price_change / prev_price * 100) if prev_price else 0

        st.success(f"✅ Data loaded for **{symbol}**")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price", f"₹ {current_price:,.2f}", f"{price_change:,.2f}")
        c2.metric("Daily Change %", f"{pct_change:,.2f}%")
        c3.metric("Data Points", f"{len(data)}")
        c4.metric("Prediction Horizon", f"{days} days")

        # Historical Candlestick
        st.subheader(f"📉 {symbol} Price History")
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=data.index,
                    open=data["Open"],
                    high=data["High"],
                    low=data["Low"],
                    close=data["Close"],
                    name=symbol
                )
            ]
        )
        fig.update_layout(
            title=f"{symbol} Historical Candlestick",
            xaxis_title="Date",
            yaxis_title="Price (₹)",
            xaxis_rangeslider_visible=False,
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # Forecast
        st.subheader(f"📈 Prediction for Next {days} Days")

        try:
            hist_df, forecast = build_forecast(data, days)

            future_rows = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(days).copy()
            final_pred = float(future_rows["yhat"].iloc[-1])
            final_lower = float(future_rows["yhat_lower"].iloc[-1])
            final_upper = float(future_rows["yhat_upper"].iloc[-1])

            trend = "Bullish 📈" if final_pred > current_price else "Bearish 📉"
            expected_move = final_pred - current_price
            expected_move_pct = (expected_move / current_price * 100) if current_price else 0

            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Predicted End Price", f"₹ {final_pred:,.2f}")
            a2.metric("Expected Move", f"₹ {expected_move:,.2f}", f"{expected_move_pct:,.2f}%")
            a3.metric("Range Low", f"₹ {final_lower:,.2f}")
            a4.metric("Range High", f"₹ {final_upper:,.2f}")

            st.info(f"Prediction Status: Complete ✅ | Trend Outlook: {trend}")

            # Forecast chart with confidence range
            fig2 = go.Figure()

            fig2.add_trace(
                go.Scatter(
                    x=hist_df["ds"],
                    y=hist_df["y"],
                    mode="lines",
                    name="Historical"
                )
            )

            fig2.add_trace(
                go.Scatter(
                    x=forecast["ds"],
                    y=forecast["yhat_upper"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    name="Upper Band"
                )
            )

            fig2.add_trace(
                go.Scatter(
                    x=forecast["ds"],
                    y=forecast["yhat_lower"],
                    mode="lines",
                    fill="tonexty",
                    line=dict(width=0),
                    name="Confidence Range"
                )
            )

            fig2.add_trace(
                go.Scatter(
                    x=forecast["ds"],
                    y=forecast["yhat"],
                    mode="lines",
                    name="Predicted"
                )
            )

            fig2.update_layout(
                title="Historical + Forecast with Confidence Range",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                height=520
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Next days table
            st.subheader("📋 Next Days Forecast Table")
            prediction_table = (
                future_rows
                .round(2)
                .rename(
                    columns={
                        "ds": "Date",
                        "yhat": "Predicted ₹",
                        "yhat_lower": "Lower ₹",
                        "yhat_upper": "Upper ₹"
                    }
                )
            )
            st.dataframe(prediction_table, use_container_width=True)

        except Exception as ex:
            st.error("❌ Prediction part incomplete due to processing error.")
            st.code(str(ex))

st.caption("⚠️ This is an educational demo only. Not financial advice. Data from Yahoo Finance may be unreliable for some NSE/BSE symbols.")