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
    if matches:
        return matches[0]

    return ""


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

    cleaned = cleaned.dropna(subset=["Close"]).copy()
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


# Sidebar
st.sidebar.header("Stock Selection")
symbol_input = st.sidebar.text_input("Enter Stock Symbol", value="RELIANCE.NS").strip().upper()
symbol = normalize_symbol(symbol_input)

days = st.sidebar.slider("Days to Predict", min_value=7, max_value=60, value=15)

if st.sidebar.button("Fetch Data & Predict", type="primary"):
    with st.spinner(f"Fetching data for **{symbol}**..."):
        try:
            data = fetch_stock_data(symbol)
        except Exception as ex:
            data = pd.DataFrame()
            st.error(f"❌ Error while fetching data for **{symbol}**")
            st.code(str(ex))

        if data.empty or len(data) < 20:
            st.error(f"❌ Could not fetch enough data for **{symbol}** right now.")
            st.info("**Why?** Yahoo Finance sometimes returns empty or unstable data for Indian symbols on cloud servers.")
            st.info("**Try these exact symbols:** RELIANCE.NS, HDFCBANK.NS, TCS.NS, INFY.NS, SBIN.NS, ^NSEI")
            st.info("💡 Refresh after 1-2 minutes or try locally with `streamlit run app.py`")
        else:
            close_series = pd.to_numeric(data["Close"], errors="coerce").dropna()

            if close_series.empty:
                st.error("❌ Close price data is empty after cleaning.")
                st.write("Returned columns:", list(data.columns))
                st.stop()

            current_price = float(close_series.iloc[-1])

            st.success(f"✅ Data loaded for **{symbol}**")
            st.metric("Current Price", f"₹ {current_price:,.2f}")

            # Candlestick Chart
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
                title=f"{symbol} Price History",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig, use_container_width=True)

            # Prophet Forecast
            st.subheader(f"📈 Prediction for Next {days} Days")

            df = data[["Close"]].reset_index().copy()

            date_col = df.columns[0]
            df = df.rename(columns={date_col: "ds", "Close": "y"})
            df["ds"] = pd.to_datetime(df["ds"])
            df["y"] = pd.to_numeric(df["y"], errors="coerce")
            df = df.dropna(subset=["ds", "y"]).copy()

            if len(df) < 20:
                st.error("❌ Not enough clean data available for prediction.")
                st.stop()

            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True
            )
            model.fit(df)

            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)

            fig2 = go.Figure()
            fig2.add_trace(
                go.Scatter(
                    x=df["ds"],
                    y=df["y"],
                    name="Historical",
                    mode="lines"
                )
            )
            fig2.add_trace(
                go.Scatter(
                    x=forecast["ds"],
                    y=forecast["yhat"],
                    name="Predicted",
                    mode="lines"
                )
            )
            fig2.update_layout(
                title="Forecast",
                xaxis_title="Date",
                yaxis_title="Price (₹)"
            )
            st.plotly_chart(fig2, use_container_width=True)

            prediction_table = (
                forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
                .tail(days)
                .copy()
                .round(2)
                .rename(
                    columns={
                        "ds": "Date",
                        "yhat": "Predicted ₹",
                        "yhat_lower": "Lower ₹",
                        "yhat_upper": "Upper ₹",
                    }
                )
            )

            st.dataframe(prediction_table, use_container_width=True)

st.caption("⚠️ This is an educational demo only. Not financial advice. Data from Yahoo Finance may be unreliable for some NSE/BSE symbols.")