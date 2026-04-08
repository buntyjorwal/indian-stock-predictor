import time
import math
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from prophet import Prophet
import plotly.graph_objects as go

st.set_page_config(page_title="Indian Stock Market Predictor Pro", page_icon="📈", layout="wide")

st.title("📊 Indian Stock Market Predictor Pro")
st.markdown("**Real-time NSE/BSE data + Forecast + Technical Analysis + Signal Engine + Backtest**")


# -----------------------------
# Helpers
# -----------------------------
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
    volume_col = find_price_column(df, "Volume")

    required = {"Open": open_col, "High": high_col, "Low": low_col, "Close": close_col}
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise ValueError(f"Required columns not found: {', '.join(missing)}. Returned columns: {list(df.columns)}")

    cleaned = pd.DataFrame(index=df.index)
    cleaned["Open"] = pd.to_numeric(df[open_col], errors="coerce")
    cleaned["High"] = pd.to_numeric(df[high_col], errors="coerce")
    cleaned["Low"] = pd.to_numeric(df[low_col], errors="coerce")
    cleaned["Close"] = pd.to_numeric(df[close_col], errors="coerce")
    cleaned["Volume"] = pd.to_numeric(df[volume_col], errors="coerce") if volume_col else np.nan

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
            if len(cleaned) >= 50:
                return cleaned

        except Exception as ex:
            last_error = ex

        time.sleep(3)

    if last_error:
        raise last_error

    return pd.DataFrame()


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()

    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI14"] = 100 - (100 / (1 + rs))

    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACDSignal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACDHist"] = df["MACD"] - df["MACDSignal"]

    rolling_mean = df["Close"].rolling(20).mean()
    rolling_std = df["Close"].rolling(20).std()
    df["BB_Mid"] = rolling_mean
    df["BB_Upper"] = rolling_mean + (2 * rolling_std)
    df["BB_Lower"] = rolling_mean - (2 * rolling_std)

    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - df["Close"].shift(1)).abs()
    tr3 = (df["Low"] - df["Close"].shift(1)).abs()
    df["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR14"] = df["TR"].rolling(14).mean()

    return df


def build_forecast(data: pd.DataFrame, days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = data[["Close"]].reset_index().copy()
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "ds", "Close": "y"})
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["ds", "y"]).copy()

    if len(df) < 50:
        raise ValueError("Not enough clean data for prediction.")

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.15
    )
    model.fit(df)

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    return df, forecast


def simple_backtest(data: pd.DataFrame, holdout_days: int = 30) -> dict:
    df = data[["Close"]].reset_index().copy()
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "ds", "Close": "y"})
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna().copy()

    if len(df) < 120:
        return {"ok": False, "reason": "Not enough data for backtest."}

    holdout_days = min(holdout_days, max(7, len(df) // 5))
    train = df.iloc[:-holdout_days].copy()
    test = df.iloc[-holdout_days:].copy()

    if len(train) < 60:
        return {"ok": False, "reason": "Training window too small for backtest."}

    try:
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.15
        )
        model.fit(train)

        future = model.make_future_dataframe(periods=holdout_days)
        fc = model.predict(future)[["ds", "yhat"]].tail(holdout_days).copy()

        merged = test.merge(fc, on="ds", how="left").dropna().copy()
        if merged.empty:
            return {"ok": False, "reason": "Backtest forecast merge failed."}

        merged["abs_err"] = (merged["y"] - merged["yhat"]).abs()
        merged["pct_err"] = merged["abs_err"] / merged["y"].replace(0, np.nan)
        merged["sq_err"] = (merged["y"] - merged["yhat"]) ** 2

        mae = float(merged["abs_err"].mean())
        rmse = float(math.sqrt(merged["sq_err"].mean()))
        mape = float((merged["pct_err"].dropna().mean()) * 100)

        return {
            "ok": True,
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "actual_pred": merged
        }
    except Exception as ex:
        return {"ok": False, "reason": str(ex)}


def generate_signal(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]

    score = 0
    reasons = []

    close = float(last["Close"])
    sma20 = float(last["SMA20"]) if pd.notna(last["SMA20"]) else None
    sma50 = float(last["SMA50"]) if pd.notna(last["SMA50"]) else None
    sma200 = float(last["SMA200"]) if pd.notna(last["SMA200"]) else None
    rsi = float(last["RSI14"]) if pd.notna(last["RSI14"]) else None
    macd = float(last["MACD"]) if pd.notna(last["MACD"]) else None
    macd_signal = float(last["MACDSignal"]) if pd.notna(last["MACDSignal"]) else None

    if sma20 is not None and close > sma20:
        score += 1
        reasons.append("Price above SMA20")
    elif sma20 is not None:
        score -= 1
        reasons.append("Price below SMA20")

    if sma50 is not None and close > sma50:
        score += 1
        reasons.append("Price above SMA50")
    elif sma50 is not None:
        score -= 1
        reasons.append("Price below SMA50")

    if sma200 is not None and close > sma200:
        score += 1
        reasons.append("Price above SMA200")
    elif sma200 is not None:
        score -= 1
        reasons.append("Price below SMA200")

    if rsi is not None:
        if rsi < 30:
            score += 1
            reasons.append("RSI oversold")
        elif rsi > 70:
            score -= 1
            reasons.append("RSI overbought")
        elif 45 <= rsi <= 60:
            score += 1
            reasons.append("RSI healthy")

    if macd is not None and macd_signal is not None:
        if macd > macd_signal:
            score += 1
            reasons.append("MACD bullish crossover")
        else:
            score -= 1
            reasons.append("MACD bearish bias")

    if score >= 4:
        label = "BUY"
        color = "green"
    elif score >= 2:
        label = "WEAK BUY"
        color = "green"
    elif score <= -4:
        label = "SELL"
        color = "red"
    elif score <= -2:
        label = "WEAK SELL"
        color = "red"
    else:
        label = "HOLD"
        color = "orange"

    return {"label": label, "score": score, "reasons": reasons, "color": color}


def risk_level(df: pd.DataFrame) -> tuple[str, float]:
    daily_ret = df["Close"].pct_change().dropna()
    if daily_ret.empty:
        return "Unknown", 0.0

    vol_annual = float(daily_ret.std() * np.sqrt(252) * 100)

    if vol_annual < 20:
        return "Low", vol_annual
    elif vol_annual < 35:
        return "Moderate", vol_annual
    else:
        return "High", vol_annual


def fmt_num(x) -> str:
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return "-"


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Stock Selection")

quick_stocks = {
    "RELIANCE": "RELIANCE.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "SBIN": "SBIN.NS",
    "ITC": "ITC.NS",
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
show_backtest = st.sidebar.checkbox("Show Backtest", value=True)
show_technical = st.sidebar.checkbox("Show Technical Indicators", value=True)

run_btn = st.sidebar.button("Fetch Data & Predict", type="primary")


# -----------------------------
# Main
# -----------------------------
if run_btn:
    with st.spinner(f"Fetching data for **{symbol}**..."):
        try:
            raw_data = fetch_stock_data(symbol)
        except Exception as ex:
            raw_data = pd.DataFrame()
            st.error(f"❌ Error while fetching data for **{symbol}**")
            st.code(str(ex))

    if raw_data.empty or len(raw_data) < 50:
        st.error(f"❌ Could not fetch enough data for **{symbol}** right now.")
        st.info("Try these symbols: RELIANCE.NS, HDFCBANK.NS, TCS.NS, INFY.NS, SBIN.NS, ITC.NS, ^NSEI")
    else:
        data = add_indicators(raw_data)

        close_series = pd.to_numeric(data["Close"], errors="coerce").dropna()
        current_price = float(close_series.iloc[-1])
        prev_price = float(close_series.iloc[-2]) if len(close_series) > 1 else current_price
        day_change = current_price - prev_price
        day_change_pct = (day_change / prev_price * 100) if prev_price else 0

        high_52w = float(data["Close"].tail(252).max()) if len(data) >= 20 else current_price
        low_52w = float(data["Close"].tail(252).min()) if len(data) >= 20 else current_price
        avg_volume = float(data["Volume"].tail(20).mean()) if data["Volume"].notna().any() else 0.0

        signal = generate_signal(data)
        risk_name, volatility = risk_level(data)

        st.success(f"✅ Data loaded for **{symbol}**")

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Current Price", f"₹ {fmt_num(current_price)}", f"{day_change:,.2f}")
        m2.metric("Daily Change %", f"{day_change_pct:,.2f}%")
        m3.metric("52W High", f"₹ {fmt_num(high_52w)}")
        m4.metric("52W Low", f"₹ {fmt_num(low_52w)}")
        m5.metric("20D Avg Volume", f"{avg_volume:,.0f}" if avg_volume else "-")
        m6.metric("Volatility", f"{volatility:.2f}%")

        st.markdown(
            f"""
            <div style="padding:12px 16px;border-radius:12px;background:#111827;margin:10px 0 18px 0;">
                <span style="font-size:18px;font-weight:700;">Signal:</span>
                <span style="font-size:20px;font-weight:800;color:{signal['color']};margin-left:8px;">{signal['label']}</span>
                <span style="margin-left:16px;font-size:16px;">Risk: <b>{risk_name}</b></span>
                <span style="margin-left:16px;font-size:16px;">Score: <b>{signal['score']}</b></span>
            </div>
            """,
            unsafe_allow_html=True
        )

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Overview", "Technical Analysis", "Forecast", "Backtest", "Data Table"]
        )

        with tab1:
            st.subheader(f"📉 {symbol} Price Overview")

            fig = go.Figure()
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data["Open"],
                    high=data["High"],
                    low=data["Low"],
                    close=data["Close"],
                    name="Price"
                )
            )
            fig.add_trace(go.Scatter(x=data.index, y=data["SMA20"], mode="lines", name="SMA20"))
            fig.add_trace(go.Scatter(x=data.index, y=data["SMA50"], mode="lines", name="SMA50"))
            fig.add_trace(go.Scatter(x=data.index, y=data["SMA200"], mode="lines", name="SMA200"))

            fig.update_layout(
                title=f"{symbol} Historical Price with Moving Averages",
                xaxis_title="Date",
                yaxis_title="Price",
                xaxis_rangeslider_visible=False,
                height=650
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("🧾 Summary")
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**Current Price:** ₹ {fmt_num(current_price)}")
                st.write(f"**52 Week High:** ₹ {fmt_num(high_52w)}")
                st.write(f"**52 Week Low:** ₹ {fmt_num(low_52w)}")
                st.write(f"**Volatility:** {volatility:.2f}% ({risk_name})")
            with c2:
                st.write(f"**Signal:** {signal['label']}")
                st.write("**Why:**")
                for reason in signal["reasons"]:
                    st.write(f"- {reason}")

        with tab2:
            if show_technical:
                st.subheader("📊 RSI & MACD")

                t1, t2 = st.columns(2)

                with t1:
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(x=data.index, y=data["RSI14"], mode="lines", name="RSI14"))
                    fig_rsi.add_hline(y=70)
                    fig_rsi.add_hline(y=30)
                    fig_rsi.update_layout(title="RSI (14)", xaxis_title="Date", yaxis_title="RSI", height=350)
                    st.plotly_chart(fig_rsi, use_container_width=True)

                with t2:
                    fig_macd = go.Figure()
                    fig_macd.add_trace(go.Scatter(x=data.index, y=data["MACD"], mode="lines", name="MACD"))
                    fig_macd.add_trace(go.Scatter(x=data.index, y=data["MACDSignal"], mode="lines", name="Signal"))
                    fig_macd.update_layout(title="MACD", xaxis_title="Date", yaxis_title="Value", height=350)
                    st.plotly_chart(fig_macd, use_container_width=True)

                st.subheader("📈 Bollinger Bands")
                fig_bb = go.Figure()
                fig_bb.add_trace(go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Close"))
                fig_bb.add_trace(go.Scatter(x=data.index, y=data["BB_Upper"], mode="lines", name="BB Upper"))
                fig_bb.add_trace(go.Scatter(x=data.index, y=data["BB_Mid"], mode="lines", name="BB Mid"))
                fig_bb.add_trace(go.Scatter(x=data.index, y=data["BB_Lower"], mode="lines", name="BB Lower"))
                fig_bb.update_layout(title="Bollinger Bands", xaxis_title="Date", yaxis_title="Price", height=420)
                st.plotly_chart(fig_bb, use_container_width=True)

                last = data.iloc[-1]
                st.subheader("Technical Snapshot")
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("RSI 14", fmt_num(last["RSI14"]))
                s2.metric("MACD", fmt_num(last["MACD"]))
                s3.metric("Signal Line", fmt_num(last["MACDSignal"]))
                s4.metric("ATR 14", fmt_num(last["ATR14"]))
            else:
                st.info("Technical indicators are hidden from sidebar settings.")

        with tab3:
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

                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Predicted End Price", f"₹ {fmt_num(final_pred)}")
                p2.metric("Expected Move", f"₹ {expected_move:,.2f}", f"{expected_move_pct:,.2f}%")
                p3.metric("Range Low", f"₹ {fmt_num(final_lower)}")
                p4.metric("Range High", f"₹ {fmt_num(final_upper)}")

                st.info(f"Prediction Status: Complete ✅ | Trend Outlook: {trend}")

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=hist_df["ds"], y=hist_df["y"], mode="lines", name="Historical"))

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
                fig2.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Predicted"))

                fig2.update_layout(
                    title="Historical + Forecast with Confidence Range",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    height=560
                )
                st.plotly_chart(fig2, use_container_width=True)

                st.subheader("📋 Next Days Forecast Table")
                prediction_table = (
                    future_rows.round(2).rename(
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

        with tab4:
            st.subheader("🧪 Backtest Accuracy")

            if show_backtest:
                bt = simple_backtest(data, holdout_days=30)

                if bt["ok"]:
                    b1, b2, b3 = st.columns(3)
                    b1.metric("MAE", f"{bt['mae']:.2f}")
                    b2.metric("RMSE", f"{bt['rmse']:.2f}")
                    b3.metric("MAPE", f"{bt['mape']:.2f}%")

                    backtest_df = bt["actual_pred"].copy()

                    fig_bt = go.Figure()
                    fig_bt.add_trace(go.Scatter(x=backtest_df["ds"], y=backtest_df["y"], mode="lines", name="Actual"))
                    fig_bt.add_trace(go.Scatter(x=backtest_df["ds"], y=backtest_df["yhat"], mode="lines", name="Predicted"))
                    fig_bt.update_layout(
                        title="Backtest: Actual vs Predicted",
                        xaxis_title="Date",
                        yaxis_title="Price",
                        height=450
                    )
                    st.plotly_chart(fig_bt, use_container_width=True)

                    if bt["mape"] <= 2:
                        st.success("Backtest quality looks strong for recent data.")
                    elif bt["mape"] <= 5:
                        st.info("Backtest quality is reasonable for a simple model.")
                    else:
                        st.warning("Backtest error is high. Use forecast carefully.")
                else:
                    st.warning("Backtest unavailable.")
                    st.write(bt["reason"])
            else:
                st.info("Backtest is hidden from sidebar settings.")

        with tab5:
            st.subheader("📄 Latest Data")
            out_df = data.tail(120).copy().reset_index()
            st.dataframe(out_df, use_container_width=True)

            csv_data = out_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Latest Data CSV",
                data=csv_data,
                file_name=f"{symbol.replace('^', '')}_latest_data.csv",
                mime="text/csv"
            )

st.caption("⚠️ This is an educational demo only. Not financial advice. Data from Yahoo Finance may be unreliable for some NSE/BSE symbols.")