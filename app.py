import time
import math
import re
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from prophet import Prophet
import plotly.graph_objects as go

st.set_page_config(page_title="Indian Stock Market Predictor Ultimate", page_icon="📈", layout="wide")

st.title("📊 Indian Stock Market Predictor Ultimate")
st.markdown("**Forecast + Technical Analysis + Fundamentals + Watchlist + Comparison + News Sentiment + Portfolio Tracker**")


# -----------------------------
# Session State
# -----------------------------
if "watchlist" not in st.session_state:
    st.session_state.watchlist = ["RELIANCE.NS", "HDFCBANK.NS", "TCS.NS", "INFY.NS"]

if "portfolio" not in st.session_state:
    st.session_state.portfolio = [
        {"Symbol": "RELIANCE.NS", "Quantity": 10.0, "Buy Price": 2500.0},
        {"Symbol": "HDFCBANK.NS", "Quantity": 5.0, "Buy Price": 1500.0},
    ]


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

    confidence = min(95, max(35, 50 + (abs(score) * 8)))

    return {
        "label": label,
        "score": score,
        "reasons": reasons,
        "color": color,
        "confidence": confidence
    }


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


def get_support_resistance(df: pd.DataFrame) -> dict:
    recent = df.tail(60).copy()
    if recent.empty:
        return {"support": None, "resistance": None, "stop_loss": None, "breakout": None}

    current = float(recent["Close"].iloc[-1])

    supports = sorted(recent["Low"].dropna().unique().tolist())
    resistances = sorted(recent["High"].dropna().unique().tolist())

    support = None
    resistance = None

    lower_supports = [x for x in supports if x < current]
    higher_resistances = [x for x in resistances if x > current]

    if lower_supports:
        support = max(lower_supports)
    if higher_resistances:
        resistance = min(higher_resistances)

    stop_loss = None
    breakout = None

    if support is not None:
        stop_loss = round(support * 0.985, 2)
    if resistance is not None:
        breakout = round(resistance * 1.01, 2)

    return {
        "support": round(support, 2) if support is not None else None,
        "resistance": round(resistance, 2) if resistance is not None else None,
        "stop_loss": stop_loss,
        "breakout": breakout
    }


@st.cache_data(show_spinner=False, ttl=1800)
def fetch_fundamentals(symbol: str) -> dict:
    try:
        tk = yf.Ticker(symbol)
        info = tk.info if hasattr(tk, "info") else {}

        return {
            "longName": info.get("longName", ""),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "marketCap": info.get("marketCap", None),
            "trailingPE": info.get("trailingPE", None),
            "forwardPE": info.get("forwardPE", None),
            "dividendYield": info.get("dividendYield", None),
            "bookValue": info.get("bookValue", None),
            "priceToBook": info.get("priceToBook", None),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh", None),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow", None),
            "currency": info.get("currency", "INR"),
            "website": info.get("website", ""),
        }
    except Exception:
        return {}


def simple_sentiment_score(text: str) -> tuple[str, int]:
    positive_words = {
        "gain", "gains", "surge", "surges", "up", "beat", "beats", "strong", "growth", "bullish",
        "profit", "profits", "record", "expands", "expansion", "buy", "outperform", "positive",
        "rise", "rises", "jump", "jumps", "higher", "improves", "improvement"
    }
    negative_words = {
        "fall", "falls", "down", "miss", "misses", "weak", "loss", "losses", "bearish", "drop",
        "drops", "lower", "cuts", "cut", "decline", "declines", "risk", "risks", "warning",
        "lawsuit", "probe", "crash", "slump", "pressure"
    }

    text = (text or "").lower()
    words = re.findall(r"[a-zA-Z]+", text)

    pos = sum(1 for w in words if w in positive_words)
    neg = sum(1 for w in words if w in negative_words)
    score = pos - neg

    if score > 0:
        return "Positive", score
    elif score < 0:
        return "Negative", score
    return "Neutral", score


@st.cache_data(show_spinner=False, ttl=900)
def fetch_news(symbol: str) -> pd.DataFrame:
    try:
        tk = yf.Ticker(symbol)
        news_items = getattr(tk, "news", None)
        if not news_items:
            return pd.DataFrame()

        rows = []
        for item in news_items[:15]:
            title = item.get("title", "")
            publisher = item.get("publisher", "")
            link = item.get("link", "")
            provider_time = item.get("providerPublishTime", None)

            published = pd.to_datetime(provider_time, unit="s", errors="coerce") if provider_time else pd.NaT
            sentiment, score = simple_sentiment_score(title)

            rows.append({
                "Published": published,
                "Title": title,
                "Publisher": publisher,
                "Sentiment": sentiment,
                "SentimentScore": score,
                "Link": link,
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("Published", ascending=False)
        return df
    except Exception:
        return pd.DataFrame()


def format_large_number(x):
    if x is None or pd.isna(x):
        return "-"
    x = float(x)
    if x >= 1_00_00_00_000:
        return f"₹ {x / 1_00_00_00_000:.2f} Cr"
    if x >= 1_00_00_000:
        return f"₹ {x / 1_00_00_000:.2f} Cr"
    if x >= 1_00_000:
        return f"₹ {x / 1_00_000:.2f} Lakh"
    return f"₹ {x:,.2f}"


def fmt_num(x) -> str:
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return "-"


def add_to_watchlist(symbol: str):
    symbol = normalize_symbol(symbol)
    if symbol not in st.session_state.watchlist:
        st.session_state.watchlist.append(symbol)


def remove_from_watchlist(symbol: str):
    if symbol in st.session_state.watchlist:
        st.session_state.watchlist.remove(symbol)


def add_portfolio_row(symbol: str, qty: float, buy_price: float):
    symbol = normalize_symbol(symbol)
    st.session_state.portfolio.append({
        "Symbol": symbol,
        "Quantity": float(qty),
        "Buy Price": float(buy_price)
    })


def remove_portfolio_row(index: int):
    if 0 <= index < len(st.session_state.portfolio):
        st.session_state.portfolio.pop(index)


@st.cache_data(show_spinner=False, ttl=600)
def build_watchlist_snapshot(symbols: tuple) -> pd.DataFrame:
    rows = []

    for sym in symbols:
        try:
            d = fetch_stock_data(sym)
            if d.empty or len(d) < 3:
                continue

            d = add_indicators(d)
            close = float(d["Close"].iloc[-1])
            prev = float(d["Close"].iloc[-2])
            chg = close - prev
            chg_pct = (chg / prev * 100) if prev else 0.0

            sig = generate_signal(d)
            risk_name, vol = risk_level(d)

            rows.append({
                "Symbol": sym,
                "Price": round(close, 2),
                "Day Change": round(chg, 2),
                "Day Change %": round(chg_pct, 2),
                "Signal": sig["label"],
                "Confidence %": sig["confidence"],
                "Risk": risk_name,
                "RSI14": round(float(d["RSI14"].iloc[-1]), 2) if pd.notna(d["RSI14"].iloc[-1]) else None,
                "SMA20": round(float(d["SMA20"].iloc[-1]), 2) if pd.notna(d["SMA20"].iloc[-1]) else None,
            })
        except Exception:
            continue

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, ttl=600)
def build_comparison_snapshot(symbols: tuple, days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    norm_df = pd.DataFrame()

    for sym in symbols:
        try:
            d = fetch_stock_data(sym)
            if d.empty or len(d) < 50:
                continue

            d = add_indicators(d)

            close = float(d["Close"].iloc[-1])
            prev = float(d["Close"].iloc[-2])
            day_change = close - prev
            day_change_pct = (day_change / prev * 100) if prev else 0.0

            ret_30 = ((float(d["Close"].iloc[-1]) / float(d["Close"].iloc[-31])) - 1) * 100 if len(d) > 31 else np.nan
            ret_90 = ((float(d["Close"].iloc[-1]) / float(d["Close"].iloc[-91])) - 1) * 100 if len(d) > 91 else np.nan

            sig = generate_signal(d)
            risk_name, vol = risk_level(d)

            forecast_end = np.nan
            forecast_move_pct = np.nan
            try:
                _, fc = build_forecast(d, days)
                future_rows = fc[["ds", "yhat"]].tail(days).copy()
                forecast_end = float(future_rows["yhat"].iloc[-1])
                forecast_move_pct = ((forecast_end / close) - 1) * 100 if close else np.nan
            except Exception:
                pass

            rows.append({
                "Symbol": sym,
                "Price": round(close, 2),
                "Day Change %": round(day_change_pct, 2),
                "30D Return %": round(ret_30, 2) if pd.notna(ret_30) else None,
                "90D Return %": round(ret_90, 2) if pd.notna(ret_90) else None,
                "Volatility %": round(vol, 2),
                "RSI14": round(float(d["RSI14"].iloc[-1]), 2) if pd.notna(d["RSI14"].iloc[-1]) else None,
                "Signal": sig["label"],
                "Confidence %": sig["confidence"],
                "Risk": risk_name,
                "Forecast End": round(forecast_end, 2) if pd.notna(forecast_end) else None,
                "Forecast Move %": round(forecast_move_pct, 2) if pd.notna(forecast_move_pct) else None,
            })

            series = d["Close"].tail(60).copy()
            base = float(series.iloc[0]) if len(series) > 0 else 1.0
            norm_series = (series / base) * 100
            norm_df[sym] = norm_series

        except Exception:
            continue

    norm_df = norm_df.sort_index()
    return pd.DataFrame(rows), norm_df


@st.cache_data(show_spinner=False, ttl=300)
def current_price_for_symbol(symbol: str):
    try:
        d = fetch_stock_data(symbol)
        if d.empty:
            return None
        return float(d["Close"].iloc[-1])
    except Exception:
        return None


def build_portfolio_snapshot(portfolio_rows: list) -> pd.DataFrame:
    rows = []

    for row in portfolio_rows:
        try:
            sym = normalize_symbol(row["Symbol"])
            qty = float(row["Quantity"])
            buy_price = float(row["Buy Price"])

            current_price = current_price_for_symbol(sym)
            if current_price is None:
                continue

            invested = qty * buy_price
            current_value = qty * current_price
            pnl = current_value - invested
            pnl_pct = (pnl / invested * 100) if invested else 0.0

            rows.append({
                "Symbol": sym,
                "Quantity": qty,
                "Buy Price": round(buy_price, 2),
                "Current Price": round(current_price, 2),
                "Invested": round(invested, 2),
                "Current Value": round(current_value, 2),
                "P/L": round(pnl, 2),
                "P/L %": round(pnl_pct, 2),
            })
        except Exception:
            continue

    return pd.DataFrame(rows)


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
show_fundamentals = st.sidebar.checkbox("Show Fundamentals", value=True)

st.sidebar.markdown("### Comparison Mode")
cmp1 = st.sidebar.text_input("Compare Stock 1", value="RELIANCE.NS").strip().upper()
cmp2 = st.sidebar.text_input("Compare Stock 2", value="HDFCBANK.NS").strip().upper()
cmp3 = st.sidebar.text_input("Compare Stock 3", value="TCS.NS").strip().upper()

compare_symbols = []
for s in [cmp1, cmp2, cmp3]:
    if s:
        ns = normalize_symbol(s)
        if ns not in compare_symbols:
            compare_symbols.append(ns)

col_w1, col_w2 = st.sidebar.columns(2)
with col_w1:
    if st.button("Add to Watchlist"):
        add_to_watchlist(symbol)
        st.sidebar.success(f"Added {symbol}")
with col_w2:
    if st.button("Remove"):
        remove_from_watchlist(symbol)
        st.sidebar.warning(f"Removed {symbol}")

st.sidebar.markdown("### Watchlist")
if st.session_state.watchlist:
    for wl in st.session_state.watchlist:
        st.sidebar.write(f"• {wl}")
else:
    st.sidebar.write("No watchlist items")

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
        levels = get_support_resistance(data)
        fundamentals = fetch_fundamentals(symbol) if show_fundamentals else {}
        news_df = fetch_news(symbol)

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
                <span style="margin-left:16px;font-size:16px;">Confidence: <b>{signal['confidence']}%</b></span>
                <span style="margin-left:16px;font-size:16px;">Risk: <b>{risk_name}</b></span>
                <span style="margin-left:16px;font-size:16px;">Score: <b>{signal['score']}</b></span>
            </div>
            """,
            unsafe_allow_html=True
        )

        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
            ["Overview", "Technical Analysis", "Forecast", "Backtest", "Fundamentals", "Watchlist", "Comparison", "News Sentiment", "Portfolio"]
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

            if levels["support"] is not None:
                fig.add_hline(y=levels["support"], annotation_text=f"Support {levels['support']}", line_dash="dot")
            if levels["resistance"] is not None:
                fig.add_hline(y=levels["resistance"], annotation_text=f"Resistance {levels['resistance']}", line_dash="dot")

            fig.update_layout(
                title=f"{symbol} Historical Price with Moving Averages",
                xaxis_title="Date",
                yaxis_title="Price",
                xaxis_rangeslider_visible=False,
                height=650
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("🎯 Support / Resistance")
            sr1, sr2, sr3, sr4 = st.columns(4)
            sr1.metric("Nearest Support", f"₹ {fmt_num(levels['support'])}" if levels["support"] is not None else "-")
            sr2.metric("Nearest Resistance", f"₹ {fmt_num(levels['resistance'])}" if levels["resistance"] is not None else "-")
            sr3.metric("Suggested Stop Loss", f"₹ {fmt_num(levels['stop_loss'])}" if levels["stop_loss"] is not None else "-")
            sr4.metric("Breakout Zone", f"₹ {fmt_num(levels['breakout'])}" if levels["breakout"] is not None else "-")

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

                fig_bb = go.Figure()
                fig_bb.add_trace(go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Close"))
                fig_bb.add_trace(go.Scatter(x=data.index, y=data["BB_Upper"], mode="lines", name="BB Upper"))
                fig_bb.add_trace(go.Scatter(x=data.index, y=data["BB_Mid"], mode="lines", name="BB Mid"))
                fig_bb.add_trace(go.Scatter(x=data.index, y=data["BB_Lower"], mode="lines", name="BB Lower"))
                fig_bb.update_layout(title="Bollinger Bands", xaxis_title="Date", yaxis_title="Price", height=420)
                st.plotly_chart(fig_bb, use_container_width=True)
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
                fig2.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", line=dict(width=0), showlegend=False))
                fig2.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", fill="tonexty", line=dict(width=0), name="Confidence Range"))
                fig2.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Predicted"))
                fig2.update_layout(title="Historical + Forecast", xaxis_title="Date", yaxis_title="Price", height=560)
                st.plotly_chart(fig2, use_container_width=True)

                prediction_table = future_rows.round(2).rename(
                    columns={"ds": "Date", "yhat": "Predicted ₹", "yhat_lower": "Lower ₹", "yhat_upper": "Upper ₹"}
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
                    fig_bt.update_layout(title="Backtest: Actual vs Predicted", xaxis_title="Date", yaxis_title="Price", height=450)
                    st.plotly_chart(fig_bt, use_container_width=True)
                else:
                    st.warning(bt["reason"])
            else:
                st.info("Backtest is hidden from sidebar settings.")

        with tab5:
            st.subheader("🏢 Fundamentals")

            if not show_fundamentals:
                st.info("Fundamentals are hidden from sidebar settings.")
            elif not fundamentals:
                st.warning("Fundamentals not available for this symbol right now.")
            else:
                name = fundamentals.get("longName", "") or symbol
                st.markdown(f"### {name}")

                f1, f2, f3, f4 = st.columns(4)
                f1.metric("Market Cap", format_large_number(fundamentals.get("marketCap")))
                f2.metric("Trailing PE", fmt_num(fundamentals.get("trailingPE")))
                f3.metric("Forward PE", fmt_num(fundamentals.get("forwardPE")))
                f4.metric("Dividend Yield %", f"{(fundamentals.get('dividendYield') or 0) * 100:.2f}" if fundamentals.get("dividendYield") is not None else "-")

                f5, f6, f7, f8 = st.columns(4)
                f5.metric("Book Value", fmt_num(fundamentals.get("bookValue")))
                f6.metric("Price to Book", fmt_num(fundamentals.get("priceToBook")))
                f7.metric("52W High", fmt_num(fundamentals.get("fiftyTwoWeekHigh")))
                f8.metric("52W Low", fmt_num(fundamentals.get("fiftyTwoWeekLow")))

        with tab6:
            st.subheader("⭐ Watchlist Dashboard")

            if not st.session_state.watchlist:
                st.info("Your watchlist is empty.")
            else:
                with st.spinner("Building watchlist snapshot..."):
                    wl_df = build_watchlist_snapshot(tuple(st.session_state.watchlist))

                if wl_df.empty:
                    st.warning("Watchlist data could not be loaded right now.")
                else:
                    st.dataframe(wl_df, use_container_width=True)

        with tab7:
            st.subheader("⚖️ Compare 2–3 Stocks")

            if len(compare_symbols) < 2:
                st.warning("Please enter at least 2 symbols in the sidebar for comparison.")
            else:
                with st.spinner("Building comparison view..."):
                    cmp_df, norm_df = build_comparison_snapshot(tuple(compare_symbols), days)

                if cmp_df.empty:
                    st.warning("Comparison data could not be loaded.")
                else:
                    st.dataframe(cmp_df, use_container_width=True)

                    fig_cmp = go.Figure()
                    for col in norm_df.columns:
                        fig_cmp.add_trace(go.Scatter(x=norm_df.index, y=norm_df[col], mode="lines", name=col))
                    fig_cmp.update_layout(title="Normalized Performance", xaxis_title="Date", yaxis_title="Base = 100", height=500)
                    st.plotly_chart(fig_cmp, use_container_width=True)

        with tab8:
            st.subheader("📰 News Sentiment")

            if news_df.empty:
                st.warning("No recent news available for this symbol right now.")
            else:
                pos_count = int((news_df["Sentiment"] == "Positive").sum())
                neg_count = int((news_df["Sentiment"] == "Negative").sum())
                neu_count = int((news_df["Sentiment"] == "Neutral").sum())
                avg_score = float(news_df["SentimentScore"].mean()) if not news_df.empty else 0.0

                n1, n2, n3, n4 = st.columns(4)
                n1.metric("Positive Headlines", pos_count)
                n2.metric("Negative Headlines", neg_count)
                n3.metric("Neutral Headlines", neu_count)
                n4.metric("Avg Sentiment Score", f"{avg_score:.2f}")

                overall = "Positive" if avg_score > 0 else "Negative" if avg_score < 0 else "Neutral"
                st.info(f"Overall headline sentiment: {overall}")

                display_df = news_df.copy()
                display_df["Published"] = pd.to_datetime(display_df["Published"], errors="coerce")
                display_df["Published"] = display_df["Published"].dt.strftime("%Y-%m-%d %H:%M")
                st.dataframe(display_df[["Published", "Publisher", "Title", "Sentiment", "SentimentScore", "Link"]], use_container_width=True)

        with tab9:
            st.subheader("💼 Portfolio Tracker")

            with st.expander("Add Holding", expanded=False):
                pcol1, pcol2, pcol3, pcol4 = st.columns(4)
                with pcol1:
                    port_symbol = st.text_input("Portfolio Symbol", value="SBIN.NS")
                with pcol2:
                    port_qty = st.number_input("Quantity", min_value=0.0, value=1.0, step=1.0)
                with pcol3:
                    port_buy = st.number_input("Buy Price", min_value=0.0, value=100.0, step=1.0)
                with pcol4:
                    st.write("")
                    st.write("")
                    if st.button("Add Holding"):
                        add_portfolio_row(port_symbol, port_qty, port_buy)
                        st.success("Holding added.")

            if st.session_state.portfolio:
                raw_port_df = pd.DataFrame(st.session_state.portfolio)
                raw_port_df.index = range(len(raw_port_df))
                st.write("Current Holdings Input")
                st.dataframe(raw_port_df, use_container_width=True)

                remove_index = st.number_input("Remove holding index", min_value=0, max_value=max(0, len(st.session_state.portfolio) - 1), value=0, step=1)
                if st.button("Remove Holding"):
                    remove_portfolio_row(int(remove_index))
                    st.warning("Holding removed.")

                with st.spinner("Building portfolio snapshot..."):
                    portfolio_df = build_portfolio_snapshot(st.session_state.portfolio)

                if portfolio_df.empty:
                    st.warning("Portfolio data could not be built.")
                else:
                    total_invested = float(portfolio_df["Invested"].sum())
                    total_value = float(portfolio_df["Current Value"].sum())
                    total_pnl = float(portfolio_df["P/L"].sum())
                    total_pnl_pct = (total_pnl / total_invested * 100) if total_invested else 0.0

                    pf1, pf2, pf3, pf4 = st.columns(4)
                    pf1.metric("Total Invested", f"₹ {fmt_num(total_invested)}")
                    pf2.metric("Current Value", f"₹ {fmt_num(total_value)}")
                    pf3.metric("Net P/L", f"₹ {fmt_num(total_pnl)}")
                    pf4.metric("Net P/L %", f"{total_pnl_pct:.2f}%")

                    st.dataframe(portfolio_df, use_container_width=True)

                    fig_pf = go.Figure()
                    fig_pf.add_trace(go.Bar(x=portfolio_df["Symbol"], y=portfolio_df["P/L"], name="P/L"))
                    fig_pf.update_layout(title="Portfolio P/L by Holding", xaxis_title="Symbol", yaxis_title="P/L", height=420)
                    st.plotly_chart(fig_pf, use_container_width=True)

                    csv_pf = portfolio_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Portfolio CSV",
                        data=csv_pf,
                        file_name="portfolio_snapshot.csv",
                        mime="text/csv"
                    )
            else:
                st.info("No portfolio holdings yet.")

# st.caption("⚠️ This is an educational demo only. Not financial advice. Data from Yahoo Finance may be unreliable for some NSE/BSE symbols.")