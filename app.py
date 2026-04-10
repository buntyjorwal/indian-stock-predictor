import math
import re
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from prophet import Prophet

try:
    from nselib import capital_market
except Exception:
    capital_market = None


st.set_page_config(page_title="Indian Stock Market Predictor Ultimate", page_icon="📈", layout="wide")

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

if "last_fetch_source" not in st.session_state:
    st.session_state.last_fetch_source = "-"

if "last_fetch_note" not in st.session_state:
    st.session_state.last_fetch_note = ""


# -----------------------------
# UI helpers
# -----------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
            .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
            .app-card {
                background: linear-gradient(135deg, #0f172a 0%, #111827 100%);
                padding: 18px 20px;
                border-radius: 18px;
                border: 1px solid rgba(255,255,255,0.08);
                margin-bottom: 14px;
            }
            .app-card h3, .app-card p { margin: 0; }
            .ai-box {
                background: #f8fafc;
                color : #333333;
                border: 1px solid #e2e8f0;
                padding: 16px;
                border-radius: 16px;
            }
            .disclaimer-box {
                border-left: 6px solid #f59e0b;
                background: #fff7ed;
                padding: 14px 16px;
                color : #333333;
                border-radius: 12px;
                margin: 10px 0 16px 0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def top_banner() -> None:
    st.markdown(
        """
        <div class="app-card">
            <h2>📊 Indian Stock Market Predictor Ultimate</h2>
            <p style="margin-top:6px;opacity:0.9;">
                Forecast + Technical Analysis + Fundamentals + Watchlist + Comparison + News Sentiment + Portfolio Tracker
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# General helpers
# -----------------------------
def normalize_symbol(symbol_input: str) -> str:
    s = (symbol_input or "").strip().upper()
    if s in ["NIFTY", "NSEI", "^NSEI"]:
        return "^NSEI"
    if not s.endswith((".NS", ".BO")) and not s.startswith("^"):
        return s + ".NS"
    return s


def nse_symbol(symbol: str) -> str:
    s = normalize_symbol(symbol)
    if s.startswith("^"):
        return s.replace("^", "")
    if s.endswith(".NS"):
        return s[:-3]
    if s.endswith(".BO"):
        return s[:-3]
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

    cleaned = pd.DataFrame(index=pd.to_datetime(df.index))
    cleaned["Open"] = pd.to_numeric(df[open_col], errors="coerce")
    cleaned["High"] = pd.to_numeric(df[high_col], errors="coerce")
    cleaned["Low"] = pd.to_numeric(df[low_col], errors="coerce")
    cleaned["Close"] = pd.to_numeric(df[close_col], errors="coerce")
    cleaned["Volume"] = pd.to_numeric(df[volume_col], errors="coerce") if volume_col else np.nan

    cleaned = cleaned.dropna(subset=["Open", "High", "Low", "Close"]).sort_index().copy()
    return cleaned


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


# -----------------------------
# Hybrid data fetching
# -----------------------------
def fetch_from_yfinance(symbol: str) -> pd.DataFrame:
    raw = yf.download(
        symbol,
        period="2y",
        interval="1d",
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()
    return clean_market_data(raw)


def _try_nselib_call(func_name: str, symbol: str):
    if capital_market is None or not hasattr(capital_market, func_name):
        return None

    func = getattr(capital_market, func_name)
    to_dt = date.today()
    from_dt = to_dt - timedelta(days=800)
    symbol_nse = nse_symbol(symbol)

    call_variants = [
        {"symbol": symbol_nse, "from_date": from_dt, "to_date": to_dt},
        {"symbol": symbol_nse, "from_date": from_dt.strftime("%d-%m-%Y"), "to_date": to_dt.strftime("%d-%m-%Y")},
        {"symbol": symbol_nse, "from_date": from_dt.strftime("%Y-%m-%d"), "to_date": to_dt.strftime("%Y-%m-%d")},
        {"symbol": symbol_nse, "start_date": from_dt, "end_date": to_dt},
        {"symbol": symbol_nse, "start_date": from_dt.strftime("%d-%m-%Y"), "end_date": to_dt.strftime("%d-%m-%Y")},
        {"symbol": symbol_nse},
        {"security": symbol_nse, "from_date": from_dt.strftime("%d-%m-%Y"), "to_date": to_dt.strftime("%d-%m-%Y")},
    ]

    for kwargs in call_variants:
        try:
            data = func(**kwargs)
            if data is not None:
                return data
        except Exception:
            continue
    return None


def fetch_from_nselib(symbol: str) -> pd.DataFrame:
    if capital_market is None:
        return pd.DataFrame()
    if symbol.startswith("^"):
        return pd.DataFrame()

    func_candidates = [
        "price_volume_and_deliverable_position_data",
        "equity_price_volume_data",
        "bhav_copy_equities",
    ]

    raw = None
    for func_name in func_candidates:
        raw = _try_nselib_call(func_name, symbol)
        if raw is not None:
            break

    if raw is None:
        return pd.DataFrame()

    if isinstance(raw, list):
        df = pd.DataFrame(raw)
    elif isinstance(raw, pd.DataFrame):
        df = raw.copy()
    else:
        try:
            df = pd.DataFrame(raw)
        except Exception:
            return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    rename_map = {}
    for c in df.columns:
        c_low = str(c).strip().lower()
        if c_low in {"date", "mdate", "timestamp", "tradingdate", "trade_date", "bhavdate"}:
            rename_map[c] = "Date"
        elif c_low in {"open", "open price", "open_price", "prev open"}:
            rename_map[c] = "Open"
        elif c_low in {"high", "high price", "high_price"}:
            rename_map[c] = "High"
        elif c_low in {"low", "low price", "low_price"}:
            rename_map[c] = "Low"
        elif c_low in {"close", "close price", "close_price", "ltp", "last", "last_price"}:
            rename_map[c] = "Close"
        elif c_low in {"volume", "totaltradedvolume", "ttl_trd_qnty", "tradedqty", "traded_qty"}:
            rename_map[c] = "Volume"

    df = df.rename(columns=rename_map)
    if "Date" not in df.columns:
        for candidate in df.columns:
            if "date" in str(candidate).lower():
                df = df.rename(columns={candidate: "Date"})
                break

    needed = {"Date", "Open", "High", "Low", "Close"}
    if not needed.issubset(set(df.columns)):
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Date"]).copy()
    df = df.set_index("Date")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "High", "Low", "Close"]).sort_index()
    return df


MIN_PRICE_ROWS = 3
MIN_INDICATOR_ROWS = 20
PREDICTION_MIN_ROWS = 50
BACKTEST_MIN_ROWS = 120
FORECAST_MIN_CLIP = 0.01
FORECAST_CAP_SIGMA = 2.0
FORECAST_CAP_MIN_PCT = 0.08
FORECAST_CAP_MAX_PCT = 0.35


@st.cache_data(show_spinner=False, ttl=900)
def fetch_stock_data(symbol: str) -> tuple[pd.DataFrame, str, str]:
    last_error = None

    symbols_to_try = [symbol]
    if symbol.endswith(".NS"):
        symbols_to_try.append(symbol.replace(".NS", ".BO"))
    elif symbol.endswith(".BO"):
        symbols_to_try.append(symbol.replace(".BO", ".NS"))

    best_df = pd.DataFrame()
    best_source = "none"
    best_note = "No market data returned from available sources."

    for sym in symbols_to_try:
        for _ in range(2):
            try:
                yf_df = fetch_from_yfinance(sym)
                if len(yf_df) > len(best_df):
                    best_df = yf_df.copy()
                    best_source = "yfinance"
                    best_note = f"Primary source loaded for {sym}."
                if len(yf_df) >= MIN_PRICE_ROWS:
                    return yf_df, "yfinance", f"Primary source loaded successfully for {sym}."
            except Exception as ex:
                last_error = ex
            time.sleep(1)

        try:
            nse_df = fetch_from_nselib(sym)
            if len(nse_df) > len(best_df):
                best_df = nse_df.copy()
                best_source = "nselib"
                best_note = f"Fallback source loaded for {sym}."
            if len(nse_df) >= MIN_PRICE_ROWS:
                return nse_df, "nselib", f"Fallback source used for {sym} because yfinance was unavailable or incomplete."
        except Exception as ex:
            last_error = ex

    if not best_df.empty:
        return best_df, best_source, best_note

    if last_error:
        raise last_error
    return pd.DataFrame(), "none", "No market data returned from available sources."


# -----------------------------
# Analytics
# -----------------------------
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


def sanitize_forecast(forecast: pd.DataFrame, hist_df: pd.DataFrame, days: int) -> tuple[pd.DataFrame, dict]:
    fc = forecast.copy()
    hist = hist_df.copy()

    if hist.empty:
        meta = {
            "is_capped": False,
            "cap_pct": None,
            "floor_price": None,
            "ceiling_price": None,
            "annual_vol_pct": None,
        }
        return fc, meta

    current_price = float(hist["y"].iloc[-1])
    daily_ret = hist["y"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    ann_vol = float(daily_ret.std() * np.sqrt(252)) if not daily_ret.empty else 0.0
    horizon_vol = ann_vol * np.sqrt(max(days, 1) / 252.0)
    cap_pct = min(FORECAST_CAP_MAX_PCT, max(FORECAST_CAP_MIN_PCT, horizon_vol * FORECAST_CAP_SIGMA))

    floor_price = max(FORECAST_MIN_CLIP, current_price * (1 - cap_pct))
    ceiling_price = max(floor_price, current_price * (1 + cap_pct))

    future_mask = fc["ds"] > hist["ds"].max()
    if future_mask.any():
        pre_clip = fc.loc[future_mask, ["yhat", "yhat_lower", "yhat_upper"]].copy()
        fc.loc[future_mask, "yhat"] = fc.loc[future_mask, "yhat"].clip(lower=floor_price, upper=ceiling_price)
        fc.loc[future_mask, "yhat_lower"] = fc.loc[future_mask, "yhat_lower"].clip(lower=floor_price, upper=ceiling_price)
        fc.loc[future_mask, "yhat_upper"] = fc.loc[future_mask, "yhat_upper"].clip(lower=floor_price, upper=ceiling_price)
        fc.loc[future_mask, "yhat_lower"] = np.minimum(fc.loc[future_mask, "yhat_lower"], fc.loc[future_mask, "yhat"])
        fc.loc[future_mask, "yhat_upper"] = np.maximum(fc.loc[future_mask, "yhat_upper"], fc.loc[future_mask, "yhat"])
        is_capped = not pre_clip.round(6).equals(fc.loc[future_mask, ["yhat", "yhat_lower", "yhat_upper"]].round(6))
    else:
        is_capped = False

    meta = {
        "is_capped": is_capped,
        "cap_pct": cap_pct * 100,
        "floor_price": floor_price,
        "ceiling_price": ceiling_price,
        "annual_vol_pct": ann_vol * 100,
    }
    return fc, meta


def describe_forecast_reliability(current_price: float, final_pred: float, cap_meta: dict) -> tuple[str, str]:
    if current_price <= 0:
        return "Unknown", "Forecast reliability could not be assessed."

    move_pct = abs((final_pred - current_price) / current_price) * 100
    if cap_meta.get("is_capped"):
        return "Low", "Forecast has been capped to a realistic range because the raw model output looked too extreme."
    if move_pct >= 20:
        return "Low", "Forecast implies a very large move, so treat it as a direction warning rather than an exact target."
    if move_pct >= 10:
        return "Moderate", "Forecast is usable as a rough directional estimate, but exact price targeting may be noisy."
    return "Moderate-High", "Forecast range looks comparatively stable for the selected horizon."


def build_forecast(data: pd.DataFrame, days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = data[["Close"]].reset_index().copy()
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "ds", "Close": "y"})
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["ds", "y"]).copy()
    df = df[df["y"] > 0].copy()

    if len(df) < PREDICTION_MIN_ROWS:
        raise ValueError("Not enough clean data for prediction.")

    log_df = df.copy()
    log_df["y"] = np.log(log_df["y"])

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.15,
    )
    model.fit(log_df)

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        forecast[col] = np.exp(forecast[col]).clip(lower=FORECAST_MIN_CLIP)

    hist_df = df[["ds", "y"]].copy()
    forecast, cap_meta = sanitize_forecast(forecast, hist_df, days)
    forecast.attrs["cap_meta"] = cap_meta
    return hist_df, forecast


def sanitize_single_prediction(pred_value: float, current_price: float, hist_prices: pd.Series) -> float:
    try:
        pred_value = float(pred_value)
        current_price = float(current_price)
    except Exception:
        return max(FORECAST_MIN_CLIP, current_price)

    daily_ret = hist_prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    ann_vol = float(daily_ret.std() * np.sqrt(252)) if not daily_ret.empty else 0.0
    horizon_vol = ann_vol * np.sqrt(1 / 252.0)
    cap_pct = min(FORECAST_CAP_MAX_PCT, max(FORECAST_CAP_MIN_PCT, horizon_vol * FORECAST_CAP_SIGMA))

    floor_price = max(FORECAST_MIN_CLIP, current_price * (1 - cap_pct))
    ceiling_price = max(floor_price, current_price * (1 + cap_pct))
    return float(np.clip(pred_value, floor_price, ceiling_price))


def classify_prediction_result(pred_price: float, actual_price: float, prev_close: float) -> tuple[str, str]:
    pred_move = float(pred_price) - float(prev_close)
    actual_move = float(actual_price) - float(prev_close)

    direction_match = (
        (abs(pred_move) < 1e-9 and abs(actual_move) < 1e-9)
        or (pred_move > 0 and actual_move > 0)
        or (pred_move < 0 and actual_move < 0)
    )
    abs_error_pct = (abs(actual_price - pred_price) / actual_price * 100) if actual_price else np.nan

    if direction_match and pd.notna(abs_error_pct) and abs_error_pct <= 1.5:
        return "Strong True", "green"
    if direction_match and pd.notna(abs_error_pct) and abs_error_pct <= 4.0:
        return "Near True", "green"
    return "False", "red"


def build_past_prediction_review(data: pd.DataFrame, days: int) -> pd.DataFrame:
    df = data[["Close"]].reset_index().copy()
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "ds", "Close": "y"})
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["ds", "y"]).copy()
    df = df[df["y"] > 0].copy().reset_index(drop=True)

    if len(df) < max(PREDICTION_MIN_ROWS + 5, days + 30):
        return pd.DataFrame()

    rows = []
    review_days = min(days, len(df) - PREDICTION_MIN_ROWS - 1)

    for step_back in range(review_days, 0, -1):
        target_idx = len(df) - step_back
        if target_idx <= 1:
            continue

        train = df.iloc[:target_idx].copy()
        target_row = df.iloc[target_idx].copy()
        prev_row = df.iloc[target_idx - 1].copy()

        if len(train) < PREDICTION_MIN_ROWS:
            continue

        try:
            log_train = train.copy()
            log_train["y"] = np.log(log_train["y"])

            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.15,
            )
            model.fit(log_train)

            future = model.make_future_dataframe(periods=1)
            fc = model.predict(future)

            pred_price = float(np.exp(fc["yhat"].iloc[-1]))
            pred_low = float(np.exp(fc["yhat_lower"].iloc[-1]))
            pred_high = float(np.exp(fc["yhat_upper"].iloc[-1]))

            pred_price = sanitize_single_prediction(pred_price, float(prev_row["y"]), train["y"])
            pred_low = sanitize_single_prediction(pred_low, float(prev_row["y"]), train["y"])
            pred_high = sanitize_single_prediction(pred_high, float(prev_row["y"]), train["y"])

            pred_low = min(pred_low, pred_price)
            pred_high = max(pred_high, pred_price)

            actual_price = float(target_row["y"])
            prev_close = float(prev_row["y"])
            pred_move_pct = ((pred_price - prev_close) / prev_close * 100) if prev_close else np.nan
            actual_move_pct = ((actual_price - prev_close) / prev_close * 100) if prev_close else np.nan
            abs_error_pct = (abs(actual_price - pred_price) / actual_price * 100) if actual_price else np.nan
            result, result_color = classify_prediction_result(pred_price, actual_price, prev_close)

            rows.append(
                {
                    "Date": target_row["ds"],
                    "Previous Close ₹": prev_close,
                    "Predicted ₹": pred_price,
                    "Lower ₹": pred_low,
                    "Upper ₹": pred_high,
                    "Actual ₹": actual_price,
                    "Predicted Move %": pred_move_pct,
                    "Actual Move %": actual_move_pct,
                    "Error %": abs_error_pct,
                    "Direction Match": "Yes" if ((pred_move_pct == 0 and actual_move_pct == 0) or (pred_move_pct > 0 and actual_move_pct > 0) or (pred_move_pct < 0 and actual_move_pct < 0)) else "No",
                    "Result": result,
                    "ResultColor": result_color,
                    "Type": "Past Review",
                }
            )
        except Exception:
            continue

    return pd.DataFrame(rows)


def style_prediction_review(df: pd.DataFrame):
    if df.empty:
        return df

    def color_result(val):
        key = str(val).strip().lower()
        if key == "strong true":
            return "background-color:#bbf7d0;color:#14532d;font-weight:700;"
        if key == "near true":
            return "background-color:#dcfce7;color:#166534;font-weight:700;"
        if key == "false":
            return "background-color:#fee2e2;color:#991b1b;font-weight:700;"
        return ""

    def color_error(val):
        try:
            val = float(val)
        except Exception:
            return ""
        if val <= 1.5:
            return "background-color:#bbf7d0;color:#14532d;font-weight:700;"
        if val <= 4.0:
            return "background-color:#fef3c7;color:#92400e;font-weight:700;"
        return "background-color:#fee2e2;color:#991b1b;font-weight:700;"

    return (
        df.style
        .map(color_result, subset=["Result"])
        .map(color_error, subset=["Error %"])
        .format(
            {
                "Previous Close ₹": "{:,.2f}",
                "Predicted ₹": "{:,.2f}",
                "Lower ₹": "{:,.2f}",
                "Upper ₹": "{:,.2f}",
                "Actual ₹": "{:,.2f}",
                "Predicted Move %": "{:,.2f}%",
                "Actual Move %": "{:,.2f}%",
                "Error %": "{:,.2f}%",
            },
            na_rep="-",
        )
    )

def simple_backtest(data: pd.DataFrame, holdout_days: int = 30) -> dict:
    df = data[["Close"]].reset_index().copy()
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "ds", "Close": "y"})
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna().copy()

    if len(df) < BACKTEST_MIN_ROWS:
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
            changepoint_prior_scale=0.15,
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

        return {"ok": True, "mae": mae, "rmse": rmse, "mape": mape, "actual_pred": merged}
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
        label, color = "BUY", "green"
    elif score >= 2:
        label, color = "WEAK BUY", "green"
    elif score <= -4:
        label, color = "SELL", "red"
    elif score <= -2:
        label, color = "WEAK SELL", "red"
    else:
        label, color = "HOLD", "orange"

    confidence = min(95, max(35, 50 + (abs(score) * 8)))
    return {"label": label, "score": score, "reasons": reasons, "color": color, "confidence": confidence}


def risk_level(df: pd.DataFrame) -> tuple[str, float]:
    daily_ret = df["Close"].pct_change().dropna()
    if daily_ret.empty:
        return "Unknown", 0.0
    vol_annual = float(daily_ret.std() * np.sqrt(252) * 100)
    if vol_annual < 20:
        return "Low", vol_annual
    if vol_annual < 35:
        return "Moderate", vol_annual
    return "High", vol_annual


def get_support_resistance(df: pd.DataFrame) -> dict:
    recent = df.tail(60).copy()
    if recent.empty:
        return {"support": None, "resistance": None, "stop_loss": None, "breakout": None}

    current = float(recent["Close"].iloc[-1])
    supports = sorted(recent["Low"].dropna().unique().tolist())
    resistances = sorted(recent["High"].dropna().unique().tolist())

    lower_supports = [x for x in supports if x < current]
    higher_resistances = [x for x in resistances if x > current]

    support = max(lower_supports) if lower_supports else None
    resistance = min(higher_resistances) if higher_resistances else None

    stop_loss = round(support * 0.985, 2) if support is not None else None
    breakout = round(resistance * 1.01, 2) if resistance is not None else None

    return {
        "support": round(support, 2) if support is not None else None,
        "resistance": round(resistance, 2) if resistance is not None else None,
        "stop_loss": stop_loss,
        "breakout": breakout,
    }


def build_ai_insight(symbol: str, current_price: float, signal: dict, forecast_tail: pd.DataFrame | None, risk_name: str, volatility: float, levels: dict, cap_meta: dict | None = None) -> str:
    move_text = "Forecast currently unavailable."
    reliability_text = ""
    if forecast_tail is not None and not forecast_tail.empty:
        final_pred = float(forecast_tail["yhat"].iloc[-1])
        if final_pred > 0:
            delta = final_pred - current_price
            delta_pct = (delta / current_price * 100) if current_price else 0
            direction = "upside" if delta >= 0 else "downside"
            move_text = f"Model forecast suggests {abs(delta_pct):.2f}% {direction} over the selected horizon."
            if cap_meta:
                reliability, reliability_note = describe_forecast_reliability(current_price, final_pred, cap_meta)
                reliability_text = f" Forecast reliability is {reliability}. {reliability_note}"
                if cap_meta.get("is_capped"):
                    move_text += f" A realistic forecast band was applied between ₹ {fmt_num(cap_meta.get('floor_price'))} and ₹ {fmt_num(cap_meta.get('ceiling_price'))}."
        else:
            move_text = "Forecast was suppressed because the raw output was not price-valid."

    reasons = ", ".join(signal["reasons"][:4]) if signal["reasons"] else "limited technical confirmation"
    support_text = f"Nearest support is ₹ {fmt_num(levels['support'])}" if levels["support"] is not None else "Support level is not clearly identified"
    resistance_text = f"nearest resistance is ₹ {fmt_num(levels['resistance'])}" if levels["resistance"] is not None else "resistance level is not clearly identified"

    return (
        f"For {symbol}, the technical signal is {signal['label']} with {signal['confidence']}% confidence. "
        f"Key drivers are: {reasons}. {move_text}{reliability_text} Annualized volatility is {volatility:.2f}%, so risk is classified as {risk_name}. "
        f"{support_text}, and {resistance_text}. This is a model-assisted summary, not trading advice."
    )


# -----------------------------
# External info blocks
# -----------------------------
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
        "rise", "rises", "jump", "jumps", "higher", "improves", "improvement",
    }
    negative_words = {
        "fall", "falls", "down", "miss", "misses", "weak", "loss", "losses", "bearish", "drop",
        "drops", "lower", "cuts", "cut", "decline", "declines", "risk", "risks", "warning",
        "lawsuit", "probe", "crash", "slump", "pressure",
    }

    words = re.findall(r"[a-zA-Z]+", (text or "").lower())
    pos = sum(1 for w in words if w in positive_words)
    neg = sum(1 for w in words if w in negative_words)
    score = pos - neg
    if score > 0:
        return "Positive", score
    if score < 0:
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


# -----------------------------
# Watchlist / comparison / portfolio
# -----------------------------
def add_to_watchlist(symbol: str):
    symbol = normalize_symbol(symbol)
    if symbol not in st.session_state.watchlist:
        st.session_state.watchlist.append(symbol)


def remove_from_watchlist(symbol: str):
    if symbol in st.session_state.watchlist:
        st.session_state.watchlist.remove(symbol)


def add_portfolio_row(symbol: str, qty: float, buy_price: float):
    symbol = normalize_symbol(symbol)
    st.session_state.portfolio.append({"Symbol": symbol, "Quantity": float(qty), "Buy Price": float(buy_price)})


def remove_portfolio_row(index: int):
    if 0 <= index < len(st.session_state.portfolio):
        st.session_state.portfolio.pop(index)


@st.cache_data(show_spinner=False, ttl=600)
def build_watchlist_snapshot(symbols: tuple) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        try:
            d, _, _ = fetch_stock_data(sym)
            if d.empty or len(d) < 3:
                continue
            d = add_indicators(d)
            close = float(d["Close"].iloc[-1])
            prev = float(d["Close"].iloc[-2])
            chg = close - prev
            chg_pct = (chg / prev * 100) if prev else 0.0
            sig = generate_signal(d)
            risk_name, _ = risk_level(d)
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
            d, _, _ = fetch_stock_data(sym)
            if d.empty or len(d) < MIN_PRICE_ROWS:
                continue
            d = add_indicators(d)
            close = float(d["Close"].iloc[-1])
            prev = float(d["Close"].iloc[-2])
            day_change_pct = ((close - prev) / prev * 100) if prev else 0.0
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
            norm_df[sym] = (series / base) * 100
        except Exception:
            continue
    return pd.DataFrame(rows), norm_df.sort_index()


@st.cache_data(show_spinner=False, ttl=300)
def current_price_for_symbol(symbol: str):
    try:
        d, _, _ = fetch_stock_data(symbol)
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
inject_css()
top_banner()

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

selected_quick = st.sidebar.selectbox("Quick Select", ["Custom"] + list(quick_stocks.keys()), index=0)
default_symbol = quick_stocks[selected_quick] if selected_quick != "Custom" else "RELIANCE.NS"

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

c1, c2 = st.sidebar.columns(2)
with c1:
    run_btn = st.button("Fetch Data & Predict", type="primary", use_container_width=True)
with c2:
    refresh_btn = st.button("Refresh All Data", use_container_width=True)

if refresh_btn:
    st.cache_data.clear()
    st.success("All cached data cleared. Fresh market data will be fetched on the next run.")

st.markdown(
    """
    <div class="disclaimer-box">
        <b>Disclaimer:</b> This app is for educational and research purposes only. Forecasts and signals are model-based estimates,
        not financial advice. Always verify with multiple sources and use your own judgment before taking any trade or investment decision.
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Main
# -----------------------------
if run_btn:
    loading_msg = st.empty()
    progress = st.progress(0)

    try:
        loading_msg.info(f"Loading market data for {symbol}...")
        progress.progress(15)

        raw_data, source_used, source_note = fetch_stock_data(symbol)
        st.session_state.last_fetch_source = source_used
        st.session_state.last_fetch_note = source_note
        progress.progress(45)

        if raw_data.empty or len(raw_data) < MIN_PRICE_ROWS:
            progress.empty()
            loading_msg.empty()
            st.error(f"Could not fetch enough usable data for {symbol}.")
            st.info("Try symbols like SPICEJET.BO, RELIANCE.NS, HDFCBANK.NS, TCS.NS, INFY.NS, SBIN.NS, ITC.NS or ^NSEI.")
            st.stop()

        loading_msg.info("Processing indicators, forecast, fundamentals, and news...")
        data = raw_data.copy()
        close_series = pd.to_numeric(data["Close"], errors="coerce").dropna()
        current_price = float(close_series.iloc[-1])
        prev_price = float(close_series.iloc[-2]) if len(close_series) > 1 else current_price
        day_change = current_price - prev_price
        day_change_pct = (day_change / prev_price * 100) if prev_price else 0
        high_52w = float(data["Close"].tail(252).max()) if len(data) >= MIN_INDICATOR_ROWS else current_price
        low_52w = float(data["Close"].tail(252).min()) if len(data) >= MIN_INDICATOR_ROWS else current_price
        avg_volume = float(data["Volume"].tail(20).mean()) if data["Volume"].notna().any() else 0.0

        has_indicator_data = len(raw_data) >= MIN_INDICATOR_ROWS
        if has_indicator_data:
            data = add_indicators(raw_data)
            signal = generate_signal(data)
            risk_name, volatility = risk_level(data)
            levels = get_support_resistance(data)
        else:
            signal = {
                "label": "LIMITED DATA",
                "score": 0,
                "reasons": ["Not enough rows for full technical analysis"],
                "color": "orange",
                "confidence": 35,
            }
            risk_name, volatility = "Unknown", 0.0
            levels = {"support": None, "resistance": None, "stop_loss": None, "breakout": None}
        progress.progress(70)

        fundamentals = fetch_fundamentals(symbol) if show_fundamentals else {}
        news_df = fetch_news(symbol)
        progress.progress(88)

        forecast_error = None
        hist_df = None
        forecast = None
        future_rows = None
        try:
            hist_df, forecast = build_forecast(data, days)
            future_rows = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(days).copy()
        except Exception as ex:
            forecast_error = str(ex)

        if has_indicator_data:
            ai_summary = build_ai_insight(symbol, current_price, signal, future_rows, risk_name, volatility, levels, forecast.attrs.get("cap_meta", {}))
        else:
            ai_summary = f"For {symbol}, limited market history was available in this run, so full technical insight and forecast confidence may be restricted."
        progress.progress(100)
        progress.empty()
        loading_msg.empty()

        source_label = {
            "yfinance": "yfinance",
            "nselib": "nselib fallback",
            "none": "unknown source",
        }.get(source_used, source_used)
        st.success(f"Data loaded for {symbol} using {source_label}.")
        if source_note:
            st.caption(source_note)
        if not has_indicator_data:
            st.warning("Limited market history received. Showing basic price analysis only.")

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Current Price", f"₹ {fmt_num(current_price)}", f"{day_change:,.2f}")
        m2.metric("Daily Change %", f"{day_change_pct:,.2f}%")
        m3.metric("52W High", f"₹ {fmt_num(high_52w)}")
        m4.metric("52W Low", f"₹ {fmt_num(low_52w)}")
        m5.metric("20D Avg Volume", f"{avg_volume:,.0f}" if avg_volume else "-")
        m6.metric("Volatility", f"{volatility:.2f}%")

        st.markdown(
            f"""
            <div style="padding:12px 16px;border-radius:12px;background:#111827;margin:10px 0 12px 0;">
                <span style="font-size:18px;font-weight:700;color:white;">Signal:</span>
                <span style="font-size:20px;font-weight:800;color:{signal['color']};margin-left:8px;">{signal['label']}</span>
                <span style="margin-left:16px;font-size:16px;color:white;">Confidence: <b>{signal['confidence']}%</b></span>
                <span style="margin-left:16px;font-size:16px;color:white;">Risk: <b>{risk_name}</b></span>
                <span style="margin-left:16px;font-size:16px;color:white;">Score: <b>{signal['score']}</b></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(f"<div class='ai-box'><b>🧠 AI Insight</b><br><br>{ai_summary}</div>", unsafe_allow_html=True)

        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
            ["Overview", "Technical Analysis", "Forecast", "Backtest", "Fundamentals", "Watchlist", "Comparison", "News Sentiment", "Portfolio"]
        )

        with tab1:
            st.subheader(f"📉 {symbol} Price Overview")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"], name="Price"))
            if has_indicator_data:
                fig.add_trace(go.Scatter(x=data.index, y=data["SMA20"], mode="lines", name="SMA20"))
                fig.add_trace(go.Scatter(x=data.index, y=data["SMA50"], mode="lines", name="SMA50"))
                fig.add_trace(go.Scatter(x=data.index, y=data["SMA200"], mode="lines", name="SMA200"))
            if levels["support"] is not None:
                fig.add_hline(y=levels["support"], annotation_text=f"Support {levels['support']}", line_dash="dot")
            if levels["resistance"] is not None:
                fig.add_hline(y=levels["resistance"], annotation_text=f"Resistance {levels['resistance']}", line_dash="dot")
            title_suffix = "with Moving Averages" if has_indicator_data else "(Limited Data Mode)"
            fig.update_layout(title=f"{symbol} Historical Price {title_suffix}", xaxis_title="Date", yaxis_title="Price", xaxis_rangeslider_visible=False, height=650)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("🎯 Support / Resistance")
            sr1, sr2, sr3, sr4 = st.columns(4)
            sr1.metric("Nearest Support", f"₹ {fmt_num(levels['support'])}" if levels["support"] is not None else "-")
            sr2.metric("Nearest Resistance", f"₹ {fmt_num(levels['resistance'])}" if levels["resistance"] is not None else "-")
            sr3.metric("Suggested Stop Loss", f"₹ {fmt_num(levels['stop_loss'])}" if levels["stop_loss"] is not None else "-")
            sr4.metric("Breakout Zone", f"₹ {fmt_num(levels['breakout'])}" if levels["breakout"] is not None else "-")

        with tab2:
            if not show_technical:
                st.info("Technical indicators are hidden from sidebar settings.")
            elif not has_indicator_data:
                st.warning("Technical indicators need at least 20 valid rows. Current run has limited market history.")
            else:
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

        with tab3:
            st.subheader(f"📈 Prediction for Next {days} Days")
            if future_rows is None or future_rows.empty:
                st.warning("Prediction is not available right now.")
                if forecast_error:
                    st.code(forecast_error)
            else:
                final_pred = float(future_rows["yhat"].iloc[-1])
                final_lower = float(future_rows["yhat_lower"].iloc[-1])
                final_upper = float(future_rows["yhat_upper"].iloc[-1])
                trend = "Bullish 📈" if final_pred > current_price else "Bearish 📉"
                expected_move = final_pred - current_price
                expected_move_pct = (expected_move / current_price * 100) if current_price else 0
                cap_meta = forecast.attrs.get("cap_meta", {}) if forecast is not None else {}
                reliability, reliability_note = describe_forecast_reliability(current_price, final_pred, cap_meta)
                status = "Capped & Sanitized ✅" if cap_meta.get("is_capped") else "Complete ✅"

                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Predicted End Price", f"₹ {fmt_num(final_pred)}")
                p2.metric("Expected Move", f"₹ {expected_move:,.2f}", f"{expected_move_pct:,.2f}%")
                p3.metric("Range Low", f"₹ {fmt_num(final_lower)}")
                p4.metric("Range High", f"₹ {fmt_num(final_upper)}")
                st.info(f"Prediction Status: {status} | Trend Outlook: {trend} | Reliability: {reliability}")
                st.caption(reliability_note)
                if cap_meta.get("is_capped"):
                    st.warning(
                        f"Raw model output looked too extreme for this horizon, so the forecast was clipped into a realistic band of ₹ {fmt_num(cap_meta.get('floor_price'))} to ₹ {fmt_num(cap_meta.get('ceiling_price'))}."
                    )

                past_review_df = build_past_prediction_review(data, days)

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=hist_df["ds"], y=hist_df["y"], mode="lines", name="Historical Actual"))

                if not past_review_df.empty:
                    true_df = past_review_df[past_review_df["Result"] == "TRUE"].copy()
                    false_df = past_review_df[past_review_df["Result"] == "FALSE"].copy()

                    if not true_df.empty:
                        fig2.add_trace(
                            go.Scatter(
                                x=true_df["Date"],
                                y=true_df["Actual ₹"],
                                mode="markers",
                                name="Past Prediction True",
                                marker=dict(color="green", size=10, symbol="circle"),
                                customdata=np.stack(
                                    [
                                        true_df["Previous Close ₹"],
                                        true_df["Predicted ₹"],
                                        true_df["Actual ₹"],
                                        true_df["Predicted Move %"],
                                        true_df["Actual Move %"],
                                        true_df["Error %"],
                                    ],
                                    axis=-1,
                                ),
                                hovertemplate=(
                                    "<b>%{x|%d-%b-%Y}</b><br>"
                                    "Prev Close: ₹ %{customdata[0]:,.2f}<br>"
                                    "Predicted: ₹ %{customdata[1]:,.2f}<br>"
                                    "Actual: ₹ %{customdata[2]:,.2f}<br>"
                                    "Pred Move: %{customdata[3]:,.2f}%<br>"
                                    "Actual Move: %{customdata[4]:,.2f}%<br>"
                                    "Error: %{customdata[5]:,.2f}%<br>"
                                    "Status: TRUE<extra></extra>"
                                ),
                            )
                        )

                    if not false_df.empty:
                        fig2.add_trace(
                            go.Scatter(
                                x=false_df["Date"],
                                y=false_df["Actual ₹"],
                                mode="markers",
                                name="Past Prediction False",
                                marker=dict(color="red", size=10, symbol="x"),
                                customdata=np.stack(
                                    [
                                        false_df["Previous Close ₹"],
                                        false_df["Predicted ₹"],
                                        false_df["Actual ₹"],
                                        false_df["Predicted Move %"],
                                        false_df["Actual Move %"],
                                        false_df["Error %"],
                                    ],
                                    axis=-1,
                                ),
                                hovertemplate=(
                                    "<b>%{x|%d-%b-%Y}</b><br>"
                                    "Prev Close: ₹ %{customdata[0]:,.2f}<br>"
                                    "Predicted: ₹ %{customdata[1]:,.2f}<br>"
                                    "Actual: ₹ %{customdata[2]:,.2f}<br>"
                                    "Pred Move: %{customdata[3]:,.2f}%<br>"
                                    "Actual Move: %{customdata[4]:,.2f}%<br>"
                                    "Error: %{customdata[5]:,.2f}%<br>"
                                    "Status: FALSE<extra></extra>"
                                ),
                            )
                        )

                fig2.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", line=dict(width=0), showlegend=False))
                fig2.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", fill="tonexty", line=dict(width=0), name="Confidence Range"))
                fig2.add_trace(go.Scatter(x=future_rows["ds"], y=future_rows["yhat"], mode="lines+markers", name="Future Predicted"))
                fig2.update_layout(title="Historical + Past Review + Future Forecast", xaxis_title="Date", yaxis_title="Price", height=580)
                st.plotly_chart(fig2, use_container_width=True)

                st.markdown("### Past Prediction Review")
                if not past_review_df.empty:
                    true_count = int((past_review_df["Result"] == "TRUE").sum())
                    false_count = int((past_review_df["Result"] == "FALSE").sum())
                    accuracy_pct = (true_count / len(past_review_df) * 100) if len(past_review_df) else 0

                    r1, r2, r3 = st.columns(3)
                    r1.metric("Reviewed Days", len(past_review_df))
                    r2.metric("True", true_count)
                    r3.metric("Accuracy", f"{accuracy_pct:.2f}%")

                    st.dataframe(style_prediction_review(past_review_df), use_container_width=True, height=420)
                else:
                    st.info("Not enough historical data to build past prediction review for the selected horizon.")

                st.markdown("### Upcoming Forecast")
                future_prediction_table = future_rows.copy().rename(
                    columns={"ds": "Date", "yhat": "Predicted ₹", "yhat_lower": "Lower ₹", "yhat_upper": "Upper ₹"}
                )
                future_prediction_table["Actual ₹"] = np.nan
                future_prediction_table["Result"] = "Pending"
                future_prediction_table["Type"] = "Future Forecast"

                st.dataframe(
                    future_prediction_table[["Date", "Predicted ₹", "Lower ₹", "Upper ₹", "Actual ₹", "Result", "Type"]].round(2),
                    use_container_width=True,
                    height=420,
                )

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
                wl_df = build_watchlist_snapshot(tuple(st.session_state.watchlist))
                if wl_df.empty:
                    st.warning("Watchlist data is not available right now.")
                else:
                    st.dataframe(wl_df, use_container_width=True)

        with tab7:
            st.subheader("⚖️ Comparison Dashboard")
            if len(compare_symbols) < 2:
                st.info("Add at least 2 stocks in comparison mode.")
            else:
                cmp_df, norm_df = build_comparison_snapshot(tuple(compare_symbols), days)
                if cmp_df.empty:
                    st.warning("Comparison data not available right now.")
                else:
                    st.dataframe(cmp_df, use_container_width=True)
                    if not norm_df.empty:
                        fig_cmp = go.Figure()
                        for col in norm_df.columns:
                            fig_cmp.add_trace(go.Scatter(x=norm_df.index, y=norm_df[col], mode="lines", name=col))
                        fig_cmp.update_layout(title="60-Day Relative Performance (Base = 100)", xaxis_title="Date", yaxis_title="Indexed Value", height=500)
                        st.plotly_chart(fig_cmp, use_container_width=True)

        with tab8:
            st.subheader("📰 News Sentiment")
            if news_df.empty:
                st.info("Recent news not available right now.")
            else:
                st.dataframe(news_df[["Published", "Title", "Publisher", "Sentiment", "SentimentScore"]], use_container_width=True)

        with tab9:
            st.subheader("💼 Portfolio Tracker")
            pf1, pf2, pf3 = st.columns(3)
            with pf1:
                pf_symbol = st.text_input("Portfolio Symbol", value="SBIN.NS")
            with pf2:
                pf_qty = st.number_input("Quantity", min_value=0.0, value=1.0, step=1.0)
            with pf3:
                pf_buy = st.number_input("Buy Price", min_value=0.0, value=100.0, step=1.0)

            add_col, remove_col = st.columns(2)
            with add_col:
                if st.button("Add Portfolio Row"):
                    add_portfolio_row(pf_symbol, pf_qty, pf_buy)
                    st.success("Portfolio row added.")
            with remove_col:
                if st.session_state.portfolio and st.button("Remove Last Row"):
                    remove_portfolio_row(len(st.session_state.portfolio) - 1)
                    st.warning("Last portfolio row removed.")

            portfolio_df = build_portfolio_snapshot(st.session_state.portfolio)
            if portfolio_df.empty:
                st.info("Portfolio data is not available right now.")
            else:
                total_invested = float(portfolio_df["Invested"].sum())
                total_value = float(portfolio_df["Current Value"].sum())
                total_pnl = total_value - total_invested
                total_pnl_pct = (total_pnl / total_invested * 100) if total_invested else 0.0

                pp1, pp2, pp3 = st.columns(3)
                pp1.metric("Total Invested", f"₹ {fmt_num(total_invested)}")
                pp2.metric("Current Value", f"₹ {fmt_num(total_value)}")
                pp3.metric("Total P/L", f"₹ {fmt_num(total_pnl)}", f"{total_pnl_pct:.2f}%")
                st.dataframe(portfolio_df, use_container_width=True)

    except Exception as ex:
        progress.empty()
        loading_msg.empty()
        st.error("Something went wrong while processing the app.")
        st.code(str(ex))
else:
    st.info("Select a stock, then click 'Fetch Data & Predict' to load fresh analysis.")
