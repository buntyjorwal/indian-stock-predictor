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
    from streamlit_autorefresh import st_autorefresh
    AUTO_REFRESH_OK = True
except Exception:
    AUTO_REFRESH_OK = False

try:
    from nselib import capital_market
except Exception:
    capital_market = None

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


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

if "live_auto_refresh" not in st.session_state:
    st.session_state.live_auto_refresh = True

if "live_refresh_sec" not in st.session_state:
    st.session_state.live_refresh_sec = 60

# -----------------------------
# UI helpers
# -----------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
            :root {
                --bg-main: #06101f;
                --bg-soft: #0b1729;
                --bg-card: linear-gradient(180deg, rgba(13,24,43,0.96) 0%, rgba(8,15,28,0.98) 100%);
                --line: rgba(148,163,184,0.16);
                --line-strong: rgba(96,165,250,0.28);
                --text-main: #f8fafc;
                --text-soft: #cbd5e1;
                --text-dim: #93a4bd;
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(37, 99, 235, 0.18), transparent 30%),
                    radial-gradient(circle at top right, rgba(14, 165, 233, 0.12), transparent 28%),
                    linear-gradient(180deg, #030814 0%, #06101f 45%, #040b16 100%);
                color: var(--text-main);
            }

            .block-container {
                max-width: 1800px;
                padding-top: 1.25rem;
                padding-bottom: 1.75rem;
            }

            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #06101f 0%, #081324 100%);
                border-right: 1px solid rgba(96, 165, 250, 0.10);
            }

            section[data-testid="stSidebar"] > div {
                padding-top: 0.8rem;
            }

            .app-card, .panel-box, .mini-window, .hero-metric, .stat-card, .section-card, .info-card, .reason-card {
                background: var(--bg-card);
                border: 1px solid var(--line);
                box-shadow: 0 18px 40px rgba(0,0,0,0.26);
            }

            .app-card {
                padding: 18px 22px;
                border-radius: 20px;
                margin-bottom: 14px;
                position: relative;
                overflow: hidden;
            }

            .app-card::before {
                content: "";
                position: absolute;
                inset: 0 0 auto 0;
                height: 3px;
                background: linear-gradient(90deg, #38bdf8, #60a5fa, #22c55e);
            }

            .hero-title {
                font-size: 2rem;
                font-weight: 800;
                line-height: 1.1;
                color: var(--text-main);
                margin: 0;
            }

            .hero-subtitle {
                color: var(--text-soft);
                margin-top: 8px;
                font-size: 0.98rem;
            }

            .terminal-toolbar {
                background: linear-gradient(180deg, rgba(15,23,42,0.92) 0%, rgba(9,16,31,0.98) 100%);
                padding: 12px 14px;
                border-radius: 16px;
                border: 1px solid var(--line-strong);
                margin-bottom: 14px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 12px;
                color: var(--text-main);
                flex-wrap: wrap;
            }

            .terminal-chip {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                padding: 6px 12px;
                border-radius: 999px;
                border: 1px solid rgba(148,163,184,0.18);
                background: rgba(15,23,42,0.95);
                color: #dbeafe;
                font-size: 12px;
                font-weight: 600;
                margin-right: 8px;
                margin-bottom: 6px;
            }

            .terminal-chip-soft {
                background: rgba(10,18,34,0.85);
                color: var(--text-soft);
            }

            .dock-title {
                font-size: 11px;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                color: #93c5fd;
                margin-bottom: 10px;
                font-weight: 800;
            }

            .panel-box {
                border-radius: 18px;
                padding: 14px 14px 12px 14px;
                margin-bottom: 12px;
            }

            .tree-group-title {
                font-size: 12px;
                color: #dbeafe;
                font-weight: 700;
                margin: 12px 0 8px 0;
                padding-top: 4px;
                border-top: 1px solid rgba(148,163,184,0.10);
            }

            .tree-group-title:first-of-type {
                margin-top: 2px;
                border-top: none;
            }

            .market-watch-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 9px 10px;
                border-bottom: 1px solid rgba(148,163,184,0.10);
                border-radius: 10px;
                font-size: 13px;
                color: var(--text-soft);
            }

            .market-watch-item:hover {
                background: rgba(96,165,250,0.08);
            }

            .market-watch-item:last-child {
                border-bottom: none;
            }

            .tree-badge {
                min-width: 34px;
                text-align: center;
                padding: 2px 8px;
                border-radius: 999px;
                font-size: 11px;
                color: #dbeafe;
                border: 1px solid rgba(148,163,184,0.12);
                background: rgba(15,23,42,0.75);
            }

            .hero-metric-grid, .stat-card-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 12px;
                margin: 10px 0 14px 0;
            }

            .hero-metric, .stat-card {
                border-radius: 18px;
                padding: 14px 16px;
                min-height: 108px;
                position: relative;
                overflow: hidden;
            }

            .hero-metric::after, .stat-card::after {
                content: "";
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 2px;
                background: linear-gradient(90deg, rgba(56,189,248,0.85), rgba(34,197,94,0.85));
            }

            .metric-label {
                color: var(--text-dim);
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                margin-bottom: 10px;
            }

            .metric-value {
                color: var(--text-main);
                font-size: 1.85rem;
                font-weight: 800;
                line-height: 1.1;
            }

            .metric-sub {
                color: var(--text-soft);
                margin-top: 8px;
                font-size: 0.88rem;
            }

            .metric-positive {color: #4ade80;}
            .metric-negative {color: #f87171;}
            .metric-warning {color: #fbbf24;}
            .metric-accent {color: #7dd3fc;}

            .info-card, .reason-card, .section-card {
                border-radius: 18px;
                padding: 14px 16px;
                margin-bottom: 12px;
            }

            .info-kv {
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 12px;
                padding: 10px 0;
                border-bottom: 1px dashed rgba(148,163,184,0.14);
                font-size: 13px;
                color: var(--text-soft);
            }

            .info-kv strong {
                color: var(--text-main);
                font-size: 13px;
            }

            .info-kv:last-child {border-bottom: none;}

            .signal-pill {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 8px 12px;
                border-radius: 999px;
                font-weight: 800;
                font-size: 12px;
                letter-spacing: 0.04em;
                margin-bottom: 10px;
                color: white;
            }

            .signal-buy {background: linear-gradient(90deg, rgba(34,197,94,0.88), rgba(22,163,74,0.92));}
            .signal-sell {background: linear-gradient(90deg, rgba(239,68,68,0.9), rgba(220,38,38,0.92));}
            .signal-hold, .signal-watch {background: linear-gradient(90deg, rgba(245,158,11,0.9), rgba(217,119,6,0.92));}

            .reason-list {
                margin: 10px 0 0 0;
                padding-left: 16px;
                color: var(--text-soft);
            }

            .reason-list li {
                margin-bottom: 8px;
                line-height: 1.45;
            }

            .mini-window {
                border-radius: 18px;
                overflow: hidden;
                margin-top: 8px;
            }

            .mini-window-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px 14px;
                background: linear-gradient(180deg, rgba(16,24,40,0.95), rgba(8,14,26,0.96));
                border-bottom: 1px solid rgba(148,163,184,0.12);
                color: var(--text-main);
                font-size: 13px;
                font-weight: 700;
            }

            .mini-window-body {
                padding: 12px;
            }

            .custom-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 13px;
                color: var(--text-soft);
                overflow: hidden;
            }

            .custom-table thead th {
                background: rgba(15,23,42,0.96);
                color: #dbeafe;
                padding: 10px 12px;
                text-align: left;
                font-size: 12px;
                letter-spacing: 0.04em;
                border-bottom: 1px solid rgba(148,163,184,0.16);
            }

            .custom-table tbody td {
                padding: 10px 12px;
                border-bottom: 1px solid rgba(148,163,184,0.08);
                background: rgba(7,12,24,0.38);
            }

            .custom-table tbody tr:hover td {
                background: rgba(96,165,250,0.08);
            }

            .custom-table-wrap {
                overflow-x: auto;
                border-radius: 14px;
                border: 1px solid rgba(148,163,184,0.10);
            }

            .ai-box {
                background: linear-gradient(180deg, rgba(239,246,255,0.96) 0%, rgba(248,250,252,0.96) 100%);
                color: #0f172a;
                border: 1px solid #bfdbfe;
                padding: 16px;
                border-radius: 16px;
                margin-bottom: 16px;
                box-shadow: 0 12px 26px rgba(2, 6, 23, 0.10);
            }

            div[data-testid="stTabs"] {
                margin-top: 10px !important;
            }

            div[data-testid="stTabs"] [data-baseweb="tab-list"] {
                gap: 10px;
                padding-top: 6px;
            }

            .disclaimer-box {
                border-left: 4px solid #f59e0b;
                background: rgba(66, 32, 6, 0.55);
                padding: 12px 14px;
                color: #fde68a;
                border-radius: 12px;
                margin: 8px 0 14px 0;
            }

            div[data-testid="stMetric"] {
                background: var(--bg-card);
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 10px 14px;
            }

            div[data-testid="stMetricLabel"] {
                color: var(--text-dim);
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }

            div[data-testid="stMetricValue"] {
                color: var(--text-main);
            }

            div.stButton > button {
                border-radius: 12px;
                border: 1px solid rgba(96,165,250,0.22);
                background: linear-gradient(180deg, #10213f 0%, #0b162c 100%);
                color: #eff6ff;
                font-weight: 700;
                min-height: 44px;
            }

            div.stButton > button[kind="primary"] {
                background: linear-gradient(90deg, #2563eb, #1d4ed8);
                border: 1px solid rgba(96,165,250,0.45);
            }

            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
                padding: 0;
                margin-bottom: 12px;
                overflow-x: auto;
            }

            .stTabs [data-baseweb="tab"] {
                background: rgba(15,23,42,0.85);
                border: 1px solid rgba(148,163,184,0.12);
                color: var(--text-soft);
                border-radius: 12px;
                padding: 8px 14px;
                font-weight: 700;
            }

            .stTabs [aria-selected="true"] {
                background: linear-gradient(90deg, rgba(37,99,235,0.95), rgba(14,165,233,0.90));
                color: white !important;
                border-color: rgba(96,165,250,0.55) !important;
            }

            div[data-testid="stDataFrame"] {
                border: 1px solid rgba(148,163,184,0.12);
                border-radius: 16px;
                overflow: hidden;
                background: rgba(10,15,28,0.55);
            }

            @media (max-width: 1100px) {
                .hero-metric-grid, .stat-card-grid {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
            }

            @media (max-width: 640px) {
                .hero-metric-grid, .stat-card-grid {
                    grid-template-columns: repeat(1, minmax(0, 1fr));
                }
                .hero-title {font-size: 1.45rem;}
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def top_banner() -> None:
    st.markdown(
        """
        <div class="app-card">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:18px;flex-wrap:wrap;">
                <div style="max-width:860px;">
                    <div class="dock-title" style="margin-bottom:12px;">Professional Trading Workspace</div>
                    <h1 class="hero-title">AI Market Workstation</h1>
                    <div class="hero-subtitle">
                        Cleaner terminal layout, stronger visual hierarchy, safer native Streamlit rendering,
                        better chart presentation, and a more polished scan, watchlist, and support-resistance experience.
                    </div>
                </div>
                <div style="display:flex;flex-wrap:wrap;justify-content:flex-end;gap:8px;">
                    <span class="terminal-chip">Market Watch</span>
                    <span class="terminal-chip">Chart Lab</span>
                    <span class="terminal-chip">Forecast Lab</span>
                    <span class="terminal-chip">Model Explorer</span>
                    <span class="terminal-chip">Backtest Lab</span>
                    <span class="terminal-chip">Portfolio Terminal</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_terminal_toolbar(symbol: str, compare_count: int, source_label: str) -> None:
    st.markdown(
        f"""
        <div class="terminal-toolbar">
            <div>
                <span class="terminal-chip">Active Symbol: {symbol}</span>
                <span class="terminal-chip terminal-chip-soft">Compare Basket: {compare_count}</span>
                <span class="terminal-chip terminal-chip-soft">Data Source: {source_label}</span>
            </div>
            <div style="color:#93a4bd;font-size:12px;font-weight:600;">
                Pro terminal mode enabled
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _signal_class(signal_label: str) -> str:
    label = str(signal_label or "").upper()
    if "BUY" in label:
        return "signal-buy"
    if "SELL" in label:
        return "signal-sell"
    if "WATCH" in label:
        return "signal-watch"
    return "signal-hold"


def render_market_watch_panel(active_symbol: str, watchlist: list[str]) -> None:
    rows = list(dict.fromkeys((watchlist or []) + [active_symbol]))
    groups = {
        "Indices": ["^NSEI", "^NSEBANK"],
        "Banking": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS"],
        "Technology": ["TCS.NS", "INFY.NS", "WIPRO.NS"],
        "Energy": ["RELIANCE.NS", "ONGC.NS", "IOC.NS"],
        "Watchlist": rows[:12],
    }
    st.markdown("<div class='dock-title'>Symbols / Explorer Tree</div>", unsafe_allow_html=True)
    for title, symbols in groups.items():
        with st.expander(title, expanded=(title in ["Indices", "Watchlist"])):
            for sym in symbols:
                marker = "🟢" if sym == active_symbol else ("🔵" if sym in rows else "⚪")
                left, right = st.columns([5, 1])
                left.markdown(f"{marker} {sym}")
                right.markdown("EQ")


def render_info_panel(current_price: float, signal: dict, risk_name: str, levels: dict, fundamentals: dict) -> None:
    signal_label = signal.get("label", "-")
    signal_class = _signal_class(signal_label)
    pairs = [
        ("Current", f"₹ {fmt_num(current_price)}"),
        ("Signal", signal_label),
        ("Confidence", f"{signal.get('confidence', 0)}%"),
        ("Risk", risk_name),
        ("Support", f"₹ {fmt_num(levels.get('support'))}" if levels.get('support') is not None else "-"),
        ("Resistance", f"₹ {fmt_num(levels.get('resistance'))}" if levels.get('resistance') is not None else "-"),
        ("Market Cap", format_large_number(fundamentals.get('marketCap')) if fundamentals else "-"),
        ("PE", fmt_num(fundamentals.get('trailingPE')) if fundamentals else "-"),
    ]
    html = f'<div class="info-card"><div class="dock-title">Information</div><div class="signal-pill {signal_class}">{signal_label}</div>'
    html += ''.join([f'<div class="info-kv"><span>{k}</span><strong>{v}</strong></div>' for k, v in pairs])
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_signal_reason_panel(signal: dict) -> None:
    reasons = signal.get("reasons", [])[:8]
    if not reasons:
        reasons = ["No specific reason text available for the current signal."]
    reasons_html = ''.join([f'<li>{r}</li>' for r in reasons])
    st.markdown(
        f"""
        <div class="reason-card">
            <div class="dock-title">Signal Reasons</div>
            <ul class="reason-list">{reasons_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero_metrics(current_price: float, day_change: float, day_change_pct: float, high_52w: float, low_52w: float, avg_volume: float) -> None:
    change_class = "metric-positive" if day_change >= 0 else "metric-negative"
    change_sign = "+" if day_change >= 0 else ""
    html = f"""
        <div class="hero-metric-grid">
            <div class="hero-metric">
                <div class="metric-label">Current Price</div>
                <div class="metric-value">₹ {fmt_num(current_price)}</div>
                <div class="metric-sub">Latest close from selected symbol</div>
            </div>
            <div class="hero-metric">
                <div class="metric-label">Day Change</div>
                <div class="metric-value {change_class}">{change_sign}{fmt_num(day_change)}</div>
                <div class="metric-sub {change_class}">{change_sign}{day_change_pct:.2f}% vs previous close</div>
            </div>
            <div class="hero-metric">
                <div class="metric-label">52W Range</div>
                <div class="metric-value metric-accent">₹ {fmt_num(low_52w)} – ₹ {fmt_num(high_52w)}</div>
                <div class="metric-sub">Low to high range snapshot</div>
            </div>
            <div class="hero-metric">
                <div class="metric-label">20D Avg Volume</div>
                <div class="metric-value">{fmt_num(avg_volume)}</div>
                <div class="metric-sub">Average traded volume</div>
            </div>
        </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_stat_cards(items: list[tuple[str, str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, sub) in zip(cols, items):
        with col:
            with st.container(border=True):
                st.caption(label.upper())
                st.markdown(f"### {value}")
                st.caption(sub)


def render_table_card(title: str, df: pd.DataFrame, max_rows: int = 10) -> None:
    st.markdown(f"<div class='dock-title'>{title}</div>", unsafe_allow_html=True)
    with st.container(border=True):
        if df is None or df.empty:
            st.info('No data available.')
            return
        show_df = df.head(max_rows).copy()
        for col in show_df.columns:
            if pd.api.types.is_float_dtype(show_df[col]):
                show_df[col] = show_df[col].map(lambda x: round(float(x), 2) if pd.notna(x) else x)
        st.dataframe(show_df, use_container_width=True, hide_index=True, height=min(420, 42 * (len(show_df) + 1)))


def dataframe_to_html(df: pd.DataFrame, max_rows: int = 10) -> str:
    if df is None or df.empty:
        return "<div style='color:#cbd5e1;padding:8px 2px;'>No data available.</div>"
    show_df = df.head(max_rows).copy()
    for col in show_df.columns:
        if pd.api.types.is_float_dtype(show_df[col]):
            show_df[col] = show_df[col].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "")
    return f"<div class='custom-table-wrap'>{show_df.to_html(index=False, classes='custom-table', border=0)}</div>"


def render_mini_window(title: str, body_html: str) -> None:
    st.markdown(f"<div class='dock-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(body_html, unsafe_allow_html=True)


def build_scanner_snapshot(symbols: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        try:
            df, _, _ = fetch_stock_data(sym)
            if df is None or len(df) < 25:
                continue
            df = add_indicators(df)
            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else last
            close = float(last["Close"])
            chg_pct = ((close - float(prev["Close"])) / float(prev["Close"]) * 100) if float(prev["Close"]) else 0.0
            sma20 = float(last["SMA20"]) if pd.notna(last.get("SMA20")) else np.nan
            rsi = float(last["RSI14"]) if pd.notna(last.get("RSI14")) else np.nan
            trend = "Bullish" if pd.notna(sma20) and close > sma20 else ("Bearish" if pd.notna(sma20) and close < sma20 else "Sideways")
            signal = "Breakout" if chg_pct > 1.5 and trend == "Bullish" else ("Weak" if abs(chg_pct) < 0.5 else trend)
            rows.append({"Ticker": sym, "Last": round(close,2), "%Chg": round(chg_pct,2), "RSI": round(rsi,2) if pd.notna(rsi) else np.nan, "Trend": trend, "Signal": signal})
        except Exception:
            continue
    return pd.DataFrame(rows)



def build_symbol_heatmap(symbols: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        try:
            df, _, _ = fetch_stock_data(sym)
            if df is None or len(df) < 6:
                continue
            close = float(df["Close"].iloc[-1])
            ret_5 = ((close / float(df["Close"].iloc[-6])) - 1) * 100 if len(df) >= 6 else np.nan
            vol = float(df["Close"].pct_change().dropna().tail(20).std() * np.sqrt(252) * 100) if len(df) > 20 else np.nan
            rows.append({"Ticker": sym, "5D %": round(ret_5,2), "Volatility %": round(vol,2) if pd.notna(vol) else np.nan})
        except Exception:
            continue
    return pd.DataFrame(rows)


def build_corr_matrix(symbols: tuple[str, ...]) -> pd.DataFrame:
    series = {}
    for sym in symbols:
        try:
            df, _, _ = fetch_stock_data(sym)
            if not df.empty:
                series[sym] = df['Close'].pct_change()
        except Exception:
            continue
    if len(series) < 2:
        return pd.DataFrame()
    return pd.DataFrame(series).dropna(how='all').corr().round(2)

# -----------------------------
# Live dashboard helpers
# -----------------------------
LIVE_INDEX_SYMBOLS = {
    "NIFTY 50": "^NSEI",
    "BANK NIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
}

LIVE_WATCHLIST_FALLBACK = [
    "RELIANCE.NS", "HDFCBANK.NS", "TCS.NS", "INFY.NS",
    "ICICIBANK.NS", "SBIN.NS", "ITC.NS", "LT.NS"
]


@st.cache_data(ttl=15, show_spinner=False)
def fetch_intraday_data(symbol: str, period: str = "1d", interval: str = "5m") -> pd.DataFrame:
    try:
        raw = yf.download(
            symbol,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        if raw is None or raw.empty:
            return pd.DataFrame()
        raw = flatten_columns(raw.copy())

        open_col = find_price_column(raw, "Open")
        high_col = find_price_column(raw, "High")
        low_col = find_price_column(raw, "Low")
        close_col = find_price_column(raw, "Close")
        volume_col = find_price_column(raw, "Volume")

        if not close_col:
            return pd.DataFrame()

        df = pd.DataFrame(index=pd.to_datetime(raw.index))
        if open_col:
            df["Open"] = pd.to_numeric(raw[open_col], errors="coerce")
        if high_col:
            df["High"] = pd.to_numeric(raw[high_col], errors="coerce")
        if low_col:
            df["Low"] = pd.to_numeric(raw[low_col], errors="coerce")
        df["Close"] = pd.to_numeric(raw[close_col], errors="coerce")
        if volume_col:
            df["Volume"] = pd.to_numeric(raw[volume_col], errors="coerce")
        return df.dropna(subset=["Close"]).copy()
    except Exception:
        return pd.DataFrame()


def safe_float(v, default=0.0):
    try:
        if v is None or pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


@st.cache_data(ttl=15, show_spinner=False)
def build_live_symbol_snapshot(symbols: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        try:
            intraday = fetch_intraday_data(sym, period="2d", interval="5m")
            if intraday.empty:
                daily, _, _ = fetch_stock_data(sym)
                if daily is None or daily.empty or len(daily) < 2:
                    continue

                last_close = safe_float(daily["Close"].iloc[-1])
                prev_close = safe_float(daily["Close"].iloc[-2], last_close)
                day_change = last_close - prev_close
                day_change_pct = (day_change / prev_close * 100) if prev_close else 0.0

                rows.append({
                    "Symbol": sym,
                    "Name": sym.replace(".NS", "").replace("^", ""),
                    "LTP": last_close,
                    "Change": day_change,
                    "Change %": day_change_pct,
                    "Open": safe_float(daily["Open"].iloc[-1], np.nan),
                    "High": safe_float(daily["High"].iloc[-1], np.nan),
                    "Low": safe_float(daily["Low"].iloc[-1], np.nan),
                    "Volume": safe_float(daily["Volume"].iloc[-1], np.nan),
                    "Source": "Daily"
                })
                continue

            ltp = safe_float(intraday["Close"].iloc[-1])
            prev_close = safe_float(intraday["Close"].iloc[0], ltp)
            opn = safe_float(intraday["Open"].iloc[0], ltp) if "Open" in intraday.columns else ltp
            high = safe_float(intraday["High"].max(), ltp) if "High" in intraday.columns else ltp
            low = safe_float(intraday["Low"].min(), ltp) if "Low" in intraday.columns else ltp
            vol = safe_float(intraday["Volume"].sum(), np.nan) if "Volume" in intraday.columns else np.nan

            chg = ltp - prev_close
            chg_pct = (chg / prev_close * 100) if prev_close else 0.0

            rows.append({
                "Symbol": sym,
                "Name": sym.replace(".NS", "").replace("^", ""),
                "LTP": ltp,
                "Change": chg,
                "Change %": chg_pct,
                "Open": opn,
                "High": high,
                "Low": low,
                "Volume": vol,
                "Source": "Intraday"
            })
        except Exception:
            continue

    return pd.DataFrame(rows)


@st.cache_data(ttl=15, show_spinner=False)
def build_market_breadth_snapshot(symbols: tuple[str, ...]) -> dict:
    df = build_live_symbol_snapshot(symbols)

    if df.empty or "Change" not in df.columns or "Change %" not in df.columns:
        return {
            "Traded": 0,
            "Advances": 0,
            "Declines": 0,
            "Unchanged": 0,
            "Top Gainers": pd.DataFrame(),
            "Top Losers": pd.DataFrame(),
        }

    df = df.copy()
    df["Change"] = pd.to_numeric(df["Change"], errors="coerce").fillna(0.0)
    df["Change %"] = pd.to_numeric(df["Change %"], errors="coerce").fillna(0.0)

    advances = int((df["Change"] > 0).sum())
    declines = int((df["Change"] < 0).sum())
    unchanged = int((df["Change"] == 0).sum())

    # Only positive movers in gainers
    gainers = (
        df[df["Change"] > 0]
        .sort_values(["Change %", "Change"], ascending=[False, False])
        .head(8)
        .copy()
    )

    # Only negative movers in losers
    losers = (
        df[df["Change"] < 0]
        .sort_values(["Change %", "Change"], ascending=[True, True])
        .head(8)
        .copy()
    )

    # Fallback in case market has no gainers / no losers
    if gainers.empty:
        gainers = df.sort_values(["Change %", "Change"], ascending=[False, False]).head(8).copy()

    if losers.empty:
        losers = df.sort_values(["Change %", "Change"], ascending=[True, True]).head(8).copy()

    cols = ["Name", "LTP", "Change", "Change %"]

    return {
        "Traded": int(len(df)),
        "Advances": advances,
        "Declines": declines,
        "Unchanged": unchanged,
        "Top Gainers": gainers[cols].reset_index(drop=True),
        "Top Losers": losers[cols].reset_index(drop=True),
    }


def render_live_ticker_bar(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        st.info("Live ticker data is not available right now.")
        return

    chips = []
    for _, r in df.iterrows():
        chg = safe_float(r.get("Change"))
        pct = safe_float(r.get("Change %"))
        color = "#4ade80" if chg >= 0 else "#f87171"
        sign = "+" if chg >= 0 else ""

        chip_html = f"""
        <div style="
            min-width:210px;
            padding:12px 14px;
            border-radius:16px;
            border:1px solid rgba(148,163,184,0.12);
            background:linear-gradient(180deg, rgba(13,24,43,0.96) 0%, rgba(8,15,28,0.98) 100%);
            box-shadow:0 10px 24px rgba(0,0,0,0.20);
        ">
            <div style="font-size:12px;color:#cbd5e1;font-weight:700;">{r.get('Name')}</div>
            <div style="font-size:1.55rem;color:#f8fafc;font-weight:800;line-height:1.15;">{fmt_num(r.get('LTP'))}</div>
            <div style="font-size:0.92rem;color:{color};font-weight:700;">{sign}{fmt_num(chg)} ({sign}{pct:.2f}%)</div>
        </div>
        """
        chips.append(chip_html)

    st.markdown(
        f"""
        <div style="
            display:flex;
            gap:12px;
            overflow-x:auto;
            padding-bottom:8px;
            margin-bottom:14px;
        ">
            {''.join(chips)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_market_stats_cards(breadth: dict) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stock Traded", breadth.get("Traded", 0))
    c2.metric("Advances", breadth.get("Advances", 0))
    c3.metric("Declines", breadth.get("Declines", 0))
    c4.metric("Unchanged", breadth.get("Unchanged", 0))


def plot_live_intraday_chart(symbol: str, title: str = ""):
    df = fetch_intraday_data(symbol, period="1d", interval="5m")
    if df.empty:
        st.warning(f"Intraday chart not available for {symbol}")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name=title or symbol,
            fill="tozeroy"
        )
    )

    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=35, b=10),
        title=title or symbol,
        template="plotly_dark",
        xaxis_title="Time",
        yaxis_title="Price",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)


def render_live_dashboard_home(selected_symbol: str) -> None:
    live_symbols = list(LIVE_INDEX_SYMBOLS.values())


    wl = list(dict.fromkeys((st.session_state.watchlist or []) + LIVE_WATCHLIST_FALLBACK))
    if selected_symbol and selected_symbol not in wl and not selected_symbol.startswith("^"):
        wl.insert(0, selected_symbol)

    live_symbols.extend(wl[:10])
    live_df = build_live_symbol_snapshot(tuple(dict.fromkeys(live_symbols)))
    index_df = live_df[live_df["Symbol"].isin(LIVE_INDEX_SYMBOLS.values())].copy() if not live_df.empty else pd.DataFrame()

    st.markdown("### 📡 Live Market Dashboard")
    render_live_ticker_bar(index_df if not index_df.empty else live_df.head(6))

    breadth = build_market_breadth_snapshot(tuple(wl[:12]))
    render_market_stats_cards(breadth)

    left, right = st.columns([2.15, 1], gap="large")

    with left:
        with st.container(border=True):
            plot_live_intraday_chart("^NSEI", "NIFTY 50 Intraday")

        st.markdown("#### 🚀 Top Gainers")
        tg = breadth.get("Top Gainers", pd.DataFrame())
        if tg.empty:
            st.info("Top gainers not available.")
        else:
            st.dataframe(tg, use_container_width=True, hide_index=True, height=320)

    with right:
        st.markdown("#### Live Index Snapshot")

        for label, sym in LIVE_INDEX_SYMBOLS.items():
            row_df = live_df[live_df["Symbol"] == sym]
            with st.container(border=True):
                st.write(f"**{label}**")
                if row_df.empty:
                    st.caption("Data not available")
                else:
                    r = row_df.iloc[0]
                    st.metric(
                        label="LTP",
                        value=fmt_num(r["LTP"]),
                        delta=f"{r['Change']:+.2f} ({r['Change %']:+.2f}%)"
                    )
                    st.caption(
                        f"Open: {fmt_num(r['Open'])} | High: {fmt_num(r['High'])} | Low: {fmt_num(r['Low'])}"
                    )

        st.markdown("#### 🔻 Top Losers")
        tl = breadth.get("Top Losers", pd.DataFrame())
        if tl.empty:
            st.info("Top losers not available.")
        else:
            st.dataframe(tl, use_container_width=True, hide_index=True, height=320)

    st.markdown("#### ⭐ Live Watchlist")
    watch_df = live_df[live_df["Symbol"].isin(wl[:8])].copy()
    if watch_df.empty:
        st.info("Watchlist live data is not available.")
    else:
        show_cols = ["Name", "LTP", "Change", "Change %", "Open", "High", "Low", "Source"]
        st.dataframe(watch_df[show_cols], use_container_width=True, hide_index=True, height=320)

    st.caption(
        f"Last updated: {pd.Timestamp.now().strftime('%d-%b-%Y %I:%M:%S %p')} | "
        f"Auto refresh: {'ON' if st.session_state.live_auto_refresh else 'OFF'} | "
        f"Interval: {st.session_state.live_refresh_sec}s"
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
HYBRID_MIN_CONFIDENCE = 60
HYBRID_REVIEW_MAX_WINDOWS = 60
ENSEMBLE_MIN_MODELS = 3
MODEL_BACKTEST_DAYS_DEFAULT = 30


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




def classify_direction(pred_move: float, actual_move: float) -> str:
    if (pred_move > 0 and actual_move > 0) or (pred_move < 0 and actual_move < 0):
        return "Yes"
    if abs(pred_move) < 1e-9 and abs(actual_move) < 1e-9:
        return "Yes"
    return "No"


def get_trend_bias(last_row: pd.Series) -> tuple[int, str]:
    close = float(last_row["Close"])
    sma20 = float(last_row["SMA20"]) if pd.notna(last_row.get("SMA20")) else np.nan
    sma50 = float(last_row["SMA50"]) if pd.notna(last_row.get("SMA50")) else np.nan
    sma200 = float(last_row["SMA200"]) if pd.notna(last_row.get("SMA200")) else np.nan

    if pd.notna(sma50) and pd.notna(sma200) and close > sma50 > sma200:
        return 1, "Bullish Trend"
    if pd.notna(sma50) and pd.notna(sma200) and close < sma50 < sma200:
        return -1, "Bearish Trend"
    if pd.notna(sma20) and close > sma20:
        return 1, "Short-term Bullish"
    if pd.notna(sma20) and close < sma20:
        return -1, "Short-term Bearish"
    return 0, "Sideways"


def hybrid_model_prediction(train_price_df: pd.DataFrame, horizon: int = 1) -> tuple[float, float, float]:
    log_train = train_price_df.copy()
    log_train["y"] = np.log(log_train["y"])

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.10 if horizon <= 3 else 0.15,
    )
    model.fit(log_train)

    future = model.make_future_dataframe(periods=horizon)
    fc = model.predict(future)
    pred_price = float(np.exp(fc["yhat"].iloc[-1]))
    pred_low = float(np.exp(fc["yhat_lower"].iloc[-1]))
    pred_high = float(np.exp(fc["yhat_upper"].iloc[-1]))
    return pred_price, pred_low, pred_high


def hybrid_decision_from_train(train_market_df: pd.DataFrame, horizon: int = 1, min_confidence: int = HYBRID_MIN_CONFIDENCE) -> dict:
    if len(train_market_df) < max(PREDICTION_MIN_ROWS, 220):
        return {"ok": False, "reason": "Not enough rows for hybrid decision."}

    train_price_df = train_market_df[["Close"]].reset_index().rename(columns={train_market_df.index.name or train_market_df.reset_index().columns[0]: "ds", "Close": "y"})
    # safer rename if ds missing
    if "ds" not in train_price_df.columns:
        train_price_df.columns = ["ds", "y"]
    train_price_df["ds"] = pd.to_datetime(train_price_df["ds"])
    train_price_df["y"] = pd.to_numeric(train_price_df["y"], errors="coerce")
    train_price_df = train_price_df.dropna()

    last = train_market_df.iloc[-1]
    prev_close = float(last["Close"])

    try:
        pred_price, pred_low, pred_high = hybrid_model_prediction(train_price_df[["ds", "y"]].copy(), horizon=horizon)
        pred_price = sanitize_single_prediction(pred_price, prev_close, train_price_df["y"])
        pred_low = sanitize_single_prediction(pred_low, prev_close, train_price_df["y"])
        pred_high = sanitize_single_prediction(pred_high, prev_close, train_price_df["y"])
    except Exception as ex:
        return {"ok": False, "reason": str(ex)}

    prophet_move_pct = ((pred_price - prev_close) / prev_close * 100) if prev_close else 0.0
    signal = generate_signal(train_market_df)
    trend_bias, trend_label = get_trend_bias(last)
    rsi = float(last["RSI14"]) if pd.notna(last.get("RSI14")) else np.nan
    macd = float(last["MACD"]) if pd.notna(last.get("MACD")) else np.nan
    macd_signal = float(last["MACDSignal"]) if pd.notna(last.get("MACDSignal")) else np.nan
    atr14 = float(last["ATR14"]) if pd.notna(last.get("ATR14")) else np.nan
    atr_pct = ((atr14 / prev_close) * 100) if prev_close and pd.notna(atr14) else np.nan

    score = 0.0
    reasons = []

    if prophet_move_pct > 0.15:
        score += 2.0
        reasons.append("Prophet bullish")
    elif prophet_move_pct < -0.15:
        score -= 2.0
        reasons.append("Prophet bearish")
    else:
        reasons.append("Prophet flat")

    score += max(-2, min(2, signal["score"] / 2.0))
    reasons.append(f"Technical score {signal['score']}")

    score += 1.5 * trend_bias
    reasons.append(trend_label)

    if pd.notna(rsi):
        if 52 <= rsi <= 68:
            score += 0.75
            reasons.append("RSI bullish zone")
        elif 32 <= rsi <= 48:
            score -= 0.75
            reasons.append("RSI bearish zone")
        elif 48 < rsi < 52:
            reasons.append("RSI neutral")

    if pd.notna(macd) and pd.notna(macd_signal):
        if macd > macd_signal:
            score += 0.75
            reasons.append("MACD support bullish")
        else:
            score -= 0.75
            reasons.append("MACD support bearish")

    if pd.notna(atr_pct) and atr_pct > 4.0:
        score *= 0.85
        reasons.append("High volatility penalty")

    direction = "UP" if score > 0.35 else "DOWN" if score < -0.35 else "SKIP"
    confidence = min(95, max(35, int(52 + abs(score) * 10 + abs(prophet_move_pct) * 2)))

    if abs(prophet_move_pct) < 0.35:
        confidence = max(35, confidence - 12)
        reasons.append("Tiny expected move")
    if trend_bias == 0 and abs(signal["score"]) <= 1:
        confidence = max(35, confidence - 10)
        reasons.append("Sideways setup")

    is_skipped = direction == "SKIP" or confidence < min_confidence

    return {
        "ok": True,
        "pred_price": pred_price,
        "pred_low": min(pred_low, pred_price),
        "pred_high": max(pred_high, pred_price),
        "prev_close": prev_close,
        "prophet_move_pct": prophet_move_pct,
        "hybrid_score": score,
        "direction": direction,
        "confidence": confidence,
        "skip": is_skipped,
        "signal": signal,
        "trend": trend_label,
        "atr_pct": atr_pct,
        "reasons": reasons[:6],
    }


def classify_hybrid_result(pred_price: float, actual_price: float, prev_close: float, skipped: bool) -> tuple[str, str, float, str]:
    pred_move = float(pred_price) - float(prev_close)
    actual_move = float(actual_price) - float(prev_close)
    error_pct = (abs(actual_price - pred_price) / actual_price * 100) if actual_price else np.nan
    direction_match = classify_direction(pred_move, actual_move)

    if skipped:
        return "Skipped", "gray", error_pct, direction_match
    if direction_match == "Yes" and pd.notna(error_pct) and error_pct <= 1.5:
        return "Strong True", "green", error_pct, direction_match
    if direction_match == "Yes" and pd.notna(error_pct) and error_pct <= 4.0:
        return "Near True", "green", error_pct, direction_match
    return "False", "red", error_pct, direction_match


def build_hybrid_prediction_review(data: pd.DataFrame, lookback_windows: int = 30, horizon: int = 1, min_confidence: int = HYBRID_MIN_CONFIDENCE) -> pd.DataFrame:
    market_df = add_indicators(data.copy()).dropna(subset=["Close"]).copy()
    market_df = market_df.sort_index()
    if len(market_df) < max(240, PREDICTION_MIN_ROWS + horizon + 10):
        return pd.DataFrame()

    rows = []
    max_windows = min(lookback_windows, len(market_df) - 220 - horizon)
    for step_back in range(max_windows, 0, -1):
        target_idx = len(market_df) - step_back
        future_idx = target_idx + horizon
        if future_idx >= len(market_df):
            continue

        train_df = market_df.iloc[:target_idx].copy()
        if len(train_df) < 220:
            continue

        decision = hybrid_decision_from_train(train_df, horizon=horizon, min_confidence=min_confidence)
        if not decision.get("ok"):
            continue

        prev_close = decision["prev_close"]
        actual_price = float(market_df.iloc[future_idx]["Close"])
        result, color, error_pct, direction_match = classify_hybrid_result(decision["pred_price"], actual_price, prev_close, decision["skip"])
        actual_move_pct = ((actual_price - prev_close) / prev_close * 100) if prev_close else np.nan

        rows.append({
            "Signal Date": train_df.index[-1],
            "Target Date": market_df.index[future_idx],
            "Horizon": f"{horizon}D",
            "Previous Close ₹": prev_close,
            "Predicted ₹": decision["pred_price"],
            "Lower ₹": decision["pred_low"],
            "Upper ₹": decision["pred_high"],
            "Actual ₹": actual_price,
            "Predicted Move %": decision["prophet_move_pct"],
            "Actual Move %": actual_move_pct,
            "Error %": error_pct,
            "Hybrid Score": decision["hybrid_score"],
            "Confidence %": decision["confidence"],
            "Trend": decision["trend"],
            "Direction": decision["direction"],
            "Direction Match": direction_match,
            "Result": result,
            "ResultColor": color,
            "Skip": "Yes" if decision["skip"] else "No",
            "Why": ", ".join(decision["reasons"]),
            "Type": "Hybrid Review",
        })

    return pd.DataFrame(rows)


def summarize_hybrid_accuracy(review_df: pd.DataFrame) -> dict:
    if review_df.empty:
        return {"eligible": 0, "skipped": 0, "strong": 0, "near": 0, "false": 0, "accuracy": 0.0, "avg_error": np.nan}
    eligible_df = review_df[review_df["Skip"] == "No"].copy()
    skipped = int((review_df["Skip"] == "Yes").sum())
    strong = int((eligible_df["Result"] == "Strong True").sum())
    near = int((eligible_df["Result"] == "Near True").sum())
    false = int((eligible_df["Result"] == "False").sum())
    eligible = len(eligible_df)
    accuracy = ((strong + near) / eligible * 100) if eligible else 0.0
    avg_error = float(pd.to_numeric(eligible_df.get("Error %"), errors="coerce").dropna().mean()) if eligible else np.nan
    return {"eligible": eligible, "skipped": skipped, "strong": strong, "near": near, "false": false, "accuracy": accuracy, "avg_error": avg_error}


def rolling_walk_forward_backtest(data: pd.DataFrame, horizons=(1, 3, 5), min_confidence: int = HYBRID_MIN_CONFIDENCE, max_windows: int = 90) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    detail_frames = []
    for horizon in horizons:
        review_df = build_hybrid_prediction_review(data, lookback_windows=max_windows, horizon=horizon, min_confidence=min_confidence)
        if review_df.empty:
            continue
        stats = summarize_hybrid_accuracy(review_df)
        eligible_df = review_df[review_df["Skip"] == "No"].copy()
        if not eligible_df.empty:
            detail_frames.append(eligible_df)
        summary_rows.append({
            "Horizon": f"{horizon}D",
            "Signals Taken": stats["eligible"],
            "Skipped": stats["skipped"],
            "Strong True": stats["strong"],
            "Near True": stats["near"],
            "False": stats["false"],
            "Accuracy %": round(stats["accuracy"], 2),
            "Avg Error %": round(stats["avg_error"], 2) if pd.notna(stats["avg_error"]) else np.nan,
        })
    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    return summary_df, detail_df


def style_hybrid_review(df: pd.DataFrame):
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
        if key == "skipped":
            return "background-color:#e5e7eb;color:#374151;font-weight:700;"
        return ""

    def color_skip(val):
        return "background-color:#e5e7eb;color:#374151;" if str(val) == "Yes" else ""

    return (
        df.style
        .map(color_result, subset=["Result"])
        .map(color_skip, subset=["Skip"])
        .format({
            "Previous Close ₹": "{:,.2f}",
            "Predicted ₹": "{:,.2f}",
            "Lower ₹": "{:,.2f}",
            "Upper ₹": "{:,.2f}",
            "Actual ₹": "{:,.2f}",
            "Predicted Move %": "{:,.2f}%",
            "Actual Move %": "{:,.2f}%",
            "Error %": "{:,.2f}%",
            "Hybrid Score": "{:,.2f}",
            "Confidence %": "{:,.0f}%",
        }, na_rep='-')
    )

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



# -----------------------------
# Ensemble feature engineering
# -----------------------------
def create_ml_feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["Ret1"] = df["Close"].pct_change(1)
    df["Ret3"] = df["Close"].pct_change(3)
    df["Ret5"] = df["Close"].pct_change(5)
    df["Ret10"] = df["Close"].pct_change(10)
    df["Lag1"] = df["Close"].shift(1)
    df["Lag2"] = df["Close"].shift(2)
    df["Lag3"] = df["Close"].shift(3)
    df["Lag5"] = df["Close"].shift(5)
    df["Lag10"] = df["Close"].shift(10)
    df["RollingMean5"] = df["Close"].rolling(5).mean()
    df["RollingMean10"] = df["Close"].rolling(10).mean()
    df["RollingMean20"] = df["Close"].rolling(20).mean()
    df["RollingStd5"] = df["Close"].rolling(5).std()
    df["RollingStd10"] = df["Close"].rolling(10).std()
    df["RollingStd20"] = df["Close"].rolling(20).std()
    df["RangePct"] = (df["High"] - df["Low"]) / df["Close"].replace(0, np.nan)
    df["OpenClosePct"] = (df["Close"] - df["Open"]) / df["Open"].replace(0, np.nan)
    df["VolumeChg1"] = df["Volume"].pct_change(1).replace([np.inf, -np.inf], np.nan)
    df["VolumeMean5"] = df["Volume"].rolling(5).mean()
    df["VolumeMean20"] = df["Volume"].rolling(20).mean()
    df["Target"] = df["Close"].shift(-1)
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    cols = [
        "Lag1", "Lag2", "Lag3", "Lag5", "Lag10",
        "Ret1", "Ret3", "Ret5", "Ret10",
        "RollingMean5", "RollingMean10", "RollingMean20",
        "RollingStd5", "RollingStd10", "RollingStd20",
        "SMA20", "SMA50", "SMA200",
        "EMA12", "EMA26", "RSI14", "MACD", "MACDSignal", "MACDHist",
        "BB_Mid", "BB_Upper", "BB_Lower", "ATR14",
        "RangePct", "OpenClosePct", "Volume", "VolumeChg1", "VolumeMean5", "VolumeMean20",
    ]
    return [c for c in cols if c in df.columns]


def safe_mape(actual: np.ndarray, pred: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denom = np.where(np.abs(actual) < 1e-9, np.nan, np.abs(actual))
    val = np.nanmean(np.abs(actual - pred) / denom) * 100
    if pd.isna(val):
        return 999.0
    return float(val)


def direction_accuracy(actual: np.ndarray, pred: np.ndarray, prev: np.ndarray) -> float:
    actual_move = np.sign(np.asarray(actual, dtype=float) - np.asarray(prev, dtype=float))
    pred_move = np.sign(np.asarray(pred, dtype=float) - np.asarray(prev, dtype=float))
    if len(actual_move) == 0:
        return 0.0
    return float(np.mean(actual_move == pred_move) * 100)


def clip_to_realistic_band(pred_value: float, current_price: float, hist_prices: pd.Series) -> float:
    daily_ret = hist_prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    ann_vol = float(daily_ret.std() * np.sqrt(252)) if not daily_ret.empty else 0.0
    cap_pct = min(FORECAST_CAP_MAX_PCT, max(FORECAST_CAP_MIN_PCT, ann_vol * FORECAST_CAP_SIGMA / np.sqrt(252)))
    floor_price = max(FORECAST_MIN_CLIP, current_price * (1 - cap_pct))
    ceiling_price = max(floor_price, current_price * (1 + cap_pct))
    return float(np.clip(pred_value, floor_price, ceiling_price))


def train_model_by_name(name: str):
    if name == "Linear Regression":
        return LinearRegression()
    if name == "Ridge Regression":
        return Ridge(alpha=1.0)
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=3, random_state=42)
    if name == "Gradient Boosting":
        return GradientBoostingRegressor(random_state=42, n_estimators=250, learning_rate=0.04, max_depth=2)
    raise ValueError(f"Unknown model: {name}")


def prophet_next_close(train_df: pd.DataFrame) -> float:
    tmp = train_df[["Close"]].reset_index().copy()
    date_col = tmp.columns[0]
    tmp = tmp.rename(columns={date_col: "ds", "Close": "y"})
    tmp["y"] = pd.to_numeric(tmp["y"], errors="coerce")
    tmp = tmp.dropna()
    tmp = tmp[tmp["y"] > 0]
    if len(tmp) < PREDICTION_MIN_ROWS:
        return float(tmp["y"].iloc[-1]) if not tmp.empty else FORECAST_MIN_CLIP
    log_df = tmp.copy()
    log_df["y"] = np.log(log_df["y"])
    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True, changepoint_prior_scale=0.12)
    model.fit(log_df)
    future = model.make_future_dataframe(periods=1)
    fc = model.predict(future)
    return float(np.exp(fc["yhat"].iloc[-1]))


def technical_vote_prediction(train_df: pd.DataFrame) -> tuple[float, str, float]:
    last = train_df.iloc[-1]
    current = float(last["Close"])
    score = 0
    if pd.notna(last.get("SMA20")) and current > float(last["SMA20"]):
        score += 1
    else:
        score -= 1
    if pd.notna(last.get("SMA50")) and current > float(last["SMA50"]):
        score += 1
    else:
        score -= 1
    if pd.notna(last.get("RSI14")):
        rsi = float(last["RSI14"])
        if rsi < 35:
            score += 1
        elif rsi > 68:
            score -= 1
    if pd.notna(last.get("MACD")) and pd.notna(last.get("MACDSignal")):
        score += 1 if float(last["MACD"]) >= float(last["MACDSignal"]) else -1

    atr = float(last["ATR14"]) if pd.notna(last.get("ATR14")) else current * 0.015
    bias = 0.35 * atr * score
    pred = max(FORECAST_MIN_CLIP, current + bias)
    direction = "Bullish" if pred > current else "Bearish" if pred < current else "Neutral"
    confidence = min(90.0, max(40.0, 50 + abs(score) * 10))
    return pred, direction, confidence


def backtest_ml_model(feature_df: pd.DataFrame, feature_cols: list, model_name: str, holdout_days: int) -> dict:
    valid_df = feature_df.dropna(subset=feature_cols + ["Target", "Close"]).copy()
    if len(valid_df) < BACKTEST_MIN_ROWS:
        return {"ok": False, "reason": "Not enough rows for backtest."}
    holdout = min(holdout_days, max(10, len(valid_df) // 5))
    train_df = valid_df.iloc[:-holdout].copy()
    test_df = valid_df.iloc[-holdout:].copy()
    if len(train_df) < 80:
        return {"ok": False, "reason": "Training window too small."}

    try:
        if model_name == "Prophet":
            preds, prevs, actuals = [], [], []
            for i in range(len(test_df)):
                train_slice_end = len(train_df) + i
                sub_train = valid_df.iloc[:train_slice_end].copy()
                pred = prophet_next_close(sub_train)
                pred = clip_to_realistic_band(pred, float(sub_train["Close"].iloc[-1]), sub_train["Close"])
                preds.append(pred)
                prevs.append(float(sub_train["Close"].iloc[-1]))
                actuals.append(float(valid_df.iloc[train_slice_end]["Target"]))
        elif model_name == "Technical Vote":
            preds, prevs, actuals = [], [], []
            for i in range(len(test_df)):
                train_slice_end = len(train_df) + i
                sub_train = valid_df.iloc[:train_slice_end].copy()
                pred, _, _ = technical_vote_prediction(sub_train)
                pred = clip_to_realistic_band(pred, float(sub_train["Close"].iloc[-1]), sub_train["Close"])
                preds.append(pred)
                prevs.append(float(sub_train["Close"].iloc[-1]))
                actuals.append(float(valid_df.iloc[train_slice_end]["Target"]))
        else:
            model = train_model_by_name(model_name)
            model.fit(train_df[feature_cols], train_df["Target"])
            preds = model.predict(test_df[feature_cols])
            prevs = test_df["Close"].values
            actuals = test_df["Target"].values
            preds = [clip_to_realistic_band(p, c, train_df["Close"]) for p, c in zip(preds, prevs)]

        preds = np.asarray(preds, dtype=float)
        actuals = np.asarray(actuals, dtype=float)
        prevs = np.asarray(prevs, dtype=float)
        mae = float(np.mean(np.abs(actuals - preds)))
        rmse = float(np.sqrt(np.mean((actuals - preds) ** 2)))
        mape = safe_mape(actuals, preds)
        dacc = direction_accuracy(actuals, preds, prevs)
        return {"ok": True, "mae": mae, "rmse": rmse, "mape": mape, "direction_accuracy": dacc, "actuals": actuals, "preds": preds, "prevs": prevs, "dates": test_df.index}
    except Exception as ex:
        return {"ok": False, "reason": str(ex)}


def build_ensemble(data: pd.DataFrame, holdout_days: int = MODEL_BACKTEST_DAYS_DEFAULT) -> dict:
    df = create_ml_feature_frame(add_indicators(data.copy()))
    feature_cols = get_feature_columns(df)
    valid_df = df.dropna(subset=feature_cols + ["Target", "Close"]).copy()
    if len(valid_df) < BACKTEST_MIN_ROWS:
        return {"ok": False, "reason": "Not enough clean rows for ensemble. Try a stock with longer history."}
    if not SKLEARN_OK:
        return {"ok": False, "reason": "scikit-learn is not installed in this environment."}

    model_names = ["Prophet", "Linear Regression", "Ridge Regression", "Random Forest", "Gradient Boosting", "Technical Vote"]
    rows = []
    current_close = float(valid_df["Close"].iloc[-1])
    hist_prices = valid_df["Close"]

    for model_name in model_names:
        bt = backtest_ml_model(valid_df, feature_cols, model_name, holdout_days)
        if not bt.get("ok"):
            continue
        try:
            if model_name == "Prophet":
                pred = prophet_next_close(valid_df)
                pred = clip_to_realistic_band(pred, current_close, hist_prices)
                conf = max(35.0, min(90.0, 100 - bt["mape"] + (bt["direction_accuracy"] - 50) * 0.3))
            elif model_name == "Technical Vote":
                pred, _, tech_conf = technical_vote_prediction(valid_df)
                pred = clip_to_realistic_band(pred, current_close, hist_prices)
                conf = (tech_conf * 0.4) + (max(40.0, 100 - bt["mape"]) * 0.3) + (bt["direction_accuracy"] * 0.3)
            else:
                model = train_model_by_name(model_name)
                model.fit(valid_df[feature_cols], valid_df["Target"])
                X_last = valid_df[feature_cols].iloc[[-1]]
                pred = float(model.predict(X_last)[0])
                pred = clip_to_realistic_band(pred, current_close, hist_prices)
                conf = max(35.0, min(92.0, 100 - bt["mape"] + (bt["direction_accuracy"] - 50) * 0.35))

            move_pct = ((pred - current_close) / current_close * 100) if current_close else 0.0
            direction = "Bullish" if pred > current_close else "Bearish" if pred < current_close else "Neutral"
            weight = (1.0 / max(bt["mape"], 0.5)) * (0.6 + bt["direction_accuracy"] / 100)
            rows.append({"Model": model_name, "Predicted ₹": pred, "Move %": move_pct, "Direction": direction, "MAPE %": bt["mape"], "RMSE": bt["rmse"], "Dir Acc %": bt["direction_accuracy"], "Confidence %": conf, "Weight": weight})
        except Exception:
            continue

    result_df = pd.DataFrame(rows)
    if result_df.empty or len(result_df) < ENSEMBLE_MIN_MODELS:
        return {"ok": False, "reason": "Could not produce enough successful model outputs for ensemble."}

    result_df["NormWeight"] = result_df["Weight"] / result_df["Weight"].sum()
    final_pred = float((result_df["Predicted ₹"] * result_df["NormWeight"]).sum())
    final_move_pct = ((final_pred - current_close) / current_close * 100) if current_close else 0.0
    final_direction = "Bullish" if final_pred > current_close else "Bearish" if final_pred < current_close else "Neutral"
    agreement = float(max((result_df["Direction"] == final_direction).mean() * 100, 0.0))
    ensemble_conf = float(np.clip((result_df["Confidence %"] * result_df["NormWeight"]).sum() * 0.65 + agreement * 0.35, 35, 95))

    return {"ok": True, "models_df": result_df.sort_values(["Dir Acc %", "MAPE %"], ascending=[False, True]).reset_index(drop=True), "current_close": current_close, "final_pred": final_pred, "final_move_pct": final_move_pct, "final_direction": final_direction, "agreement_pct": agreement, "ensemble_confidence": ensemble_conf}


@st.cache_data(show_spinner=False, ttl=600)
def build_ensemble_comparison_snapshot(symbols: tuple, holdout_days: int = MODEL_BACKTEST_DAYS_DEFAULT) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        try:
            d, _, _ = fetch_stock_data(sym)
            if d.empty or len(d) < BACKTEST_MIN_ROWS:
                continue
            ens = build_ensemble(d, holdout_days=holdout_days)
            if not ens.get("ok"):
                continue
            rows.append({"Symbol": sym, "Current Close": round(float(ens["current_close"]), 2), "Ensemble Pred": round(float(ens["final_pred"]), 2), "Move %": round(float(ens["final_move_pct"]), 2), "Direction": ens["final_direction"], "Agreement %": round(float(ens["agreement_pct"]), 2), "Ensemble Confidence %": round(float(ens["ensemble_confidence"]), 2), "Models Used": int(len(ens["models_df"]))})
        except Exception:
            continue
    return pd.DataFrame(rows)



def build_ai_insight(symbol: str, current_price: float, signal: dict, forecast_tail: pd.DataFrame | None, risk_name: str, volatility: float, levels: dict, cap_meta: dict | None = None, ensemble_result: dict | None = None) -> str:
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
    ensemble_text = ""
    if ensemble_result and ensemble_result.get("ok"):
        ensemble_text = (
            f" Weighted ensemble output suggests ₹ {fmt_num(ensemble_result['final_pred'])} with {ensemble_result['ensemble_confidence']:.1f}% confidence, "
            f"{ensemble_result['agreement_pct']:.1f}% model agreement, and a {ensemble_result['final_direction'].lower()} bias."
        )

    return (
        f"For {symbol}, the technical signal is {signal['label']} with {signal['confidence']}% confidence. "
        f"Key drivers are: {reasons}. {move_text}{reliability_text}{ensemble_text} Annualized volatility is {volatility:.2f}%, so risk is classified as {risk_name}. "
        f"{support_text}, and {resistance_text}. This is a model-assisted summary, not trading advice."
    )


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

st.sidebar.markdown("### Stock Selection")
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
model_backtest_days = st.sidebar.slider("Ensemble Backtest Days", min_value=15, max_value=60, value=MODEL_BACKTEST_DAYS_DEFAULT)
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

st.sidebar.markdown("### Live Dashboard Settings")
st.session_state.live_auto_refresh = st.sidebar.toggle(
    "Auto Refresh Live Dashboard",
    value=st.session_state.live_auto_refresh
)

st.session_state.live_refresh_sec = st.sidebar.selectbox(
    "Refresh Interval (sec)",
    options=[15, 30, 60, 120, 300],
    index=[15, 30, 60, 120, 300].index(st.session_state.live_refresh_sec if st.session_state.live_refresh_sec in [15, 30, 60, 120, 300] else 60)
)

# -----------------------------
# On-load live dashboard
# -----------------------------
refresh_counter = 0
if st.session_state.live_auto_refresh and AUTO_REFRESH_OK:
    refresh_counter = st_autorefresh(
        interval=int(st.session_state.live_refresh_sec) * 1000,
        key="live_dash_refresh"
    )
elif st.session_state.live_auto_refresh and not AUTO_REFRESH_OK:
    st.info("Auto refresh package not installed. Install 'streamlit-autorefresh' to enable live auto refresh.")

if refresh_counter:
    try:
        fetch_intraday_data.clear()
        build_live_symbol_snapshot.clear()
        build_market_breadth_snapshot.clear()
    except Exception:
        pass

render_live_dashboard_home(symbol)

st.markdown("#### 📊 Market Overview")
o1, o2, o3, o4 = st.columns(4)

nifty_live = build_live_symbol_snapshot(("^NSEI",))
bank_live = build_live_symbol_snapshot(("^NSEBANK",))
sensex_live = build_live_symbol_snapshot(("^BSESN",))
sel_live = build_live_symbol_snapshot((symbol,))

def metric_from_df(col, title, df):
    if df.empty:
        col.metric(title, "-", "-")
    else:
        r = df.iloc[0]
        col.metric(title, fmt_num(r["LTP"]), f"{r['Change']:+.2f} ({r['Change %']:+.2f}%)")

metric_from_df(o1, "NIFTY 50", nifty_live)
metric_from_df(o2, "BANK NIFTY", bank_live)
metric_from_df(o3, "SENSEX", sensex_live)
metric_from_df(o4, "ACTIVE SYMBOL", sel_live)

# -----------------------------
# Main
# -----------------------------
should_run_main_analysis = run_btn or ("auto_first_load_done" not in st.session_state)

if "auto_first_load_done" not in st.session_state:
    st.session_state.auto_first_load_done = True

if should_run_main_analysis:
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

        ensemble_result = None
        if has_indicator_data:
            ensemble_result = build_ensemble(data, holdout_days=model_backtest_days)

        if has_indicator_data:
            ai_summary = build_ai_insight(symbol, current_price, signal, future_rows, risk_name, volatility, levels, forecast.attrs.get("cap_meta", {}), ensemble_result)
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
        render_terminal_toolbar(symbol, len(compare_symbols), source_label)
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

        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(
            ["Market Watch", "Chart Lab", "Forecast Lab", "Model Explorer", "Backtest Lab", "Fundamentals", "Watchlists", "Analysis Explorer", "News Feed", "Portfolio Terminal"]
        )

        with tab1:
            market_watch_universe = tuple(dict.fromkeys(compare_symbols + st.session_state.watchlist + [symbol, "SBIN.NS", "ICICIBANK.NS", "TCS.NS", "INFY.NS", "RELIANCE.NS"]))
            scanner_df = build_scanner_snapshot(market_watch_universe)
            wl_df = build_watchlist_snapshot(tuple(st.session_state.watchlist)) if st.session_state.watchlist else pd.DataFrame()
            heat_df = build_symbol_heatmap(market_watch_universe)

            render_hero_metrics(current_price, day_change, day_change_pct, high_52w, low_52w, avg_volume)

            left_col, center_col, right_col = st.columns([1.12, 3.95, 1.45], gap="large")

            with left_col:
                render_market_watch_panel(symbol, st.session_state.watchlist)
                render_table_card('Quick Market Watch', wl_df, max_rows=8)
                render_table_card('Explorer Scan', scanner_df, max_rows=10)

            with center_col:
                st.markdown("<div class='section-card'>", unsafe_allow_html=True)
                st.markdown(f"<div class='dock-title'>Chart Workspace</div>", unsafe_allow_html=True)
                st.markdown(f"### 📈 {symbol} Main Chart Workspace")
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
                if has_indicator_data:
                    fig.add_trace(go.Scatter(x=data.index, y=data["SMA20"], mode="lines", name="SMA20", line=dict(width=2)))
                    fig.add_trace(go.Scatter(x=data.index, y=data["SMA50"], mode="lines", name="SMA50", line=dict(width=2)))
                    fig.add_trace(go.Scatter(x=data.index, y=data["SMA200"], mode="lines", name="SMA200", line=dict(width=2.2)))

                if levels["support"] is not None:
                    fig.add_hline(
                        y=levels["support"],
                        annotation_text=f"Support ₹ {fmt_num(levels['support'])}",
                        line_dash="dot",
                        line_width=1.2
                    )
                if levels["resistance"] is not None:
                    fig.add_hline(
                        y=levels["resistance"],
                        annotation_text=f"Resistance ₹ {fmt_num(levels['resistance'])}",
                        line_dash="dot",
                        line_width=1.2
                    )

                title_suffix = "with Moving Averages" if has_indicator_data else "(Limited Data Mode)"
                fig.update_layout(
                    title=f"{symbol} Historical Price {title_suffix}",
                    height=670,
                    xaxis_title="Date",
                    yaxis_title="Price",
                    xaxis_rangeslider_visible=False,
                    template="plotly_dark",
                    plot_bgcolor="rgba(6,16,31,0.82)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=18, r=18, t=60, b=20),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                fig.update_xaxes(gridcolor="rgba(148,163,184,0.08)", zerolinecolor="rgba(148,163,184,0.08)")
                fig.update_yaxes(gridcolor="rgba(148,163,184,0.08)", zerolinecolor="rgba(148,163,184,0.08)")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with right_col:
                render_info_panel(current_price, signal, risk_name, levels, fundamentals)
                render_signal_reason_panel(signal)

            st.markdown("<div class='dock-title' style='margin-top:8px;'>Support / Resistance Desk</div>", unsafe_allow_html=True)
            render_stat_cards([
                ("Nearest Support", f"₹ {fmt_num(levels['support'])}" if levels["support"] is not None else "-", "Closest demand area"),
                ("Nearest Resistance", f"₹ {fmt_num(levels['resistance'])}" if levels["resistance"] is not None else "-", "Closest supply area"),
                ("Suggested Stop Loss", f"₹ {fmt_num(levels['stop_loss'])}" if levels["stop_loss"] is not None else "-", "Risk-control reference"),
                ("Breakout Zone", f"₹ {fmt_num(levels['breakout'])}" if levels["breakout"] is not None else "-", "Potential strength trigger"),
            ])

            mini1, mini2 = st.columns(2, gap="large")
            with mini1:
                render_table_card('Relative Strength / Heatmap', heat_df, max_rows=10)
            with mini2:
                render_table_card('Explorer / Scan Results', scanner_df, max_rows=10)


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
                    true_df = past_review_df[past_review_df["Result"].isin(["Strong True", "Near True"])].copy()
                    false_df = past_review_df[past_review_df["Result"] == "False"].copy()

                    if not true_df.empty:
                        fig2.add_trace(
                            go.Scatter(
                                x=true_df["Date"],
                                y=true_df["Actual ₹"],
                                mode="markers",
                                name="Past Prediction Correct",
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
                                    "Status: Correct<extra></extra>"
                                ),
                            )
                        )

                    if not false_df.empty:
                        fig2.add_trace(
                            go.Scatter(
                                x=false_df["Date"],
                                y=false_df["Actual ₹"],
                                mode="markers",
                                name="Past Prediction Wrong",
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
                                    "Status: Wrong<extra></extra>"
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
                    strong_true_count = int((past_review_df["Result"] == "Strong True").sum())
                    near_true_count = int((past_review_df["Result"] == "Near True").sum())
                    false_count = int((past_review_df["Result"] == "False").sum())
                    true_like_count = strong_true_count + near_true_count
                    accuracy_pct = (true_like_count / len(past_review_df) * 100) if len(past_review_df) else 0
                    avg_error_pct = float(pd.to_numeric(past_review_df["Error %"], errors="coerce").dropna().mean()) if "Error %" in past_review_df.columns else float("nan")

                    r1, r2, r3, r4, r5 = st.columns(5)
                    r1.metric("Reviewed Days", len(past_review_df))
                    r2.metric("Strong True", strong_true_count)
                    r3.metric("Near True", near_true_count)
                    r4.metric("False", false_count)
                    r5.metric("Accuracy", f"{accuracy_pct:.2f}%")

                    if pd.notna(avg_error_pct):
                        st.caption(f"Average prediction error: {avg_error_pct:.2f}%")

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

                st.markdown("### 🚀 Hybrid Accuracy Engine")
                hybrid_horizon = st.radio("Hybrid review horizon", options=[1, 3, 5], horizontal=True, key="hybrid_horizon_forecast")
                hybrid_review_df = build_hybrid_prediction_review(data, lookback_windows=min(HYBRID_REVIEW_MAX_WINDOWS, max(20, days * 3)), horizon=hybrid_horizon, min_confidence=HYBRID_MIN_CONFIDENCE)
                if hybrid_review_df.empty:
                    st.info("Not enough historical data to build hybrid review yet.")
                else:
                    hybrid_stats = summarize_hybrid_accuracy(hybrid_review_df)
                    h1, h2, h3, h4, h5, h6 = st.columns(6)
                    h1.metric("Signals Taken", hybrid_stats["eligible"])
                    h2.metric("Skipped", hybrid_stats["skipped"])
                    h3.metric("Strong True", hybrid_stats["strong"])
                    h4.metric("Near True", hybrid_stats["near"])
                    h5.metric("False", hybrid_stats["false"])
                    h6.metric("Accuracy", f"{hybrid_stats['accuracy']:.2f}%")
                    if pd.notna(hybrid_stats["avg_error"]):
                        st.caption(f"Average error on taken signals: {hybrid_stats['avg_error']:.2f}% | Low-confidence setups are automatically skipped below {HYBRID_MIN_CONFIDENCE}% confidence.")

                    next_decision = hybrid_decision_from_train(data.copy(), horizon=hybrid_horizon, min_confidence=HYBRID_MIN_CONFIDENCE)
                    if next_decision.get("ok"):
                        nh1, nh2, nh3, nh4 = st.columns(4)
                        label = "SKIP" if next_decision["skip"] else next_decision["direction"]
                        nh1.metric("Next Hybrid Call", label)
                        nh2.metric("Confidence", f"{next_decision['confidence']}%")
                        nh3.metric("Hybrid Score", f"{next_decision['hybrid_score']:.2f}")
                        nh4.metric("Trend Filter", next_decision["trend"])
                        st.caption("Why: " + ", ".join(next_decision["reasons"]))

                    display_cols = ["Signal Date", "Target Date", "Horizon", "Previous Close ₹", "Predicted ₹", "Actual ₹", "Predicted Move %", "Actual Move %", "Error %", "Confidence %", "Trend", "Direction", "Direction Match", "Result", "Skip", "Why"]
                    st.dataframe(style_hybrid_review(hybrid_review_df[display_cols]), use_container_width=True, height=430)

        with tab4:
            st.subheader("🧠 Ensemble Model Comparison")
            if not has_indicator_data:
                st.info("Ensemble needs enough indicator history to run.")
            elif not SKLEARN_OK:
                st.warning("scikit-learn is not installed, so the ensemble models cannot run in this environment.")
            elif not ensemble_result or not ensemble_result.get("ok"):
                st.warning(ensemble_result.get("reason", "Ensemble result is not available right now.") if isinstance(ensemble_result, dict) else "Ensemble result is not available right now.")
            else:
                e1, e2, e3, e4, e5 = st.columns(5)
                e1.metric("Current Close", f"₹ {fmt_num(ensemble_result['current_close'])}")
                e2.metric("Final Ensemble Pred", f"₹ {fmt_num(ensemble_result['final_pred'])}")
                e3.metric("Move %", f"{ensemble_result['final_move_pct']:.2f}%")
                e4.metric("Direction", ensemble_result["final_direction"])
                e5.metric("Confidence", f"{ensemble_result['ensemble_confidence']:.2f}%")

                ea1, ea2 = st.columns(2)
                ea1.metric("Model Agreement", f"{ensemble_result['agreement_pct']:.2f}%")
                ea2.metric("Models Used", len(ensemble_result["models_df"]))

                st.dataframe(
                    ensemble_result["models_df"][["Model", "Predicted ₹", "Move %", "Direction", "MAPE %", "RMSE", "Dir Acc %", "Confidence %", "NormWeight"]].round(2),
                    use_container_width=True,
                    height=360,
                )

                fig_ens = go.Figure()
                mdf = ensemble_result["models_df"].copy()
                fig_ens.add_trace(go.Bar(x=mdf["Model"], y=mdf["Predicted ₹"], name="Predicted Price"))
                fig_ens.add_trace(go.Scatter(x=mdf["Model"], y=mdf["Dir Acc %"], mode="lines+markers", name="Direction Accuracy %", yaxis="y2"))
                fig_ens.update_layout(
                    title="Model-wise Predicted Price vs Direction Accuracy",
                    xaxis_title="Model",
                    yaxis_title="Predicted Price",
                    yaxis2=dict(title="Direction Accuracy %", overlaying="y", side="right"),
                    height=460
                )
                st.plotly_chart(fig_ens, use_container_width=True)


        with tab5:
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

                st.markdown("### Hybrid Walk-Forward Backtest")
                summary_df, detail_df = rolling_walk_forward_backtest(data, horizons=(1, 3, 5), min_confidence=HYBRID_MIN_CONFIDENCE, max_windows=90)
                if summary_df.empty:
                    st.info("Hybrid walk-forward backtest is not available yet for this symbol.")
                else:
                    st.dataframe(summary_df, use_container_width=True)
                    fig_hbt = go.Figure()
                    fig_hbt.add_trace(go.Bar(x=summary_df["Horizon"], y=summary_df["Accuracy %"], name="Accuracy %"))
                    fig_hbt.update_layout(title="Hybrid Accuracy by Horizon", xaxis_title="Horizon", yaxis_title="Accuracy %", height=420)
                    st.plotly_chart(fig_hbt, use_container_width=True)

                    if not detail_df.empty:
                        st.markdown("#### Signal Details")
                        detail_cols = ["Signal Date", "Target Date", "Horizon", "Predicted ₹", "Actual ₹", "Error %", "Confidence %", "Trend", "Direction", "Direction Match", "Result", "Skip", "Why"]
                        st.dataframe(style_hybrid_review(detail_df[detail_cols]), use_container_width=True, height=420)
            else:
                st.info("Backtest is hidden from sidebar settings.")

        with tab6:
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

        with tab7:
            st.subheader("⭐ Watchlist Dashboard")
            if not st.session_state.watchlist:
                st.info("Your watchlist is empty.")
            else:
                wl_df = build_watchlist_snapshot(tuple(st.session_state.watchlist))
                if wl_df.empty:
                    st.warning("Watchlist data is not available right now.")
                else:
                    st.dataframe(wl_df, use_container_width=True)

        with tab8:
            st.subheader("⚖️ Analysis Explorer")
            scan_universe = tuple(dict.fromkeys(compare_symbols + st.session_state.watchlist + [symbol, "SBIN.NS", "ICICIBANK.NS", "TCS.NS", "INFY.NS", "RELIANCE.NS"]))
            scan_df = build_scanner_snapshot(scan_universe)
            if not scan_df.empty:
                sx1, sx2, sx3 = st.columns(3)
                sx1.metric("Bullish Setups", int((scan_df["Trend"] == "Bullish").sum()))
                sx2.metric("Breakout Signals", int((scan_df["Signal"] == "Breakout").sum()))
                sx3.metric("Avg RSI", f"{scan_df['RSI'].dropna().mean():.2f}" if scan_df['RSI'].dropna().size else "-")
                st.dataframe(scan_df, use_container_width=True, height=240)
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

                    st.markdown("### Ensemble Comparison")
                    ensemble_cmp_df = build_ensemble_comparison_snapshot(tuple(compare_symbols), holdout_days=model_backtest_days)
                    if ensemble_cmp_df.empty:
                        st.info("Ensemble comparison is not available for one or more selected stocks yet.")
                    else:
                        st.dataframe(ensemble_cmp_df, use_container_width=True)

                    corr_df = build_corr_matrix(tuple(compare_symbols))
                    if not corr_df.empty:
                        fig_corr = go.Figure(data=go.Heatmap(z=corr_df.values, x=corr_df.columns, y=corr_df.index, text=corr_df.values, texttemplate="%{text}", hovertemplate="%{x} vs %{y}: %{z}<extra></extra>"))
                        fig_corr.update_layout(title="Correlation Matrix", height=480)
                        st.plotly_chart(fig_corr, use_container_width=True)

        with tab9:
            st.subheader("📰 News Sentiment")
            if news_df.empty:
                st.info("Recent news not available right now.")
            else:
                st.dataframe(news_df[["Published", "Title", "Publisher", "Sentiment", "SentimentScore"]], use_container_width=True)

        with tab10:
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
