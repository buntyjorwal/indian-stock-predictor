import math
import re
import time
from datetime import date, datetime, timedelta
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

# Page Configuration
st.set_page_config(
    page_title="NSE Market Intelligence Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Professional UI Components
# -----------------------------
def inject_professional_css() -> None:
    st.markdown("""
        <style>
        /* Modern Professional Theme */
        :root {
            --primary: #0f172a;
            --primary-light: #1e293b;
            --primary-dark: #020617;
            --accent: #3b82f6;
            --accent-light: #60a5fa;
            --accent-gradient: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --border-color: rgba(51, 65, 85, 0.5);
            --card-bg: linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 0.98) 100%);
            --glass-bg: rgba(15, 23, 42, 0.7);
            --glass-border: rgba(59, 130, 246, 0.2);
        }

        /* Global Styles */
        .stApp {
            background: linear-gradient(135deg, #020617 0%, #0f172a 100%);
            color: var(--text-primary);
        }

        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: rgba(15, 23, 42, 0.5);
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-light) 100%);
            border-radius: 4px;
        }

        /* Card Components */
        .glass-card {
            background: var(--glass-bg);
            backdrop-filter: blur(10px);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 16px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        }

        .metric-card {
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px;
            transition: all 0.3s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 24px rgba(59, 130, 246, 0.15);
            border-color: var(--accent);
        }

        /* Header Styles */
        .terminal-header {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
            border-bottom: 1px solid var(--accent);
            padding: 16px 24px;
            margin-bottom: 24px;
            border-radius: 0 0 16px 16px;
        }

        .header-title {
            font-size: 1.8rem;
            font-weight: 700;
            background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
        }

        /* Status Indicators */
        .status-badge {
            display: inline-flex;
            align-items: center;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .status-live {
            background: rgba(16, 185, 129, 0.2);
            color: var(--success);
            border: 1px solid var(--success);
        }
        .status-bullish {
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
        }
        .status-bearish {
            background: rgba(239, 68, 68, 0.15);
            color: #f87171;
        }
        .status-neutral {
            background: rgba(245, 158, 11, 0.15);
            color: #fbbf24;
        }

        /* Data Tables */
        .data-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            font-size: 13px;
        }
        .data-table th {
            background: rgba(30, 41, 59, 0.8);
            color: var(--text-secondary);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-size: 11px;
            padding: 12px;
            border-bottom: 2px solid var(--accent);
        }
        .data-table td {
            padding: 10px 12px;
            border-bottom: 1px solid var(--border-color);
            color: var(--text-primary);
        }
        .data-table tr:hover td {
            background: rgba(59, 130, 246, 0.05);
        }

        /* Chart Container */
        .chart-container {
            background: var(--glass-bg);
            border-radius: 12px;
            padding: 16px;
            border: 1px solid var(--border-color);
            margin-bottom: 16px;
        }

        /* AI Insight Box */
        .ai-insight {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
            border-left: 4px solid var(--accent);
            border-radius: 12px;
            padding: 20px;
            margin: 16px 0;
        }
        .ai-insight-title {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 16px;
            font-weight: 700;
            color: var(--accent-light);
            margin-bottom: 12px;
        }

        /* Button Styles */
        .stButton > button {
            background: var(--accent-gradient);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: 600;
            transition: all 0.3s ease;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 16px rgba(59, 130, 246, 0.3);
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #020617 100%);
            border-right: 1px solid var(--border-color);
        }
        section[data-testid="stSidebar"] .stMarkdown {
            color: var(--text-primary);
        }

        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 8px 8px 0 0;
            padding: 10px 20px;
            color: var(--text-secondary);
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background: var(--accent-gradient);
            color: white;
        }

        /* Metric Value Styling */
        [data-testid="stMetricValue"] {
            font-size: 24px;
            font-weight: 700;
            color: var(--text-primary);
        }
        [data-testid="stMetricDelta"] {
            font-size: 14px;
        }

        /* Progress Bar */
        .stProgress > div > div {
            background: var(--accent-gradient);
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .header-title {
                font-size: 1.4rem;
            }
            .metric-card {
                padding: 12px;
            }
        }

        /* Animation */
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .live-indicator {
            animation: pulse 2s infinite;
        }

        /* Tooltip */
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: help;
        }
        .tooltip .tooltip-text {
            visibility: hidden;
            background: var(--primary-dark);
            color: var(--text-primary);
            text-align: center;
            padding: 8px 12px;
            border-radius: 6px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            white-space: nowrap;
            font-size: 12px;
            border: 1px solid var(--border-color);
        }
        .tooltip:hover .tooltip-text {
            visibility: visible;
        }
        </style>
    """, unsafe_allow_html=True)

def professional_header() -> None:
    st.markdown("""
        <div class="terminal-header">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1 class="header-title">📊 NSE Market Intelligence Terminal</h1>
                    <p style="color: var(--text-secondary); margin: 8px 0 0 0; font-size: 14px;">
                        Advanced Analytics • AI-Powered Predictions • Real-time Market Data
                    </p>
                </div>
                <div style="display: flex; gap: 12px;">
                    <span class="status-badge status-live">
                        <span style="display: inline-block; width: 8px; height: 8px; background: var(--success); border-radius: 50%; margin-right: 6px;"></span>
                        LIVE MARKET
                    </span>
                    <span class="status-badge" style="background: rgba(59, 130, 246, 0.2); color: var(--accent-light); border: 1px solid var(--accent);">
                        {timestamp}
                    </span>
                </div>
            </div>
        </div>
    """.format(timestamp=datetime.now().strftime("%d-%b-%Y %H:%M")), unsafe_allow_html=True)

def render_metric_card(title: str, value: str, delta: Optional[str] = None, 
                      delta_color: str = "normal", help_text: Optional[str] = None) -> None:
    """Render a professional metric card with tooltip."""
    delta_html = f'<span style="color: {"var(--success)" if delta and "+" in delta else "var(--danger)" if delta and "-" in delta else "var(--text-secondary)"}; font-size: 13px;">{delta}</span>' if delta else ""
    
    help_attr = f'<span class="tooltip">ⓘ<span class="tooltip-text">{help_text}</span></span>' if help_text else ""
    
    st.markdown(f"""
        <div class="metric-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="color: var(--text-secondary); font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">{title}</span>
                {help_attr}
            </div>
            <div style="font-size: 28px; font-weight: 700; color: var(--text-primary); margin-bottom: 4px;">{value}</div>
            {delta_html}
        </div>
    """, unsafe_allow_html=True)

def render_signal_badge(signal: str, confidence: float) -> None:
    """Render a professional signal badge."""
    signal_lower = signal.lower()
    if "buy" in signal_lower:
        badge_class = "status-bullish"
        icon = "📈"
    elif "sell" in signal_lower:
        badge_class = "status-bearish"
        icon = "📉"
    else:
        badge_class = "status-neutral"
        icon = "⏸️"
    
    st.markdown(f"""
        <div class="status-badge {badge_class}" style="font-size: 14px; padding: 8px 16px;">
            {icon} {signal} • {confidence:.0f}% Confidence
        </div>
    """, unsafe_allow_html=True)

# -----------------------------
# Enhanced Data Fetching
# -----------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_enhanced_stock_data(symbol: str, period: str = "2y") -> Tuple[pd.DataFrame, str, Dict]:
    """Enhanced data fetching with better error handling and caching."""
    try:
        # Primary source: yfinance
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval="1d")
        
        if data.empty:
            # Try fallback period
            data = ticker.history(period="1y", interval="1d")
        
        if data.empty:
            return pd.DataFrame(), "No data available", {}
        
        # Clean and validate data
        data = data[["Open", "High", "Low", "Close", "Volume"]].copy()
        data = data.dropna()
        
        # Calculate additional metrics
        info = ticker.info if hasattr(ticker, "info") else {}
        
        source = "Yahoo Finance"
        metadata = {
            "currency": info.get("currency", "INR"),
            "exchange": info.get("exchange", "NSE"),
            "market_cap": info.get("marketCap", None),
            "sector": info.get("sector", ""),
        }
        
        return data, source, metadata
        
    except Exception as e:
        return pd.DataFrame(), f"Error: {str(e)}", {}

@st.cache_data(ttl=60, show_spinner=False)
def fetch_live_quote(symbol: str) -> Dict[str, Any]:
    """Fetch live quote data with fallback."""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d", interval="1m")
        
        if data.empty:
            return {}
        
        latest = data.iloc[-1]
        prev_close = data.iloc[0]["Close"] if len(data) > 1 else latest["Close"]
        
        return {
            "price": float(latest["Close"]),
            "change": float(latest["Close"] - prev_close),
            "change_pct": float((latest["Close"] - prev_close) / prev_close * 100) if prev_close else 0,
            "high": float(data["High"].max()),
            "low": float(data["Low"].min()),
            "volume": float(data["Volume"].sum()),
            "timestamp": datetime.now(),
        }
    except Exception:
        return {}

# -----------------------------
# Advanced Technical Indicators
# -----------------------------
def calculate_advanced_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive technical indicators."""
    df = data.copy()
    
    # Basic Moving Averages
    for period in [5, 10, 20, 50, 100, 200]:
        df[f"SMA_{period}"] = df["Close"].rolling(window=period).mean()
        df[f"EMA_{period}"] = df["Close"].ewm(span=period, adjust=False).mean()
    
    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]
    
    # Bollinger Bands
    df["BB_Middle"] = df["Close"].rolling(window=20).mean()
    bb_std = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
    df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
    
    # ATR
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df["ATR"] = true_range.rolling(14).mean()
    
    # Volume Indicators
    df["Volume_SMA"] = df["Volume"].rolling(window=20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA"]
    df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
    
    # Stochastic Oscillator
    low_14 = df["Low"].rolling(window=14).min()
    high_14 = df["High"].rolling(window=14).max()
    df["Stoch_K"] = 100 * ((df["Close"] - low_14) / (high_14 - low_14))
    df["Stoch_D"] = df["Stoch_K"].rolling(window=3).mean()
    
    # Ichimoku Cloud
    high_9 = df["High"].rolling(window=9).max()
    low_9 = df["Low"].rolling(window=9).min()
    df["Ichimoku_Conversion"] = (high_9 + low_9) / 2
    
    high_26 = df["High"].rolling(window=26).max()
    low_26 = df["Low"].rolling(window=26).min()
    df["Ichimoku_Base"] = (high_26 + low_26) / 2
    
    df["Ichimoku_SpanA"] = ((df["Ichimoku_Conversion"] + df["Ichimoku_Base"]) / 2).shift(26)
    df["Ichimoku_SpanB"] = ((df["High"].rolling(window=52).max() + df["Low"].rolling(window=52).min()) / 2).shift(26)
    
    # Price channels
    df["Donchian_High"] = df["High"].rolling(window=20).max()
    df["Donchian_Low"] = df["Low"].rolling(window=20).min()
    df["Donchian_Mid"] = (df["Donchian_High"] + df["Donchian_Low"]) / 2
    
    # Volatility
    df["Returns"] = df["Close"].pct_change()
    df["Volatility"] = df["Returns"].rolling(window=20).std() * np.sqrt(252)
    
    return df

def generate_advanced_signal(data: pd.DataFrame) -> Dict[str, Any]:
    """Generate advanced trading signal with confidence score."""
    if len(data) < 50:
        return {"signal": "INSUFFICIENT DATA", "confidence": 0, "reasons": ["Not enough data"]}
    
    last = data.iloc[-1]
    prev = data.iloc[-2] if len(data) > 1 else last
    
    signals = []
    confidence_score = 0
    
    # Trend Analysis
    if last["Close"] > last["SMA_20"] > last["SMA_50"]:
        signals.append(("Bullish", 2, "Price above moving averages"))
        confidence_score += 2
    elif last["Close"] < last["SMA_20"] < last["SMA_50"]:
        signals.append(("Bearish", -2, "Price below moving averages"))
        confidence_score -= 2
    
    # RSI Analysis
    if last["RSI"] < 30:
        signals.append(("Bullish", 1.5, "RSI oversold"))
        confidence_score += 1.5
    elif last["RSI"] > 70:
        signals.append(("Bearish", -1.5, "RSI overbought"))
        confidence_score -= 1.5
    
    # MACD Analysis
    if last["MACD"] > last["MACD_Signal"] and prev["MACD"] <= prev["MACD_Signal"]:
        signals.append(("Bullish", 2, "MACD bullish crossover"))
        confidence_score += 2
    elif last["MACD"] < last["MACD_Signal"] and prev["MACD"] >= prev["MACD_Signal"]:
        signals.append(("Bearish", -2, "MACD bearish crossover"))
        confidence_score -= 2
    
    # Bollinger Bands
    if last["Close"] < last["BB_Lower"]:
        signals.append(("Bullish", 1, "Price below lower Bollinger Band"))
        confidence_score += 1
    elif last["Close"] > last["BB_Upper"]:
        signals.append(("Bearish", -1, "Price above upper Bollinger Band"))
        confidence_score -= 1
    
    # Volume Analysis
    if last["Volume_Ratio"] > 1.5 and last["Close"] > prev["Close"]:
        signals.append(("Bullish", 1.5, "High volume on up move"))
        confidence_score += 1.5
    elif last["Volume_Ratio"] > 1.5 and last["Close"] < prev["Close"]:
        signals.append(("Bearish", -1.5, "High volume on down move"))
        confidence_score -= 1.5
    
    # Stochastic
    if last["Stoch_K"] < 20 and last["Stoch_D"] < 20:
        signals.append(("Bullish", 1, "Stochastic oversold"))
        confidence_score += 1
    elif last["Stoch_K"] > 80 and last["Stoch_D"] > 80:
        signals.append(("Bearish", -1, "Stochastic overbought"))
        confidence_score -= 1
    
    # Determine final signal
    if confidence_score >= 4:
        final_signal = "STRONG BUY"
        confidence = min(95, 60 + confidence_score * 5)
    elif confidence_score >= 2:
        final_signal = "BUY"
        confidence = min(85, 50 + confidence_score * 5)
    elif confidence_score <= -4:
        final_signal = "STRONG SELL"
        confidence = min(95, 60 + abs(confidence_score) * 5)
    elif confidence_score <= -2:
        final_signal = "SELL"
        confidence = min(85, 50 + abs(confidence_score) * 5)
    else:
        final_signal = "HOLD"
        confidence = 50 + abs(confidence_score) * 3
    
    reasons = [r[2] for r in signals]
    
    return {
        "signal": final_signal,
        "confidence": confidence,
        "score": confidence_score,
        "reasons": reasons,
        "components": signals
    }

# -----------------------------
# Enhanced Forecasting
# -----------------------------
def create_professional_forecast(data: pd.DataFrame, days: int) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Create professional forecast with uncertainty bounds."""
    df = data[["Close"]].reset_index()
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.dropna()
    
    if len(df) < 30:
        return pd.DataFrame(), pd.DataFrame(), {"error": "Insufficient data for forecasting"}
    
    # Configure Prophet with optimized parameters
    model = Prophet(
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        interval_width=0.95
    )
    
    # Add custom seasonality
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    
    model.fit(df)
    
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    
    # Calculate forecast metrics
    historical_fit = forecast[forecast["ds"] <= df["ds"].max()]
    mape = np.mean(np.abs((df["y"].values - historical_fit["yhat"].values) / df["y"].values)) * 100
    
    forecast_metrics = {
        "mape": mape,
        "trend": "Upward" if forecast["trend"].iloc[-1] > forecast["trend"].iloc[0] else "Downward",
        "uncertainty": float(forecast["yhat_upper"].iloc[-1] - forecast["yhat_lower"].iloc[-1]),
    }
    
    return df, forecast, forecast_metrics

# -----------------------------
# Professional Chart Components
# -----------------------------
def create_professional_candlestick_chart(data: pd.DataFrame, title: str = "Price Chart") -> go.Figure:
    """Create professional candlestick chart with indicators."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(title, "Volume", "RSI")
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="Price",
            increasing_line_color="#10b981",
            decreasing_line_color="#ef4444"
        ),
        row=1, col=1
    )
    
    # Moving Averages
    if "SMA_20" in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data["SMA_20"], name="SMA 20",
                      line=dict(color="#3b82f6", width=1.5)),
            row=1, col=1
        )
    if "SMA_50" in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data["SMA_50"], name="SMA 50",
                      line=dict(color="#8b5cf6", width=1.5)),
            row=1, col=1
        )
    
    # Bollinger Bands
    if "BB_Upper" in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data["BB_Upper"], name="BB Upper",
                      line=dict(color="rgba(245, 158, 11, 0.5)", width=1, dash="dash")),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data["BB_Lower"], name="BB Lower",
                      line=dict(color="rgba(245, 158, 11, 0.5)", width=1, dash="dash"),
                      fill="tonexty", fillcolor="rgba(245, 158, 11, 0.1)"),
            row=1, col=1
        )
    
    # Volume
    colors = ["#10b981" if data["Close"].iloc[i] >= data["Open"].iloc[i] else "#ef4444" 
              for i in range(len(data))]
    fig.add_trace(
        go.Bar(x=data.index, y=data["Volume"], name="Volume",
               marker_color=colors, opacity=0.7),
        row=2, col=1
    )
    
    # RSI
    if "RSI" in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data["RSI"], name="RSI",
                      line=dict(color="#8b5cf6", width=2)),
            row=3, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="#ef4444", 
                      opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#10b981", 
                      opacity=0.5, row=3, col=1)
    
    # Update layout
    fig.update_layout(
        template="plotly_dark",
        height=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor="rgba(15, 23, 42, 0.5)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        xaxis_rangeslider_visible=False
    )
    
    fig.update_xaxes(gridcolor="rgba(51, 65, 85, 0.3)")
    fig.update_yaxes(gridcolor="rgba(51, 65, 85, 0.3)")
    
    return fig

def create_forecast_chart(historical: pd.DataFrame, forecast: pd.DataFrame, 
                          title: str = "Price Forecast") -> go.Figure:
    """Create professional forecast visualization."""
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(
        go.Scatter(
            x=historical["ds"], y=historical["y"],
            mode="lines", name="Historical",
            line=dict(color="#3b82f6", width=2)
        )
    )
    
    # Forecast
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"], y=forecast["yhat"],
            mode="lines", name="Forecast",
            line=dict(color="#10b981", width=2)
        )
    )
    
    # Confidence interval
    fig.add_trace(
        go.Scatter(
            x=forecast["ds"].tolist() + forecast["ds"].tolist()[::-1],
            y=forecast["yhat_upper"].tolist() + forecast["yhat_lower"].tolist()[::-1],
            fill="toself", fillcolor="rgba(16, 185, 129, 0.2)",
            line=dict(color="rgba(255, 255, 255, 0)"),
            name="95% Confidence Interval"
        )
    )
    
    fig.update_layout(
        template="plotly_dark",
        title=title,
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="rgba(15, 23, 42, 0.5)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

# -----------------------------
# Session State Management
# -----------------------------
def initialize_session_state():
    """Initialize session state variables."""
    defaults = {
        "watchlist": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "SBIN.NS"],
        "portfolio": [],
        "auto_refresh": True,
        "refresh_interval": 60,
        "selected_symbol": "RELIANCE.NS",
        "theme": "dark",
        "last_update": datetime.now(),
        "data_cache": {},
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# -----------------------------
# Main Application
# -----------------------------
def main():
    # Initialize
    inject_professional_css()
    initialize_session_state()
    
    # Auto-refresh
    if st.session_state.auto_refresh and AUTO_REFRESH_OK:
        st_autorefresh(interval=st.session_state.refresh_interval * 1000, key="auto_refresh")
    
    # Header
    professional_header()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Market Intelligence")
        
        # Symbol Selection
        symbol_input = st.text_input("Symbol", value=st.session_state.selected_symbol, 
                                     placeholder="e.g., RELIANCE.NS")
        symbol = symbol_input.upper().strip()
        if not symbol.endswith((".NS", ".BO")) and not symbol.startswith("^"):
            symbol += ".NS"
        
        st.session_state.selected_symbol = symbol
        
        # Quick Select
        quick_symbols = {
            "NIFTY 50": "^NSEI",
            "BANK NIFTY": "^NSEBANK",
            "RELIANCE": "RELIANCE.NS",
            "TCS": "TCS.NS",
            "HDFC BANK": "HDFCBANK.NS",
            "INFOSYS": "INFY.NS",
            "SBI": "SBIN.NS",
        }
        
        selected_quick = st.selectbox("Quick Select", ["Select..."] + list(quick_symbols.keys()))
        if selected_quick != "Select...":
            symbol = quick_symbols[selected_quick]
            st.session_state.selected_symbol = symbol
        
        st.divider()
        
        # Analysis Parameters
        st.markdown("### ⚙️ Analysis Settings")
        forecast_days = st.slider("Forecast Horizon (Days)", 7, 60, 30)
        show_indicators = st.checkbox("Show Technical Indicators", True)
        show_volume = st.checkbox("Show Volume Analysis", True)
        
        st.divider()
        
        # Live Settings
        st.markdown("### 🔄 Live Settings")
        st.session_state.auto_refresh = st.toggle("Auto Refresh", st.session_state.auto_refresh)
        if st.session_state.auto_refresh:
            st.session_state.refresh_interval = st.select_slider(
                "Refresh Interval (sec)",
                options=[15, 30, 60, 120, 300],
                value=st.session_state.refresh_interval
            )
        
        st.divider()
        
        # Watchlist
        st.markdown("### ⭐ Watchlist")
        for wl_symbol in st.session_state.watchlist:
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(wl_symbol, key=f"wl_{wl_symbol}", use_container_width=True):
                    st.session_state.selected_symbol = wl_symbol
                    st.rerun()
            with col2:
                if st.button("❌", key=f"remove_{wl_symbol}"):
                    st.session_state.watchlist.remove(wl_symbol)
                    st.rerun()
        
        new_symbol = st.text_input("Add to Watchlist", placeholder="Symbol")
        if st.button("Add", use_container_width=True) and new_symbol:
            clean_symbol = new_symbol.upper().strip()
            if not clean_symbol.endswith((".NS", ".BO")):
                clean_symbol += ".NS"
            if clean_symbol not in st.session_state.watchlist:
                st.session_state.watchlist.append(clean_symbol)
                st.rerun()
    
    # Main Content
    if st.button("🚀 Analyze", type="primary", use_container_width=True) or True:
        with st.spinner(f"Analyzing {symbol}..."):
            try:
                # Fetch Data
                data, source, metadata = fetch_enhanced_stock_data(symbol)
                
                if data.empty:
                    st.error(f"No data available for {symbol}")
                    st.stop()
                
                # Calculate Indicators
                data = calculate_advanced_indicators(data)
                
                # Generate Signal
                signal_data = generate_advanced_signal(data)
                
                # Live Quote
                live_quote = fetch_live_quote(symbol)
                
                # Current Price
                current_price = float(data["Close"].iloc[-1])
                prev_price = float(data["Close"].iloc[-2]) if len(data) > 1 else current_price
                day_change = current_price - prev_price
                day_change_pct = (day_change / prev_price * 100) if prev_price else 0
                
                # Key Metrics Row
                st.markdown("### 📈 Market Overview")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    render_metric_card(
                        "Current Price",
                        f"₹{current_price:,.2f}",
                        f"{day_change:+.2f} ({day_change_pct:+.2f}%)",
                        "normal" if day_change >= 0 else "inverse",
                        "Last traded price"
                    )
                
                with col2:
                    day_range = f"₹{data['Low'].iloc[-1]:,.2f} - ₹{data['High'].iloc[-1]:,.2f}"
                    render_metric_card(
                        "Day Range",
                        day_range,
                        None,
                        "normal",
                        "Today's trading range"
                    )
                
                with col3:
                    week_52_high = float(data["High"].tail(252).max())
                    week_52_low = float(data["Low"].tail(252).min())
                    render_metric_card(
                        "52 Week Range",
                        f"₹{week_52_low:,.0f} - ₹{week_52_high:,.0f}",
                        None,
                        "normal",
                        "52 week high/low range"
                    )
                
                with col4:
                    volume = data["Volume"].iloc[-1]
                    avg_volume = data["Volume"].tail(20).mean()
                    vol_ratio = volume / avg_volume if avg_volume else 0
                    render_metric_card(
                        "Volume",
                        f"{volume:,.0f}",
                        f"{vol_ratio:.1f}x avg",
                        "normal",
                        "Today's volume vs 20-day average"
                    )
                
                with col5:
                    volatility = float(data["Volatility"].iloc[-1] * 100) if "Volatility" in data.columns else 0
                    render_metric_card(
                        "Volatility",
                        f"{volatility:.2f}%",
                        None,
                        "normal",
                        "Annualized volatility"
                    )
                
                # Signal Display
                st.markdown("### 🎯 Trading Signal")
                
                sig_col1, sig_col2, sig_col3 = st.columns([2, 1, 1])
                
                with sig_col1:
                    render_signal_badge(signal_data["signal"], signal_data["confidence"])
                
                with sig_col2:
                    st.metric("Signal Score", f"{signal_data['score']:.1f}")
                
                with sig_col3:
                    st.metric("Data Source", source)
                
                # AI Insight
                if signal_data["reasons"]:
                    st.markdown(f"""
                        <div class="ai-insight">
                            <div class="ai-insight-title">
                                <span>🧠 AI Market Intelligence</span>
                            </div>
                            <div style="color: var(--text-primary); line-height: 1.6;">
                                <strong>{symbol}</strong> shows a <strong>{signal_data['signal'].lower()}</strong> signal 
                                with {signal_data['confidence']:.0f}% confidence based on:
                                <ul style="margin-top: 8px;">
                                    {''.join([f'<li>{reason}</li>' for reason in signal_data['reasons'][:5]])}
                                </ul>
                                Current technical structure suggests {'bullish momentum' if 'BUY' in signal_data['signal'] else 'bearish pressure' if 'SELL' in signal_data['signal'] else 'consolidation'} 
                                with {'above-average' if vol_ratio > 1.2 else 'normal'} volume participation.
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Tabs for Detailed Analysis
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📊 Technical Chart", "🔮 Forecast", "📈 Indicators", 
                    "📰 News & Sentiment", "💼 Portfolio"
                ])
                
                with tab1:
                    st.markdown("### Price Action & Technical Analysis")
                    
                    # Main Chart
                    fig = create_professional_candlestick_chart(
                        data.tail(100) if len(data) > 100 else data,
                        f"{symbol} - Technical Chart"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Support/Resistance
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### Support Levels")
                        recent_lows = data["Low"].tail(50).nsmallest(3)
                        for i, (idx, val) in enumerate(recent_lows.items(), 1):
                            st.metric(f"S{i}", f"₹{val:,.2f}")
                    
                    with col2:
                        st.markdown("#### Resistance Levels")
                        recent_highs = data["High"].tail(50).nlargest(3)
                        for i, (idx, val) in enumerate(recent_highs.items(), 1):
                            st.metric(f"R{i}", f"₹{val:,.2f}")
                
                with tab2:
                    st.markdown(f"### {forecast_days}-Day Price Forecast")
                    
                    # Create forecast
                    hist_df, forecast_df, forecast_metrics = create_professional_forecast(data, forecast_days)
                    
                    if not forecast_df.empty:
                        # Forecast Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        future_forecast = forecast_df.tail(forecast_days)
                        final_pred = float(future_forecast["yhat"].iloc[-1])
                        pred_change = ((final_pred - current_price) / current_price * 100)
                        
                        with col1:
                            st.metric(
                                "Predicted Price",
                                f"₹{final_pred:,.2f}",
                                f"{pred_change:+.2f}%"
                            )
                        
                        with col2:
                            st.metric(
                                "Forecast Range",
                                f"₹{future_forecast['yhat_lower'].iloc[-1]:,.0f} - ₹{future_forecast['yhat_upper'].iloc[-1]:,.0f}"
                            )
                        
                        with col3:
                            st.metric(
                                "Model Accuracy (MAPE)",
                                f"{forecast_metrics['mape']:.2f}%"
                            )
                        
                        with col4:
                            st.metric(
                                "Trend Direction",
                                forecast_metrics['trend']
                            )
                        
                        # Forecast Chart
                        fig = create_forecast_chart(hist_df, forecast_df, f"{symbol} - {forecast_days} Day Forecast")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Forecast Table
                        st.markdown("#### Daily Forecast Values")
                        forecast_table = future_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
                        forecast_table.columns = ["Date", "Predicted", "Lower Bound", "Upper Bound"]
                        forecast_table["Date"] = forecast_table["Date"].dt.strftime("%d-%b-%Y")
                        forecast_table = forecast_table.round(2)
                        st.dataframe(forecast_table, use_container_width=True, hide_index=True)
                    else:
                        st.warning("Insufficient data for forecasting. Need at least 30 days of historical data.")
                
                with tab3:
                    st.markdown("### Technical Indicators Dashboard")
                    
                    # Indicator selection
                    indicators = st.multiselect(
                        "Select Indicators",
                        ["RSI", "MACD", "Bollinger Bands", "Stochastic", "ATR", "OBV"],
                        default=["RSI", "MACD"]
                    )
                    
                    if indicators:
                        cols = st.columns(min(len(indicators), 2))
                        
                        for i, indicator in enumerate(indicators):
                            with cols[i % 2]:
                                if indicator == "RSI" and "RSI" in data.columns:
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(x=data.index, y=data["RSI"], 
                                                           name="RSI", line=dict(color="#8b5cf6")))
                                    fig.add_hline(y=70, line_dash="dash", line_color="#ef4444")
                                    fig.add_hline(y=30, line_dash="dash", line_color="#10b981")
                                    fig.update_layout(
                                        title="RSI (14)",
                                        height=300,
                                        template="plotly_dark"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                elif indicator == "MACD" and "MACD" in data.columns:
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(x=data.index, y=data["MACD"], 
                                                           name="MACD", line=dict(color="#3b82f6")))
                                    fig.add_trace(go.Scatter(x=data.index, y=data["MACD_Signal"], 
                                                           name="Signal", line=dict(color="#f59e0b")))
                                    fig.add_trace(go.Bar(x=data.index, y=data["MACD_Histogram"], 
                                                       name="Histogram", marker_color="rgba(139, 92, 246, 0.5)"))
                                    fig.update_layout(
                                        title="MACD",
                                        height=300,
                                        template="plotly_dark"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                elif indicator == "Bollinger Bands" and "BB_Upper" in data.columns:
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(x=data.index, y=data["Close"], 
                                                           name="Close", line=dict(color="white")))
                                    fig.add_trace(go.Scatter(x=data.index, y=data["BB_Upper"], 
                                                           name="Upper", line=dict(color="#f59e0b", dash="dash")))
                                    fig.add_trace(go.Scatter(x=data.index, y=data["BB_Middle"], 
                                                           name="Middle", line=dict(color="#6b7280")))
                                    fig.add_trace(go.Scatter(x=data.index, y=data["BB_Lower"], 
                                                           name="Lower", line=dict(color="#f59e0b", dash="dash")))
                                    fig.update_layout(
                                        title="Bollinger Bands",
                                        height=300,
                                        template="plotly_dark"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                elif indicator == "Stochastic" and "Stoch_K" in data.columns:
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(x=data.index, y=data["Stoch_K"], 
                                                           name="%K", line=dict(color="#3b82f6")))
                                    fig.add_trace(go.Scatter(x=data.index, y=data["Stoch_D"], 
                                                           name="%D", line=dict(color="#f59e0b")))
                                    fig.add_hline(y=80, line_dash="dash", line_color="#ef4444")
                                    fig.add_hline(y=20, line_dash="dash", line_color="#10b981")
                                    fig.update_layout(
                                        title="Stochastic Oscillator",
                                        height=300,
                                        template="plotly_dark"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                
                with tab4:
                    st.markdown("### 📰 News & Market Sentiment")
                    
                    # Fetch news
                    try:
                        ticker = yf.Ticker(symbol)
                        news = ticker.news[:10] if hasattr(ticker, "news") else []
                        
                        if news:
                            for item in news:
                                with st.container():
                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        st.markdown(f"**{item.get('title', 'No title')}**")
                                        st.caption(f"{item.get('publisher', 'Unknown')} • {datetime.fromtimestamp(item.get('providerPublishTime', 0)).strftime('%d-%b-%Y %H:%M')}")
                                    with col2:
                                        if item.get('link'):
                                            st.markdown(f"[Read More]({item['link']})")
                                    st.divider()
                        else:
                            st.info("No recent news available for this symbol.")
                    except Exception as e:
                        st.warning(f"Unable to fetch news: {str(e)}")
                
                with tab5:
                    st.markdown("### 💼 Portfolio Tracker")
                    
                    # Portfolio Summary
                    if st.session_state.portfolio:
                        portfolio_data = []
                        total_value = 0
                        total_invested = 0
                        
                        for pos in st.session_state.portfolio:
                            pos_symbol = pos.get("symbol", "")
                            quantity = pos.get("quantity", 0)
                            buy_price = pos.get("buy_price", 0)
                            
                            # Get current price
                            pos_data, _, _ = fetch_enhanced_stock_data(pos_symbol)
                            if not pos_data.empty:
                                current_price = float(pos_data["Close"].iloc[-1])
                                current_value = quantity * current_price
                                invested = quantity * buy_price
                                pnl = current_value - invested
                                pnl_pct = (pnl / invested * 100) if invested else 0
                                
                                portfolio_data.append({
                                    "Symbol": pos_symbol,
                                    "Quantity": quantity,
                                    "Buy Price": f"₹{buy_price:,.2f}",
                                    "Current Price": f"₹{current_price:,.2f}",
                                    "Invested": f"₹{invested:,.2f}",
                                    "Current Value": f"₹{current_value:,.2f}",
                                    "P&L": f"₹{pnl:,.2f}",
                                    "P&L %": f"{pnl_pct:+.2f}%"
                                })
                                
                                total_value += current_value
                                total_invested += invested
                        
                        if portfolio_data:
                            # Portfolio Metrics
                            total_pnl = total_value - total_invested
                            total_pnl_pct = (total_pnl / total_invested * 100) if total_invested else 0
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Invested", f"₹{total_invested:,.2f}")
                            with col2:
                                st.metric("Current Value", f"₹{total_value:,.2f}")
                            with col3:
                                st.metric("Total P&L", f"₹{total_pnl:,.2f}", 
                                         f"{total_pnl_pct:+.2f}%")
                            
                            st.dataframe(pd.DataFrame(portfolio_data), use_container_width=True, hide_index=True)
                    
                    # Add Position
                    st.markdown("#### Add Position")
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                    
                    with col1:
                        new_pos_symbol = st.text_input("Symbol", key="new_pos_symbol")
                    with col2:
                        new_pos_qty = st.number_input("Quantity", min_value=0.0, step=1.0, key="new_pos_qty")
                    with col3:
                        new_pos_price = st.number_input("Buy Price", min_value=0.0, step=0.01, key="new_pos_price")
                    with col4:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("Add", use_container_width=True):
                            if new_pos_symbol and new_pos_qty > 0 and new_pos_price > 0:
                                clean_symbol = new_pos_symbol.upper().strip()
                                if not clean_symbol.endswith((".NS", ".BO")):
                                    clean_symbol += ".NS"
                                st.session_state.portfolio.append({
                                    "symbol": clean_symbol,
                                    "quantity": new_pos_qty,
                                    "buy_price": new_pos_price
                                })
                                st.success(f"Added {clean_symbol} to portfolio")
                                st.rerun()
                
                # Footer
                st.divider()
                st.caption(f"""
                    <div style="text-align: center; color: var(--text-muted);">
                        <p>⚠️ Disclaimer: This application is for educational purposes only. 
                        All predictions and signals are based on historical data and technical analysis. 
                        Past performance does not guarantee future results. Always conduct your own research 
                        before making investment decisions.</p>
                        <p>Data provided by {source} • Last Updated: {datetime.now().strftime('%d-%b-%Y %H:%M:%S')}</p>
                    </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error analyzing {symbol}: {str(e)}")
                st.info("Please try another symbol or check your internet connection.")

if __name__ == "__main__":
    main()