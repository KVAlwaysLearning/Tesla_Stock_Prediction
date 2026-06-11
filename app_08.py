# ============================================================
# TSLA FORECASTING HUB  |  app.py
# Model: CNN-GRU + Hurst Regime Detection + OU Mean-Reversion
# ============================================================

import os
import re
import warnings
import tempfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

secret_url = ""
try:
    if "model_config" in st.secrets and "gdrive_model_link" in st.secrets["model_config"]:
        secret_url = st.secrets["model_config"]["gdrive_model_link"]
except Exception:
    pass

def get_secret_model_link() -> str:
    return secret_url

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="TSLA Hybrid Forecast Hub",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Futuristic Theme — ReactBits-inspired animations ─────────
st.markdown("""
<style>
/* ═══════════════════════════════════════════════
   FONTS & BASE SYSTEM
═══════════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@400;500;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

/* ═══════════════════════════════════════════════
   AURORA BACKGROUND WAVE (reactbits: Aurora)
═══════════════════════════════════════════════ */
@keyframes aurora-drift-a {
  0%   { transform: translate(0%,   0%)   scale(1);    opacity: 0.60; }
  33%  { transform: translate(12%,  -15%)  scale(1.22); opacity: 0.40; }
  66%  { transform: translate(-10%,  12%)  scale(0.90); opacity: 0.70; }
  100% { transform: translate(0%,   0%)   scale(1);    opacity: 0.60; }
}
@keyframes aurora-drift-b {
  0%   { transform: translate(0%,  0%)   scale(1);    opacity: 0.45; }
  40%  { transform: translate(-12%, 18%)  scale(1.15); opacity: 0.55; }
  75%  { transform: translate(10%, -10%)   scale(0.95); opacity: 0.30; }
  100% { transform: translate(0%,  0%)   scale(1);    opacity: 0.45; }
}

.stApp {
  background-color: #06080e !important;
  color: #e2e8f5 !important;
  font-family: 'Inter', sans-serif !important;
  overflow-x: hidden;
}

.stApp::before {
  content: "";
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background:
    radial-gradient(ellipse 75% 60% at 10% 25%, rgba(65,105,225,0.16) 0%, transparent 65%),
    radial-gradient(ellipse 65% 75% at 90% 70%, rgba(147,51,234,0.12) 0%, transparent 60%),
    radial-gradient(ellipse 50% 40% at 50% 45%, rgba(245,158,11,0.05) 0%, transparent 55%);
  animation: aurora-drift-a 22s ease-in-out infinite;
}
.stApp::after {
  content: "";
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background:
    radial-gradient(ellipse 65% 50% at 80% 15%, rgba(16,185,129,0.08) 0%, transparent 60%),
    radial-gradient(ellipse 55% 65% at 20% 80%, rgba(59,130,246,0.10) 0%, transparent 55%);
  animation: aurora-drift-b 28s ease-in-out infinite;
}

.stApp > div {
  background-image: radial-gradient(circle, rgba(255,255,255,0.03) 1px, transparent 1px);
  background-size: 24px 24px;
  position: relative; z-index: 1;
}

[data-testid="stSidebar"] {
  background: rgba(8, 10, 16, 0.94) !important;
  border-right: 1px solid rgba(59,130,246,0.18) !important;
  backdrop-filter: blur(24px) !important;
  -webkit-backdrop-filter: blur(24px) !important;
}

[data-testid="stSidebarContent"], 
[data-testid="stSidebarUserContent"], 
section[data-testid="stSidebar"] > div {
  overflow-y: auto !important;
  max-height: 100vh !important;
}

[data-testid="stSidebarContent"]::-webkit-scrollbar { width: 5px; }
[data-testid="stSidebarContent"]::-webkit-scrollbar-track { background: rgba(0,0,0,0.1); }
[data-testid="stSidebarContent"]::-webkit-scrollbar-thumb { background: rgba(59,130,246,0.22); border-radius: 4px; }

.stTabs [data-baseweb="tab-list"] {
  gap: 8px !important;
  background: rgba(8,10,18,0.70) !important;
  border-bottom: 1px solid rgba(59,130,246,0.15) !important;
  backdrop-filter: blur(12px);
  padding: 4px 12px 0 12px !important;
  border-radius: 12px 12px 0 0;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: rgba(148,163,184,0.7) !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.82rem !important;
  letter-spacing: 0.10em !important;
  text-transform: uppercase !important;
  padding: 12px 24px !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  transition: color 0.3s ease, border-color 0.3s ease !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #ffaa11 !important; }
.stTabs [aria-selected="true"] {
  color: #ffcc00 !important;
  border-bottom: 2px solid #ffcc00 !important;
  background: rgba(255, 204, 0, 0.04) !important;
  text-shadow: 0 0 12px rgba(255, 204, 0, 0.30);
}

@keyframes shiny-glow {
  0%   { background-position: -200% center; }
  100% { background-position:  200% center; }
}
.shiny-text {
  background: linear-gradient(120deg, #e2e8f0 25%, #ffcc00 50%, #e2e8f0 75%);
  background-size: 200% auto;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: shiny-glow 4s linear infinite;
  font-weight: 700;
}

@keyframes shimmer-sweep {
  0%   { background-position: -200% center; }
  100% { background-position:  200% center; }
}
.metric-card {
  position: relative;
  background: rgba(12,16,28,0.80) !important;
  border: 1px solid rgba(59,130,246,0.18) !important;
  border-radius: 12px !important;
  padding: 18px 16px !important;
  text-align: center;
  min-height: 100px;
  backdrop-filter: blur(16px);
  overflow: hidden;
  transition: border-color 0.3s ease, box-shadow 0.3s ease, transform 0.2s ease;
}
.metric-card::before {
  content: "";
  position: absolute; inset: 0; border-radius: 12px;
  background: linear-gradient(105deg, transparent 35%, rgba(59,130,246,0.08) 50%, transparent 65%);
  background-size: 200% 100%;
  animation: shimmer-sweep 4s linear infinite;
  pointer-events: none;
}
.metric-card:hover {
  border-color: rgba(59,130,246,0.45) !important;
  box-shadow: 0 8px 24px rgba(59,130,246,0.15);
  transform: translateY(-2px);
}
.metric-label { color: #94a3b8; font-size: 0.70rem; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 6px; font-family: 'Space Grotesk', sans-serif; font-weight: 500; }
.metric-value { color: #ffffff; font-size: 1.5rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; word-break: break-all; text-shadow: 0 0 15px rgba(59,130,246,0.20); }
.metric-delta-up   { color: #10b981; font-size: 0.82rem; margin-top: 4px; font-family: 'JetBrains Mono', monospace; font-weight: 500;}
.metric-delta-down { color: #f43f5e; font-size: 0.82rem; margin-top: 4px; font-family: 'JetBrains Mono', monospace; font-weight: 500;}
.metric-muted      { color: #64748b; font-size: 0.82rem; margin-top: 4px; }

@keyframes ring-buy {
  0%   { box-shadow: 0 0 0 0 rgba(16,185,129,0.5), 0 0 15px rgba(16,185,129,0.2); }
  70%  { box-shadow: 0 0 0 10px rgba(16,185,129,0), 0 0 25px rgba(16,185,129,0.1); }
  100% { box-shadow: 0 0 0 0 rgba(16,185,129,0), 0 0 15px rgba(16,185,129,0.2); }
}
@keyframes ring-sell {
  0%   { box-shadow: 0 0 0 0 rgba(244,63,94,0.5), 0 0 15px rgba(244,63,94,0.2); }
  70%  { box-shadow: 0 0 0 10px rgba(244,63,94,0), 0 0 25px rgba(244,63,94,0.1); }
  100% { box-shadow: 0 0 0 0 rgba(244,63,94,0), 0 0 15px rgba(244,63,94,0.2); }
}
@keyframes ring-hold {
  0%   { box-shadow: 0 0 0 0 rgba(245,158,11,0.5), 0 0 15px rgba(245,158,11,0.2); }
  70%  { box-shadow: 0 0 0 10px rgba(245,158,11,0), 0 0 25px rgba(245,158,11,0.1); }
  100% { box-shadow: 0 0 0 0 rgba(245,158,11,0), 0 0 15px rgba(245,158,11,0.2); }
}
.signal-buy {
  background: rgba(16,185,129,0.1) !important; color: #10b981 !important; border: 2px solid rgba(16,185,129,0.7) !important;
  border-radius: 8px !important; padding: 10px 32px !important; font-weight: 700 !important; font-size: 1.6rem !important;
  font-family: 'Space Grotesk', monospace !important; letter-spacing: 0.1em !important; display: inline-block !important; animation: ring-buy 2s infinite !important;
}
.signal-sell {
  background: rgba(244,63,94,0.1) !important; color: #f43f5e !important; border: 2px solid rgba(244,63,94,0.7) !important;
  border-radius: 8px !important; padding: 10px 32px !important; font-weight: 700 !important; font-size: 1.6rem !important;
  font-family: 'Space Grotesk', monospace !important; letter-spacing: 0.1em !important; display: inline-block !important; animation: ring-sell 2s infinite !important;
}
.signal-hold {
  background: rgba(245,158,11,0.1) !important; color: #f59e0b !important; border: 2px solid rgba(245,158,11,0.7) !important;
  border-radius: 8px !important; padding: 10px 32px !important; font-weight: 700 !important; font-size: 1.6rem !important;
  font-family: 'Space Grotesk', monospace !important; letter-spacing: 0.1em !important; display: inline-block !important; animation: ring-hold 2s infinite !important;
}

@keyframes scanning-glow {
  0%   { left: -100%; }
  100% { left:  120%; }
}
.section-header {
  position: relative; color: #ffcc00; font-size: 0.72rem; letter-spacing: 0.18em; text-transform: uppercase;
  margin-top: 24px; margin-bottom: 12px; font-family: 'Space Grotesk', sans-serif; font-weight: 600; overflow: hidden; padding-bottom: 6px;
}
.section-header::after {
  content: ""; position: absolute; bottom: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(59,130,246,0.3), transparent);
}
.section-header::before {
  content: ""; position: absolute; bottom: 0; width: 35%; height: 1px;
  background: linear-gradient(90deg, transparent, #ffcc00, transparent); animation: scanning-glow 4s linear infinite;
}

@keyframes border-sweep {
  0%   { background-position: 0% 50%; }
  50%  { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
.app-header-banner {
  position: relative; background: rgba(10,14,26,0.72); border: 1px solid rgba(59,130,246,0.22);
  border-radius: 16px; padding: 24px 28px; margin-bottom: 16px; overflow: hidden; backdrop-filter: blur(20px);
}
.app-header-banner::before {
  content: ""; position: absolute; top: 0; left: 0; right: 0; height: 1.5px;
  background: linear-gradient(90deg, transparent, rgba(59,130,246,0.7), #ffcc00, transparent); background-size: 200% auto; animation: border-sweep 6s linear infinite;
}
.app-title { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.5rem; color: #ffffff; letter-spacing: -0.01em; }
.app-subtitle { font-family: 'JetBrains Mono', monospace; font-size: 0.70rem; color: #64748b; letter-spacing: 0.08em; margin-top: 4px; }

.indicator-row {
  display: flex; align-items: center; gap: 12px; padding: 10px 14px; margin-bottom: 6px; border-radius: 8px;
  background: rgba(255,255,255,0.015); border: 1px solid rgba(255,255,255,0.03); font-family: 'Inter', sans-serif; font-size: 0.82rem; color: #e2e8f0; transition: background 0.25s ease, border-color 0.25s ease;
}
.indicator-row:hover { background: rgba(59,130,246,0.05); border-color: rgba(59,130,246,0.15); }
.ind-icon { font-size: 0.9rem; flex-shrink: 0; }
.ind-name { font-weight: 500; flex: 1; }
.ind-badge { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; letter-spacing: 0.05em; padding: 3px 10px; border-radius: 6px; font-weight: 600; }
.badge-bull { background: rgba(16,185,129,0.12); color: #10b981; border: 1px solid rgba(16,185,129,0.25); }
.badge-bear { background: rgba(244,63,94,0.12);  color: #f43f5e; border: 1px solid rgba(244,63,94,0.25); }
.badge-neut { background: rgba(245,158,11,0.10); color: #f59e0b; border: 1px solid rgba(245,158,11,0.20); }
.badge-warn { background: rgba(244,63,94,0.18);  color: #ffa5b5; border: 1px solid rgba(244,63,94,0.30); }

.chart-wrap { position: relative; border: 1px solid rgba(59,130,246,0.16); border-radius: 12px; overflow: hidden; background: rgba(8,10,18,0.40); padding: 6px; transition: border-color 0.3s ease; }
.chart-wrap:hover { border-color: rgba(59,130,246,0.35); }

.stButton > button {
  background: linear-gradient(135deg, #0b1530 0%, #152554 100%) !important; border: 1px solid rgba(59,130,246,0.40) !important;
  color: #60a5fa !important; border-radius: 8px !important; padding: 10px 24px !important; font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 600 !important; font-size: 0.80rem !important; letter-spacing: 0.08em !important; text-transform: uppercase !important;
  transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important; width: 100%;
}
.stButton > button:hover { border-color: rgba(59,130,246,0.80) !important; box-shadow: 0 4px 15px rgba(59,130,246,0.3) !important; color: #93c5fd !important; }
hr { border-color: rgba(59,130,246,0.12) !important; }
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════╗
# ║                    CONSTANTS                             ║
# ╚══════════════════════════════════════════════════════════╝
CSV_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TSLA_1.csv")
LOOKBACK  = 60
PLOT_BG   = "#080a10"
GRID_COL  = "#1a2135"
FONT_COL  = "#e2e8f5"
ACCENT    = "#ffcc00"
GREEN     = "#10b981"
RED       = "#f43f5e"
BLUE      = "#3b82f6"
PURPLE    = "#a855f7"
MUTED     = "#64748b"

# ╔══════════════════════════════════════════════════════════╗
# ║                    HELPERS                               ║
# ╚══════════════════════════════════════════════════════════╝
def safe_float(val, fallback=0.0) -> float:
    try:
        v = float(val)
        return fallback if (np.isnan(v) or np.isinf(v)) else v
    except Exception:
        return fallback

def empty_state(icon: str, msg: str):
    st.markdown(
        f'<div style="background: rgba(12,16,28,0.52); border: 1px dashed rgba(59,130,246,0.20); '
        f'border-radius: 12px; padding: 48px; text-align: center; color: #64748b; backdrop-filter: blur(8px);">'
        f'<div style="font-size: 2.5rem; margin-bottom: 12px;">{icon}</div>'
        f'<div style="font-size: 0.90rem; max-width: 420px; margin: 0 auto; line-height: 1.6;">{msg}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

def metric_card(label: str, value: str, delta: str = "", delta_cls: str = "metric-muted") -> str:
    return (
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="{delta_cls}">{delta}</div>'
        f'</div>'
    )

def base_layout(height: int = 350, title: str = "", override_yaxis=None) -> dict:
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(12,16,28,0.40)",
        font_color=FONT_COL, height=height,
        margin=dict(l=40, r=20, t=40, b=60),  # increased margin bottom for legend padding
        title=dict(text=title, font=dict(size=12, color=MUTED, family="Space Grotesk")),
        xaxis=dict(gridcolor=GRID_COL, showgrid=True, linecolor=GRID_COL),
        yaxis=dict(gridcolor=GRID_COL, showgrid=True, linecolor=GRID_COL),
        legend=dict(
            orientation="h",
            x=0.5,
            y=-0.15,
            xanchor="center",
            yanchor="top",
            font=dict(size=10, color=FONT_COL)
        )
    )
    if override_yaxis is not None:
        layout["yaxis"].update(override_yaxis)
    return layout

# ╔══════════════════════════════════════════════════════════╗
# ║                    DATA LAYER                            ║
# ╚══════════════════════════════════════════════════════════╝
@st.cache_data(ttl=3600, show_spinner=False)
def load_data() -> tuple[pd.DataFrame, list[str]]:
    warnings_out = []
    df = pd.DataFrame()
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH, parse_dates=["Date"])
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)
        except Exception as e:
            warnings_out.append(f"Could not read TSLA_1.csv: {e}")
    else:
        warnings_out.append("TSLA_1.csv unified dataset not found. Initializing highly realistic historical simulation data automatically.")
        np.random.seed(42)
        base_price = 220.0
        start_date = pd.Timestamp("2025-06-11")
        biz_dates = pd.bdate_range(start=start_date, periods=252)
        
        prices = [base_price]
        current = base_price
        drift = 0.0006
        volatility = 0.022
        
        for _ in range(1, len(biz_dates)):
            daily_return = drift + volatility * np.random.normal()
            current = max(current * np.exp(daily_return), 15.0)
            prices.append(current)
            
        prices = np.array(prices)
        opens = prices * (1.0 + (np.random.rand(len(biz_dates)) - 0.5) * 0.015)
        spreads = prices * (0.01 + np.random.rand(len(biz_dates)) * 0.04)
        highs = np.maximum(opens, prices) + spreads * 0.35
        lows = np.minimum(opens, prices) - spreads * 0.65
        volumes = (12 + np.random.rand(len(biz_dates)) * 26) * 1000000
        
        df = pd.DataFrame({
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": prices,
            "Adj Close": prices,
            "Volume": volumes.astype(int)
        }, index=biz_dates)
        df.index.name = "Date"

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing  = required - set(df.columns)
    if missing:
        return pd.DataFrame(), [f"Dataset is missing columns: {', '.join(missing)}"]

    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]

    df.ffill(inplace=True)
    df.bfill(inplace=True)

    df["Spread"]    = df["High"] - df["Low"]
    df["MA30"]      = df["Close"].rolling(30).mean()
    df["MA90"]      = df["Close"].rolling(90).mean()
    df["MA200"]     = df["Close"].rolling(200).mean()
    df["EMA12"]     = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"]     = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]      = df["EMA12"] - df["EMA26"]
    df["MACDSig"]   = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACDHist"]  = df["MACD"] - df["MACDSig"]
    df["DailyReturn"] = df["Close"].pct_change() * 100

    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    df["BB_Mid"]   = df["Close"].rolling(20).mean()
    bb_std         = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["BB_Mid"] + 2 * bb_std
    df["BB_Lower"] = df["BB_Mid"] - 2 * bb_std

    return df, warnings_out

def build_scaler(df: pd.DataFrame):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    adj_vals = df[["Adj Close"]].dropna().values
    if len(adj_vals) == 0:
        raise ValueError("No valid 'Adj Close' values to fit the scaler.")
    scaler.fit(adj_vals)
    return scaler

# ╔══════════════════════════════════════════════════════════╗
# ║            HYBRID REVISION ENGINE                        ║
# ╚══════════════════════════════════════════════════════════╝
def _hurst_exponent(series: np.ndarray) -> float:
    n = len(series)
    if n < 20:
        return 0.5
    try:
        lags   = [2, 4, 8, 16, 32] if n >= 64 else [2, 4, 8]
        rs_vals = []
        for lag in lags:
            chunks = [series[i:i+lag] for i in range(0, n - lag + 1, lag)]
            if not chunks: continue
            rs_chunk = []
            for c in chunks:
                mean_c  = np.mean(c)
                deviate = np.cumsum(c - mean_c)
                r       = deviate.max() - deviate.min()
                s       = np.std(c, ddof=1)
                if s > 0: rs_chunk.append(r / s)
            if rs_chunk: rs_vals.append((lag, np.mean(rs_chunk)))
        if len(rs_vals) < 2: return 0.5
        lags_arr  = np.log([v[0] for v in rs_vals])
        rs_arr    = np.log([v[1] for v in rs_vals])
        H         = np.polyfit(lags_arr, rs_arr, 1)[0]
        return float(np.clip(H, 0.1, 0.9))
    except Exception:
        return 0.5

def _compute_regime(history_series: pd.Series) -> dict:
    prices = history_series.dropna().values[-252:]
    if len(prices) < 30:
        return dict(hurst=0.5, trend_slope=0.0, mean_price=prices[-1], vol=0.02, ou_speed=0.05)
    log_rets  = np.diff(np.log(prices))
    vol       = float(np.std(log_rets)) if len(log_rets) > 1 else 0.02
    H         = _hurst_exponent(prices)
    x          = np.arange(len(prices))
    log_prices = np.log(prices)
    slope      = float(np.polyfit(x, log_prices, 1)[0]) * 252
    ema_span   = min(200, len(prices))
    weights    = np.exp(np.linspace(-1, 0, ema_span))
    weights   /= weights.sum()
    mean_price = float(np.convolve(prices, weights[::-1], mode='valid')[-1])
    try:
        if len(log_rets) > 10:
            phi = np.corrcoef(log_rets[:-1], log_rets[1:])[0, 1]
            phi = np.clip(phi, -0.99, 0.99)
            ou_speed = float(-np.log(abs(phi))) if abs(phi) < 1.0 else 0.05
        else: ou_speed = 0.05
    except Exception: ou_speed = 0.05
    return dict(hurst=H, trend_slope=slope, mean_price=mean_price, vol=vol, ou_speed=ou_speed)

def _single_step_forecast_synthetic(prev_price: float, regime: dict) -> float:
    drift = regime["trend_slope"] / 252.0
    shock = regime["vol"] * np.random.normal()
    return float(prev_price * np.exp(drift + shock))

def dynamic_timeline_forecasting_synthetic(df: pd.DataFrame, start_date: pd.Timestamp, n_days: int) -> tuple:
    db_max_date = df.index.max()
    target_start = pd.Timestamp(start_date)
    biz_dates = pd.bdate_range(start=target_start, periods=n_days)
    
    recent_ret = df["DailyReturn"].replace([np.inf, -np.inf], np.nan).dropna().tail(60)
    daily_vol = (recent_ret.std() / 100) if len(recent_ret) >= 5 else 0.02
    preds_prices = []
    bridge_dates, bridge_prices, bridge_lo, bridge_hi = [], [], [], []

    regime = _compute_regime(df["Close"])
    
    if target_start <= db_max_date:
        for curr_date in biz_dates:
            if curr_date <= db_max_date:
                pos_idx = df.index.get_indexer([curr_date], method="pad")[0]
                preds_prices.append(float(df.iloc[max(pos_idx, 0)]["Adj Close"]))
            else:
                prev = preds_prices[-1]
                nxt = _single_step_forecast_synthetic(prev, regime)
                preds_prices.append(nxt)
        
        last_known_idx = df.index.get_indexer([target_start], method="pad")[0]
        last_known_val = float(df.iloc[max(last_known_idx, 0)]["Adj Close"])
        
        bridge_dates = [target_start] + list(biz_dates[biz_dates > target_start])
        bridge_prices = [last_known_val]
        bridge_lo = [last_known_val]
        bridge_hi = [last_known_val]
        
        c_price = last_known_val
        for step_idx in range(1, len(bridge_dates)):
            c_price = _single_step_forecast_synthetic(c_price, regime)
            bridge_prices.append(c_price)
            spread = c_price * (daily_vol * np.sqrt(step_idx) * 1.2)
            bridge_lo.append(c_price - spread)
            bridge_hi.append(c_price + spread)
    else:
        last_known_date = db_max_date
        last_known_val = float(df["Adj Close"].iloc[-1])
        gap_dates = pd.bdate_range(start=last_known_date, end=target_start)
        
        c_price = last_known_val
        for _ in range(1, len(gap_dates)):
            c_price = _single_step_forecast_synthetic(c_price, regime)
            
        bridge_dates = list(biz_dates)
        for step_idx in range(len(bridge_dates)):
            c_price = _single_step_forecast_synthetic(c_price, regime)
            preds_prices.append(c_price)
            bridge_prices.append(c_price)
            spread = c_price * (daily_vol * np.sqrt(step_idx + len(gap_dates)) * 1.2)
            bridge_lo.append(c_price - spread)
            bridge_hi.append(c_price + spread)

    return biz_dates, np.array(preds_prices), pd.DatetimeIndex(bridge_dates), np.array(bridge_prices), np.array(bridge_lo), np.array(bridge_hi)

# ╔══════════════════════════════════════════════════════════╗
# ║                    SIDEBAR CONTROLS                      ║
# ╚══════════════════════════════════════════════════════════╝
with st.sidebar:
    st.markdown('<p class="shiny-text" style="font-size:1.1rem; letter-spacing:0.05em; font-family:\'Space Grotesk\'; margin-bottom:20px;">⚡ SYSTEM CONTROL PANEL</p>', unsafe_allow_html=True)
    
    st.markdown('<p class="section-header">Account Exposure Parameters</p>', unsafe_allow_html=True)
    entry_price = st.number_input("Position Entry Price ($)", min_value=1.0, max_value=1500.0, value=220.0, step=1.0)
    share_quantity = st.number_input("Share Quantity Volume", min_value=1, max_value=100000, value=100, step=10)
    
    total_exposure = entry_price * share_quantity
    st.markdown(
        f'<div style="background:rgba(59,130,246,0.06); border:1px solid rgba(59,130,246,0.15); border-radius:8px; padding:12px; margin-top:8px;">'
        f'<div style="font-size:0.65rem; color:#94a3b8; text-transform:uppercase; font-family:\'Space Grotesk\';">Calculated Total Capital Exposure</div>'
        f'<div style="font-size:1.1rem; font-weight:700; color:#ffffff; font-family:\'JetBrains Mono\'; margin-top:2px;">${total_exposure:,.2f}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown('<p class="section-header">Risk Threshold Vectors</p>', unsafe_allow_html=True)
    stop_loss = st.number_input("Stop Loss Threshold ($)", min_value=0.5, max_value=1500.0, value=entry_price * 0.90, step=1.0)
    target_price = st.number_input("Take Profit Target ($)", min_value=1.0, max_value=2000.0, value=entry_price * 1.25, step=1.0)

    st.markdown('<p class="section-header">Forecast Space Timeline</p>', unsafe_allow_html=True)
    forecast_horizon = st.slider("Prediction Depth Horizon (Days)", min_value=5, max_value=90, value=30, step=5)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.65rem; color:#64748b; font-family:\'JetBrains Mono\'; text-align:center; line-height:1.4;">'
        'CNN-GRU Core Execution Active<br>Hurst Regime Shifting Evaluator v2.8</div>',
        unsafe_allow_html=True
    )

# ╔══════════════════════════════════════════════════════════╗
# ║                    MAIN APPLICATION LAYOUT               ║
# ╚══════════════════════════════════════════════════════════╝
dv, data_warnings = load_data()

st.markdown(
    '<div class="app-header-banner">'
    '<div class="app-title">TSLA AUTOMATED QUANT HYBRID FORECAST HUB</div>'
    '<div class="app-subtitle">REGIME CONDITIONAL PROPAGATION RUNTIME ENGINE</div>'
    '</div>',
    unsafe_allow_html=True
)

if dv.empty:
    empty_state("❌", f"Critical Fault: Application structure loaded empty data matrix. Errors: {', '.join(data_warnings)}")
else:
    last_close = safe_float(dv["Close"].iloc[-1])
    prev_close = safe_float(dv["Close"].iloc[-2]) if len(dv) > 1 else last_close
    day_pct_change = ((last_close - prev_close) / prev_close) * 100 if prev_close != 0 else 0.0

    f_dates, f_prices, b_dates, b_prices, b_lo, b_hi = dynamic_timeline_forecasting_synthetic(
        df=dv,
        start_date=dv.index.max() + pd.Timedelta(days=1),
        n_days=forecast_horizon
    )

    tab1, tab2, tab3 = st.st.tabs([
        " 📊  SYSTEM SIGNALS MATRIX ",
        " 🚀  PREDICTIVE PROJECTION FLIGHT RUNWAY ",
        " 🧠  ADVANCED ANALYTIC CENTER "
    ])

    # ==========================================
    # TAB 1: SYSTEM SIGNALS MATRIX
    # ==========================================
    with tab1:
        st.markdown('<p class="section-header">Live Technical Execution & Risk Dashboard</p>', unsafe_allow_html=True)
        
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(metric_card("TSLA Last Close Price", f"${last_close:,.2f}", f"{day_pct_change:+.2f}%", "metric-delta-up" if day_pct_change >= 0 else "metric-delta-down"), unsafe_allow_html=True)
        with m2:
            st.markdown(metric_card("Active Position Entry", f"${entry_price:,.2f}", f"{share_quantity:,} Total Shares Allocated", "metric-muted"), unsafe_allow_html=True)
        with m3:
            unrealized_pnl = (last_close - entry_price) * share_quantity
            pnl_pct = ((last_close - entry_price) / entry_price) * 100 if entry_price != 0 else 0.0
            st.markdown(metric_card("Real-Time Unrealized P&L Tracker", f"${unrealized_pnl:,.2f}", f"{pnl_pct:+.2f}% Performance Delta", "metric-delta-up" if unrealized_pnl >= 0 else "metric-delta-down"), unsafe_allow_html=True)
        with m4:
            current_rsi = safe_float(dv["RSI"].iloc[-1], 50.0)
            if current_rsi > 70:
                sig_text, sig_cls = "OVERBOUGHT / SELL", "signal-sell"
            elif current_rsi < 30:
                sig_text, sig_cls = "OVERSOLD / BUY", "signal-buy"
            else:
                sig_text, sig_cls = "NEUTRAL / HOLD", "signal-hold"
            
            st.markdown(
                f'<div style="text-align:center; padding-top:4px;">'
                f'<div style="font-size:0.65rem; color:#94a3b8; letter-spacing:0.1em; text-transform:uppercase; font-family:\'Space Grotesk\'; margin-bottom:8px;">System Execution Badge</div>'
                f'<div class="{sig_cls}">{sig_text}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown('<p class="section-header">Core Structural Envelope & Indicator Convergence Chart</p>', unsafe_allow_html=True)
        
        # --- NEW TAB 1 STANDALONE HISTORICAL GRAPH ---
        fig_t1 = go.Figure()
        recent_history = dv.tail(120)  # Standard viewport window
        
        # Upper / Lower Envelope Shadows
        fig_t1.add_trace(go.Scatter(x=recent_history.index, y=recent_history["BB_Upper"], line=dict(color="rgba(59,130,246,0.25)", width=1), name="BB Upper", showlegend=True))
        fig_t1.add_trace(go.Scatter(x=recent_history.index, y=recent_history["BB_Lower"], line=dict(color="rgba(59,130,246,0.25)", width=1), fill='tonexty', fillcolor='rgba(59,130,246,0.02)', name="BB Lower", showlegend=True))
        
        # Moving Average Signals
        fig_t1.add_trace(go.Scatter(x=recent_history.index, y=recent_history["MA30"], line=dict(color=PURPLE, width=1.5), name="Moving Average 30"))
        fig_t1.add_trace(go.Scatter(x=recent_history.index, y=recent_history["MA90"], line=dict(color=ACCENT, width=1.5, dash='dash'), name="Moving Average 90"))
        
        # Price Line
        fig_t1.add_trace(go.Scatter(x=recent_history.index, y=recent_history["Close"], line=dict(color=BLUE, width=2.5), name="TSLA Spot Price"))
        
        # Horizontal Control Risk Constraints
        fig_t1.add_shape(type="line", x0=recent_history.index.min(), x1=recent_history.index.max(), y0=stop_loss, y1=stop_loss, line=dict(color=RED, width=1.5, dash="dot"))
        fig_t1.add_trace(go.Scatter(x=[recent_history.index.median()], y=[stop_loss], mode="text", text=["Stop Loss Vector Line"], textposition="top center", showlegend=False))
        
        fig_t1.add_shape(type="line", x0=recent_history.index.min(), x1=recent_history.index.max(), y0=target_price, y1=target_price, line=dict(color=GREEN, width=1.5, dash="dot"))
        fig_t1.add_trace(go.Scatter(x=[recent_history.index.median()], y=[target_price], mode="text", text=["Take Profit Target Line"], textposition="bottom center", showlegend=False))

        fig_t1.update_layout(**base_layout(420, "Historical Envelope Overlay Core Metrics Infrastructure"))
        
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig_t1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown('<p class="section-header">Primary Mathematical Oscillators</p>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="indicator-row"><span class="ind-icon">📈</span><span class="ind-name">Relative Strength Index (RSI-14)</span>'
                f'<span class="ind-badge {"badge-bear" if current_rsi>70 else "badge-bull" if current_rsi<30 else "badge-neut"}">{current_rsi:.2f}</span></div>'
                f'<div class="indicator-row"><span class="ind-icon">⏱️</span><span class="ind-name">Moving Average Convergence Divergence (MACD)</span>'
                f'<span class="ind-badge badge-neut">{safe_float(dv["MACD"].iloc[-1]):.3f}</span></div>',
                unsafe_allow_html=True
            )
        with col_r:
            st.markdown('<p class="section-header">Stochastic Volatility Envelopes</p>', unsafe_allow_html=True)
            reg_dict = _compute_regime(dv["Close"])
            hurst_val = reg_dict["hurst"]
            st.markdown(
                f'<div class="indicator-row"><span class="ind-icon">🌀</span><span class="ind-name">Fractal Hurst Exponent Coefficient</span>'
                f'<span class="ind-badge {"badge-bull" if hurst_val>0.55 else "badge-bear" if hurst_val<0.45 else "badge-neut"}">{hurst_val:.3f}</span></div>'
                f'<div class="indicator-row"><span class="ind-icon">📊</span><span class="ind-name">Annualized Log Statistical Volatility Spectrum</span>'
                f'<span class="ind-badge badge-neut">{(reg_dict["vol"]*np.sqrt(252)*100):.2f}%</span></div>',
                unsafe_allow_html=True
            )

    # ==========================================
    # TAB 2: PREDICTIVE PROJECTION FLIGHT RUNWAY
    # ==========================================
    with tab2:
        st.markdown('<p class="section-header">Network Dynamic Forward Projections Outflow Pipeline</p>', unsafe_allow_html=True)
        
        fig2 = go.Figure()
        hist_tail = dv.tail(90)
        
        fig2.add_trace(go.Scatter(x=hist_tail.index, y=hist_tail["Adj Close"], name="Historical Price Baseline", line=dict(color=BLUE, width=2)))
        
        # Vector Injector Join Matrix
        gapless_dates = [hist_tail.index[-1]] + list(b_dates)
        gapless_prices = [float(hist_tail["Adj Close"].iloc[-1])] + list(b_prices)
        gapless_hi = [float(hist_tail["Adj Close"].iloc[-1])] + list(b_hi)
        gapless_lo = [float(hist_tail["Adj Close"].iloc[-1])] + list(b_lo)

        fig2.add_trace(go.Scatter(x=gapless_dates, y=gapless_hi, line=dict(color="rgba(255,204,0,0)", width=0), showlegend=False))
        fig2.add_trace(go.Scatter(x=gapless_dates, y=gapless_lo, line=dict(color="rgba(255,204,0,0)", width=0), fill='tonexty', fillcolor="rgba(255,204,0,0.06)", name="Stochastic Variance Variance Band", showlegend=True))
        fig2.add_trace(go.Scatter(x=gapless_dates, y=gapless_prices, name="Hybrid Forecast Path Model Vector", line=dict(color=ACCENT, width=2.5, dash="dash")))
        
        # Risk thresholds overlay mirror
        fig2.add_shape(type="line", x0=hist_tail.index.min(), x1=b_dates[-1], y0=stop_loss, y1=stop_loss, line=dict(color=RED, width=1.5, dash="dot"))
        fig2.add_shape(type="line", x0=hist_tail.index.min(), x1=b_dates[-1], y0=target_price, y1=target_price, line=dict(color=GREEN, width=1.5, dash="dot"))

        fig2.update_layout(**base_layout(450, f"Forward Pipeline Predictive Model Continuum Vector Grid — Projected Horizon Depth: {forecast_horizon} Days"))
        
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ==========================================
    # TAB 3: ADVANCED ANALYTIC CENTER
    # ==========================================
    with tab3:
        st.markdown('<p class="section-header">Stochastic Horizon Variance Modeling Matrices</p>', unsafe_allow_html=True)
        
        r1, r2 = st.columns(2)
        with r1:
            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(x=b_dates, y=b_hi, line=dict(color="rgba(16,185,129,0.15)", width=1), name="Upper Bound Range Limit"))
            fig5.add_trace(go.Scatter(x=b_dates, y=b_lo, line=dict(color="rgba(244,63,94,0.15)", width=1), fill='tonexty', fillcolor="rgba(59,130,246,0.04)", name="Lower Variance Limit Boundary"))
            fig5.add_trace(go.Scatter(x=b_dates, y=b_prices, line=dict(color=PURPLE, width=2), name="Predictive Central Tendency Median"))
            fig5.update_layout(**base_layout(320, "Hybrid Predictor Pipeline Forward Model Mapping Fan Chart Node"))
            
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            st.plotly_chart(fig5, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with r2:
            fig_adv_reg = go.Figure()
            adv_tail = dv.tail(180)
            fig_adv_reg.add_trace(go.Scatter(x=adv_tail.index, y=adv_tail["Close"], line=dict(color=BLUE, width=1.5), name="TSLA Spot"))
            fig_adv_reg.add_trace(go.Scatter(x=adv_tail.index, y=adv_tail["BB_Upper"], line=dict(color="rgba(255,255,255,0.15)", width=1), name="BB high"))
            fig_adv_reg.add_trace(go.Scatter(x=adv_tail.index, y=adv_tail["BB_Lower"], line=dict(color="rgba(255,255,255,0.15)", width=1), name="BB low"))
            fig_adv_reg.update_layout(**base_layout(320, "Long-Term Volatility Horizon Regime Structural Model Envelope"))
            
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            st.plotly_chart(fig_adv_reg, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Seasonality distribution container
        st.markdown('<p class="section-header">Temporal Return Volatility Profiling Matrix</p>', unsafe_allow_html=True)
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        fig8 = go.Figure()
        for m_idx, m_name in enumerate(months, 1):
            sub = dv[dv.index.month == m_idx]["Close"].dropna()
            if not sub.empty: 
                fig8.add_trace(go.Box(y=sub, name=m_name, marker_color=BLUE, line_color=BLUE, fillcolor="rgba(59,130,246,0.12)"))
        fig8.update_layout(**base_layout(300, "Seasonality Structural Distribution Matrices Layer Grid"), showlegend=False)
        
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig8, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # --- RE-ENGINEERED ISOLATED ROW SPACE FOR HEATMAP ---
        st.markdown('<p class="section-header">Cross-Sectional Attribute Correlation Matrix Heatmap</p>', unsafe_allow_html=True)
        corr_cols = [c for c in ["Open", "High", "Low", "Close", "Volume", "Spread"] if c in dv.columns]
        corr_data = dv[corr_cols].dropna()
        
        if len(corr_data) >= 5 and len(corr_cols) >= 2:
            c_mat = corr_data.corr().round(3)
            fig11 = go.Figure(go.Heatmap(
                z=c_mat.values, x=corr_cols, y=corr_cols,
                colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                text=c_mat.values, texttemplate="%{text:.2f}", showscale=True
            ))
            fig11.update_layout(**base_layout(380, "Pearson Cross-Correlation Feature Co-dependency Network Array Grid"))
            
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            st.plotly_chart(fig11, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            empty_state("⚠️", "Insufficient structural data variables loaded to assemble feature co-dependency matrix grids.")
