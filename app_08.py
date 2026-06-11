# ============================================================
# TSLA FORECASTING HUB  |  app_08.py
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
   Slow-orbiting dual plasma blobs with fluid opacity
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

/* Beautiful fluid mesh behind all UI layers */
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

/* Futuristic Fine Dots Pattern Overlay (reactbits: DotGrid) */
.stApp > div {
  background-image: radial-gradient(circle, rgba(255,255,255,0.03) 1px, transparent 1px);
  background-size: 24px 24px;
  position: relative; z-index: 1;
}

/* ═══════════════════════════════════════════════
   CRITICAL SCROLLING FIX FOR THE SIDEBAR
═══════════════════════════════════════════════ */
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

[data-testid="stSidebarContent"]::-webkit-scrollbar {
  width: 5px;
}
[data-testid="stSidebarContent"]::-webkit-scrollbar-track {
  background: rgba(0,0,0,0.1);
}
[data-testid="stSidebarContent"]::-webkit-scrollbar-thumb {
  background: rgba(59,130,246,0.22);
  border-radius: 4px;
}
[data-testid="stSidebarContent"]::-webkit-scrollbar-thumb:hover {
  background: rgba(59,130,246,0.45);
}

/* ═══════════════════════════════════════════════
   TABS STYLE (Premium animated state)
═══════════════════════════════════════════════ */
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
.stTabs [data-baseweb="tab"]:hover {
  color: #ffaa11 !important;
}
.stTabs [aria-selected="true"] {
  color: #ffcc00 !important;
  border-bottom: 2px solid #ffcc00 !important;
  background: rgba(255, 204, 0, 0.04) !important;
  text-shadow: 0 0 12px rgba(255, 204, 0, 0.30);
}

/* ═══════════════════════════════════════════════
   SHINY TEXT GRADIENTS (reactbits: ShinyText)
═══════════════════════════════════════════════ */
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

/* ═══════════════════════════════════════════════
   METRIC CARDS (reactbits: Spotlight / Shimmer Card)
═══════════════════════════════════════════════ */
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
  background: linear-gradient(105deg,
    transparent 35%,
    rgba(59,130,246,0.08) 50%,
    transparent 65%);
  background-size: 200% 100%;
  animation: shimmer-sweep 4s linear infinite;
  pointer-events: none;
}
.metric-card:hover {
  border-color: rgba(59,130,246,0.45) !important;
  box-shadow: 0 8px 24px rgba(59,130,246,0.15);
  transform: translateY(-2px);
}
.metric-label {
  color: #94a3b8;
  font-size: 0.70rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin-bottom: 6px;
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 500;
}
.metric-value {
  color: #ffffff;
  font-size: 1.5rem;
  font-weight: 700;
  font-family: 'JetBrains Mono', monospace;
  word-break: break-all;
  text-shadow: 0 0 15px rgba(59,130,246,0.20);
}
.metric-delta-up   { color: #10b981; font-size: 0.82rem; margin-top: 4px; font-family: 'JetBrains Mono', monospace; font-weight: 500;}
.metric-delta-down { color: #f43f5e; font-size: 0.82rem; margin-top: 4px; font-family: 'JetBrains Mono', monospace; font-weight: 500;}
.metric-muted      { color: #64748b; font-size: 0.82rem; margin-top: 4px; }

/* ═══════════════════════════════════════════════
   PULSING SIGNAL BADGE
═══════════════════════════════════════════════ */
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
  background: rgba(16,185,129,0.1) !important;
  color: #10b981 !important;
  border: 2px solid rgba(16,185,129,0.7) !important;
  border-radius: 8px !important;
  padding: 10px 32px !important;
  font-weight: 700 !important;
  font-size: 1.6rem !important;
  font-family: 'Space Grotesk', monospace !important;
  letter-spacing: 0.1em !important;
  display: inline-block !important;
  animation: ring-buy 2s infinite !important;
}
.signal-sell {
  background: rgba(244,63,94,0.1) !important;
  color: #f43f5e !important;
  border: 2px solid rgba(244,63,94,0.7) !important;
  border-radius: 8px !important;
  padding: 10px 32px !important;
  font-weight: 700 !important;
  font-size: 1.6rem !important;
  font-family: 'Space Grotesk', monospace !important;
  letter-spacing: 0.1em !important;
  display: inline-block !important;
  animation: ring-sell 2s infinite !important;
}
.signal-hold {
  background: rgba(245,158,11,0.1) !important;
  color: #f59e0b !important;
  border: 2px solid rgba(245,158,11,0.7) !important;
  border-radius: 8px !important;
  padding: 10px 32px !important;
  font-weight: 700 !important;
  font-size: 1.6rem !important;
  font-family: 'Space Grotesk', monospace !important;
  letter-spacing: 0.1em !important;
  display: inline-block !important;
  animation: ring-hold 2s infinite !important;
}

/* ═══════════════════════════════════════════════
   SECTION HEADERS
═══════════════════════════════════════════════ */
@keyframes scanning-glow {
  0%   { left: -100%; }
  100% { left:  120%; }
}
.section-header {
  position: relative;
  color: #ffcc00;
  font-size: 0.72rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  margin-top: 24px;
  margin-bottom: 12px;
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 600;
  overflow: hidden;
  padding-bottom: 6px;
}
.section-header::after {
  content: "";
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(59,130,246,0.3), transparent);
}
.section-header::before {
  content: "";
  position: absolute;
  bottom: 0;
  width: 35%;
  height: 1px;
  background: linear-gradient(90deg, transparent, #ffcc00, transparent);
  animation: scanning-glow 4s linear infinite;
}

/* ═══════════════════════════════════════════════
   HEADER HERO STRIP
═══════════════════════════════════════════════ */
@keyframes border-sweep {
  0%   { background-position: 0% 50%; }
  50%  { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
.app-header-banner {
  position: relative;
  background: rgba(10,14,26,0.72);
  border: 1px solid rgba(59,130,246,0.22);
  border-radius: 16px;
  padding: 24px 28px;
  margin-bottom: 16px;
  overflow: hidden;
  backdrop-filter: blur(20px);
}
.app-header-banner::before {
  content: "";
  position: absolute;
  top: 0; left: 0; right: 0; height: 1.5px;
  background: linear-gradient(90deg, transparent, rgba(59,130,246,0.7), #ffcc00, transparent);
  background-size: 200% auto;
  animation: border-sweep 6s linear infinite;
}
.app-title {
  font-family: 'Space Grotesk', sans-serif;
  font-weight: 700;
  font-size: 1.5rem;
  color: #ffffff;
  letter-spacing: -0.01em;
}
.app-subtitle {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.70rem;
  color: #64748b;
  letter-spacing: 0.08em;
  margin-top: 4px;
}

/* ═══════════════════════════════════════════════
   INDICATOR COMPONENT
═══════════════════════════════════════════════ */
.indicator-row {
  display: flex; align-items: center; gap: 12px;
  padding: 10px 14px;
  margin-bottom: 6px;
  border-radius: 8px;
  background: rgba(255,255,255,0.015);
  border: 1px solid rgba(255,255,255,0.03);
  font-family: 'Inter', sans-serif;
  font-size: 0.82rem;
  color: #e2e8f0;
  transition: background 0.25s ease, border-color 0.25s ease;
}
.indicator-row:hover {
  background: rgba(59,130,246,0.05);
  border-color: rgba(59,130,246,0.15);
}
.ind-icon { font-size: 0.9rem; flex-shrink: 0; }
.ind-name { font-weight: 500; flex: 1; }
.ind-badge {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem;
  letter-spacing: 0.05em;
  padding: 3px 10px;
  border-radius: 6px;
  font-weight: 600;
}
.badge-bull { background: rgba(16,185,129,0.12); color: #10b981; border: 1px solid rgba(16,185,129,0.25); }
.badge-bear { background: rgba(244,63,94,0.12);  color: #f43f5e; border: 1px solid rgba(244,63,94,0.25); }
.badge-neut { background: rgba(245,158,11,0.10); color: #f59e0b; border: 1px solid rgba(245,158,11,0.20); }
.badge-warn { background: rgba(244,63,94,0.18);  color: #ffa5b5; border: 1px solid rgba(244,63,94,0.30); }

/* ═══════════════════════════════════════════════
   CHART DECORATION CONTAINER
═══════════════════════════════════════════════ */
.chart-wrap {
  position: relative;
  border: 1px solid rgba(59,130,246,0.16);
  border-radius: 12px;
  overflow: hidden;
  background: rgba(8,10,18,0.40);
  padding: 6px;
  transition: border-color 0.3s ease;
}
.chart-wrap:hover { border-color: rgba(59,130,246,0.35); }

.stButton > button {
  background: linear-gradient(135deg, #0b1530 0%, #152554 100%) !important;
  border: 1px solid rgba(59,130,246,0.40) !important;
  color: #60a5fa !important;
  border-radius: 8px !important;
  padding: 10px 24px !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.80rem !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
  width: 100%;
}
.stButton > button:hover {
  border-color: rgba(59,130,246,0.80) !important;
  box-shadow: 0 4px 15px rgba(59,130,246,0.3) !important;
  color: #93c5fd !important;
}

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
        margin=dict(l=40, r=20, t=40, b=30),
        title=dict(text=title, font=dict(size=12, color=MUTED, family="Space Grotesk")),
        xaxis=dict(gridcolor=GRID_COL, showgrid=True, linecolor=GRID_COL),
        yaxis=dict(gridcolor=GRID_COL, showgrid=True, linecolor=GRID_COL),
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
        warnings_out.append("TSLA_1.csv unified dataset not found. Initializing historical simulation data automatically.")
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
# ║                    MODEL LAYER                           ║
# ╚══════════════════════════════════════════════════════════╝

def extract_gdrive_id(url: str) -> str | None:
    url = url.strip()
    for pattern in [
        r"/file/d/([a-zA-Z0-9_-]{20,})",
        r"[?&]id=([a-zA-Z0-9_-]{20,})",
        r"/d/([a-zA-Z0-9_-]{20,})/",
        r"^([a-zA-Z0-9_-]{20,})$",
    ]:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return None

@st.cache_resource(show_spinner=False)
def load_model_cached(file_id: str):
    try: import gdown
    except ImportError: raise RuntimeError("'gdown' is missing from requirements.txt.")
    try: import tensorflow as tf
    except ImportError: raise RuntimeError("'tensorflow' is missing from requirements.txt.")

    download_url = f"https://drive.google.com/uc?id={file_id}"
    tmp_path     = os.path.join(tempfile.gettempdir(), f"tsla_model_{file_id[:8]}.keras")

    if not os.path.exists(tmp_path):
        result = gdown.download(download_url, tmp_path, quiet=True)
        if result is None or not os.path.exists(tmp_path):
            raise RuntimeError("Download failed. Verify link share options.")

    try:
        model = tf.keras.models.load_model(tmp_path)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"Model load exception: {e}")

    return model

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
            if not chunks:
                continue
            rs_chunk = []
            for c in chunks:
                mean_c  = np.mean(c)
                deviate = np.cumsum(c - mean_c)
                r       = deviate.max() - deviate.min()
                s       = np.std(c, ddof=1)
                if s > 0:
                    rs_chunk.append(r / s)
            if rs_chunk:
                rs_vals.append((lag, np.mean(rs_chunk)))
        if len(rs_vals) < 2:
            return 0.5
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
        else:
            ou_speed = 0.05
    except Exception:
        ou_speed = 0.05

    return dict(hurst=H, trend_slope=slope, mean_price=mean_price, vol=vol, ou_speed=ou_speed)

def _single_step_forecast(model, scaler, lookback_context: list) -> float:
    x_sc = scaler.transform(np.array(lookback_context[-LOOKBACK:]).reshape(-1, 1)).flatten()
    x_in = np.array(x_sc, dtype=np.float32).reshape(1, LOOKBACK, 1)
    raw = float(np.clip(model.predict(x_in, verbose=0)[0, 0], 0.0, 1.0))
    return float(scaler.inverse_transform([[raw]])[0, 0])

def _revise_prediction(
    raw_pred: float, prev_price: float, anchor_price: float, regime: dict, step: int, total_steps: int
) -> float:
    H = regime["hurst"]
    mean_price = regime["mean_price"]
    ou_speed = regime["ou_speed"]
    vol = regime["vol"]

    if prev_price <= 0 or anchor_price <= 0:
        return raw_pred

    raw_log_ret = np.log(max(raw_pred, 1e-6) / max(prev_price, 1e-6))
    cumulative_log_ret = np.log(max(prev_price, 1e-6) / max(anchor_price, 1e-6))

    drift_budget = 2.0 * vol * np.sqrt(step)
    drift_excess = abs(cumulative_log_ret) - drift_budget

    if drift_excess > 0:
        correction_sign = -np.sign(cumulative_log_ret)
        drift_correction = correction_sign * drift_excess * 0.15
    else:
        drift_correction = 0.0

    price_gap_pct = abs(prev_price - mean_price) / mean_price
    if price_gap_pct > 0.05:
        ou_pull = -ou_speed * np.log(max(prev_price, 1e-6) / max(mean_price, 1e-6))
        ou_scale = float(np.interp(H, [0.35, 0.50, 0.65], [0.35, 0.20, 0.08]))
    else:
        ou_pull = 0.0
        ou_scale = 0.0

    final_log_ret = raw_log_ret + drift_correction + (ou_scale * ou_pull)
    final_log_ret = float(np.clip(final_log_ret, -4.0 * vol, 4.0 * vol))

    revised_price = prev_price * np.exp(final_log_ret)
    revised_price = float(np.clip(revised_price, mean_price * 0.20, mean_price * 3.50))
    return revised_price

def dynamic_timeline_forecasting(model, scaler, df: pd.DataFrame, start_date: pd.Timestamp, n_days: int) -> tuple:
    db_max_date = df.index.max()
    target_start = pd.Timestamp(start_date)
    biz_dates = pd.bdate_range(start=target_start, periods=n_days)

    recent_ret = df["DailyReturn"].replace([np.inf, -np.inf], np.nan).dropna().tail(60)
    daily_vol = (recent_ret.std() / 100) if len(recent_ret) >= 5 else 0.02

    preds_prices = []
    bridge_dates, bridge_prices, bridge_lo, bridge_hi = [], [], [], []

    if target_start <= db_max_date:
        anchor_price_a = safe_float(df["Adj Close"].iloc[-1])
        future_step = 0
        for curr_date in biz_dates:
            if curr_date <= db_max_date:
                pos_idx = df.index.get_indexer([curr_date], method="pad")[0]
                pos_idx = max(pos_idx, 0)
                preds_prices.append(float(df.iloc[pos_idx]["Adj Close"]))
            else:
                future_step += 1
                pos_idx = df.index.get_indexer([curr_date], method="pad")[0]
                if pos_idx != -1 and pos_idx >= LOOKBACK:
                    history_slice = df.iloc[:pos_idx]
                else:
                    history_slice = df.head(LOOKBACK) if len(df) >= LOOKBACK else df

                regime = _compute_regime(history_slice["Adj Close"])
                lookback_context = list(df.iloc[:pos_idx]["Adj Close"].tail(LOOKBACK).values)
                
                while len(lookback_context) < LOOKBACK:
                    lookback_context.insert(0, lookback_context[0] if lookback_context else anchor_price_a)

                raw_f = _single_step_forecast(model, scaler, lookback_context)
                prev_p = preds_prices[-1] if preds_prices else anchor_price_a
                revised_f = _revise_prediction(raw_f, prev_p, anchor_price_a, regime, future_step, n_days)
                preds_prices.append(revised_f)

                bridge_dates.append(curr_date)
                bridge_prices.append(revised_f)
                spread = revised_f * (daily_vol * np.sqrt(future_step))
                bridge_lo.append(revised_f - spread)
                bridge_hi.append(revised_f + spread)

        return biz_dates, np.array(preds_prices), bridge_dates, bridge_prices, bridge_lo, bridge_hi

    else:
        anchor_price_b = safe_float(df["Adj Close"].iloc[-1])
        regime = _compute_regime(df["Adj Close"])
        lookback_context = list(df["Adj Close"].tail(LOOKBACK).values)
        
        gap_dates = pd.bdate_range(start=db_max_date + pd.Timedelta(days=1), end=target_start - pd.Timedelta(days=1))
        sim_step = 0

        for _ in gap_dates:
            sim_step += 1
            raw_f = _single_step_forecast(model, scaler, lookback_context)
            prev_p = lookback_context[-1] if lookback_context else anchor_price_b
            revised_f = _revise_prediction(raw_f, prev_p, anchor_price_b, regime, sim_step, len(gap_dates) + n_days)
            lookback_context.append(revised_f)
            lookback_context.pop(0)

        future_step = sim_step
        for curr_date in biz_dates:
            future_step += 1
            raw_f = _single_step_forecast(model, scaler, lookback_context)
            prev_p = lookback_context[-1] if lookback_context else anchor_price_b
            revised_f = _revise_prediction(raw_f, prev_p, anchor_price_b, regime, future_step, len(gap_dates) + n_days)
            
            preds_prices.append(revised_f)
            lookback_context.append(revised_f)
            lookback_context.pop(0)

            bridge_dates.append(curr_date)
            bridge_prices.append(revised_f)
            spread = revised_f * (daily_vol * np.sqrt(future_step))
            bridge_lo.append(revised_f - spread)
            bridge_hi.append(revised_f + spread)

        return biz_dates, np.array(preds_prices), bridge_dates, bridge_prices, bridge_lo, bridge_hi

# ╔══════════════════════════════════════════════════════════╗
# ║                    APPLICATION ENTRY                     ║
# ╚══════════════════════════════════════════════════════════╝

dv, flags = load_data()

st.markdown(
    f'<div class="app-header-banner">'
    f'<div class="app-title"><span class="shiny-text">⚡ TSLA Deep-Quant Hybrid Pipeline</span></div>'
    f'<div class="app-subtitle">Recurrent Convolutional Network + Fractal Hurst Regimes + OU Damping</div>'
    f'</div>',
    unsafe_allow_html=True,
)

for msg in flags:
    st.sidebar.warning(msg)

if dv.empty:
    empty_state("🚨", "System Data Layer Malfunction. Check formatting constraints.")
    st.stop()

# ── Sidebar Input Configuration Controls ──────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-header">Account Exposure Parameters</p>', unsafe_allow_html=True)
    
    default_entry = float(dv["Close"].iloc[-1]) if not dv.empty else 220.0
    
    entry_price = st.number_input(
        "Position Entry Price ($)", 
        min_value=1.0, 
        max_value=10000.0, 
        value=round(default_entry, 2), 
        step=0.5,
        help="Specify the asset baseline cost basis value."
    )
    
    share_quantity = st.number_input(
        "Position Share Quantity", 
        min_value=1, 
        max_value=1000000, 
        value=10, 
        step=5,
        help="Specify total held units exposed within the pipeline horizon."
    )
    
    total_exposure = entry_price * share_quantity
    st.markdown(
        f'<div style="font-size:0.75rem; color:#64748b; margin-top:-10px; padding-left:2px;">'
        f'Active Notional Value: <span style="color:#ffcc00; font-family:\'JetBrains Mono\'">${total_exposure:,.2f}</span></div>', 
        unsafe_allow_html=True
    )
    st.markdown("---")

    st.markdown('<p class="section-header">Execution Target Vector</p>', unsafe_allow_html=True)
    gdrive_link = st.text_input("Google Drive Model Path", value=get_secret_model_link())
    
    anchor_dt = st.date_input("Projection Mapping Node Anchor", value=dv.index.max())
    projection_horizon = st.slider("Projection Days Window", min_value=5, max_value=90, value=30, step=5)

model_id = extract_gdrive_id(gdrive_link) if gdrive_link else None

if not model_id:
    st.sidebar.info("Awaiting structural neural signature. Provide link node validation key.")
    empty_state("🧠", "Awaiting quantitative deep-learning deployment signature key within the control center.")
    st.stop()

try:
    nn_engine = load_model_cached(model_id)
    data_scaler = build_scaler(dv)
except Exception as ex:
    empty_state("❌", f"Structural core failure executing initialization vector: {ex}")
    st.stop()

# Compute active simulation data parameters
t_anchor = pd.Timestamp(anchor_dt)
f_dates, f_prices, b_dates, b_prices, b_lo, b_hi = dynamic_timeline_forecasting(
    nn_engine, data_scaler, dv, t_anchor, projection_horizon
)
regime_metrics = _compute_regime(dv["Adj Close"])

# Establish Tab Components
tab1, tab2, tab3 = st.tabs(["System Signals Matrix", "Predictive Projection Flight", "Advanced Analytic Center"])

# ╔══════════════════════════════════════════════════════════╗
# ║                    TAB 1: Core System Matrix             ║
# ╚══════════════════════════════════════════════════════════╝
with tab1:
    st.markdown('<p class="section-header">Execution Layer Metric Anchors</p>', unsafe_allow_html=True)
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(metric_card("Current Close", f"${dv['Close'].iloc[-1]:,.2f}", "Live Database Value", "metric-muted"), unsafe_allow_html=True)
    with m2:
        st.markdown(metric_card("Position Size", f"{share_quantity:,} Units", f"Basis: ${entry_price:,.2f}", "metric-muted"), unsafe_allow_html=True)
    with m3:
        pnl = (dv['Close'].iloc[-1] - entry_price) * share_quantity
        pnl_pct = ((dv['Close'].iloc[-1] - entry_price) / entry_price) * 100
        cls, sign = ("metric-delta-up", "+") if pnl >= 0 else ("metric-delta-down", "")
        st.markdown(metric_card("Unrealized P&L", f"{sign}${pnl:,.2f}", f"{sign}{pnl_pct:.2f}% Basis Delta", cls), unsafe_allow_html=True)
    with m4:
        signal_text = "BUY" if dv["RSI"].iloc[-1] < 45 else ("SELL" if dv["RSI"].iloc[-1] > 65 else "HOLD")
        sig_badge = f"signal-{signal_text.lower()}"
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Pipeline Consensus Node</div>'
            f'<div class="{sig_badge}" style="font-size:1.1rem !important; padding:4px 12px !important; margin-top:4px;">{signal_text}</div></div>',
            unsafe_allow_html=True
        )

    st.markdown('<p class="section-header">System Signal Nodes Matrix</p>', unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(
            f'<div class="indicator-row"><span class="ind-icon">📈</span><span class="ind-name">Hurst Regime Structural Exponent</span>'
            f'<span class="ind-badge badge-bull">{regime_metrics["hurst"]:.2f} Alpha</span></div>', unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="indicator-row"><span class="ind-icon">📊</span><span class="ind-name">Relative Strength Index (RSI-14)</span>'
            f'<span class="ind-badge {"badge-bear" if dv["RSI"].iloc[-1] > 70 else "badge-bull" if dv["RSI"].iloc[-1] < 30 else "badge-neut"}">{dv["RSI"].iloc[-1]:.1f}</span></div>', unsafe_allow_html=True
        )
    with col_r:
        st.markdown(
            f'<div class="indicator-row"><span class="ind-icon">🔄</span><span class="ind-name">Ornstein-Uhlenbeck Mean-Reversion Damping Speed</span>'
            f'<span class="ind-badge badge-neut">{regime_metrics["ou_speed"]:.3f} λ</span></div>', unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="indicator-row"><span class="ind-icon">📉</span><span class="ind-name">Moving Average Convergence Divergence (MACD)</span>'
            f'<span class="ind-badge {"badge-bull" if dv["MACDHist"].iloc[-1] > 0 else "badge-bear"}">Hist Delta: {dv["MACDHist"].iloc[-1]:.3f}</span></div>', unsafe_allow_html=True
        )

# ╔══════════════════════════════════════════════════════════╗
# ║                    TAB 2: Simplified Projections         ║
# ╚══════════════════════════════════════════════════════════╝
with tab2:
    st.markdown('<p class="section-header">Predictive Model Structural Projections Flight Path</p>', unsafe_allow_html=True)
    
    # Trace logic corrected here to capture the last point of history to close the gap
    hist_tail = dv.tail(LOOKBACK)
    last_known_date = hist_tail.index[-1]
    last_known_val = float(hist_tail["Adj Close"].iloc[-1])

    # Prepend history edge to prediction arrays to ensure perfect mathematical convergence
    gapless_dates = [last_known_date] + list(f_dates)
    gapless_prices = [last_known_val] + list(f_prices)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=hist_tail.index, y=hist_tail["Adj Close"],
        name="Historical Baseline actuals", line=dict(color=BLUE, width=2)
    ))
    fig2.add_trace(go.Scatter(
        x=gapless_dates, y=gapless_prices,
        name="Network Dynamic Projections Outflow", line=dict(color=ACCENT, width=2.5, dash="dash")
    ))
    fig2.update_layout(**base_layout(420, "Continuous Forecast Runway Node Integration Framework"))
    
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════╗
# ║                    TAB 3: Advanced Analytic Center       ║
# ╚══════════════════════════════════════════════════════════╝
with tab3:
    st.markdown('<p class="section-header">Unified Quant Analytics Hub & Subtrace Clusters</p>', unsafe_allow_html=True)
    
    # ── GRAPH 5 AND GRAPH REGIME SETUP ─────────────────────────────────────────
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=dv.index[-LOOKBACK:], y=dv["Adj Close"].tail(LOOKBACK), name="Context Historical Actuals", line=dict(color=BLUE, width=2)))
    if b_dates:
        fig5.add_trace(go.Scatter(x=[last_known_date] + list(b_dates), y=[last_known_val] + list(b_prices), name="Hybrid Recursive Mean", line=dict(color=ACCENT, width=2.5)))
        fig5.add_trace(go.Scatter(x=b_dates, y=b_hi, name="Upper Sigma Boundary Bound", line=dict(width=0), showlegend=False))
        fig5.add_trace(go.Scatter(x=b_dates, y=b_lo, name="Lower Sigma Boundary Bound", fill='tonexty', fillcolor="rgba(255,204,0,0.06)", line=dict(width=0), showlegend=False))
    # Corrected internal parameter layout assignment format!
    fig5.update_layout(**base_layout(400, "Hybrid Predictor Pipeline Forward Projection Mapping Node", override_yaxis=dict(tickprefix="$")))

    fig_adv_reg = go.Figure()
    fig_adv_reg.add_trace(go.Scatter(x=dv.index[-120:], y=dv["Adj Close"].tail(120), name="Asset Underlying Actuals Path", line=dict(color="rgba(255,255,255,0.4)", width=1.5)))
    fig_adv_reg.add_trace(go.Scatter(x=dv.index[-120:], y=dv["MA30"].tail(120), name="Micro-Trend Node (MA30)", line=dict(color=BLUE, width=1.2, dash="dot")))
    fig_adv_reg.add_trace(go.Scatter(x=dv.index[-120:], y=dv["BB_Upper"].tail(120), name="Bollinger Top Alpha Boundary", line=dict(color="rgba(16,185,129,0.25)", width=1)))
    fig_adv_reg.add_trace(go.Scatter(x=dv.index[-120:], y=dv["BB_Lower"].tail(120), name="Bollinger Bottom Alpha Boundary", line=dict(color="rgba(244,63,94,0.25)", width=1)))
    # Corrected internal parameter layout assignment format!
    fig_adv_reg.update_layout(**base_layout(400, "Advanced Analytic 2: Long-Term Horizon Regime Projections Grid", override_yaxis=dict(tickprefix="$")))

    # Display row 1
    r1, r2 = st.columns(2)
    with r1:
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with r2:
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig_adv_reg, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── BOX PLOTS AND HEATMAP SUBSURFACE PLOTS ─────────────────────────────────
    st.markdown('<p class="section-header">Cross-Sectional Factor Correlations & Seasonal Distribution Matrices</p>', unsafe_allow_html=True)
    
    r3, r4 = st.columns(2)
    with r3:
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        fig8 = go.Figure()
        for m_idx, m_name in enumerate(months, 1):
            sub = dv[dv.index.month == m_idx]["Close"].dropna()
            if not sub.empty: 
                fig8.add_trace(go.Box(y=sub, name=m_name, marker_color=BLUE, line_color=BLUE, fillcolor="rgba(59,130,246,0.12)"))
        fig8.update_layout(**base_layout(400, "Seasonality Structural Distribution Matrices"), showlegend=False)
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig8, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r4:
        corr_cols = [c for c in ["Open", "High", "Low", "Close", "Volume", "Spread"] if c in dv.columns]
        corr_data = dv[corr_cols].dropna()
        if len(corr_data) >= 5 and len(corr_cols) >= 2:
            c_mat = corr_data.corr().round(3)
            fig11 = go.Figure(go.Heatmap(
                z=c_mat.values, x=corr_cols, y=corr_cols, colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                text=c_mat.values, texttemplate="%{text:.2f}", showscale=True
            ))
            fig11.update_layout(**base_layout(400, "Cross-Sectional Attribute Correlation Matrix Heatmap"))
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            st.plotly_chart(fig11, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            empty_state("📊", "Insufficient data vector dimensions to isolate core cross-sectional attributes mapping.")
