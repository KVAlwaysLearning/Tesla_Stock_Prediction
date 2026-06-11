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
   Streamlit sidebars lock overflow by default,
   preventing input parameter access. We force scroll!
═══════════════════════════════════════════════ */
[data-testid="stSidebar"] {
  background: rgba(8, 10, 16, 0.94) !important;
  border-right: 1px solid rgba(59,130,246,0.18) !important;
  backdrop-filter: blur(24px) !important;
  -webkit-backdrop-filter: blur(24px) !important;
}

/* Targets the core structural containers in modern Streamlit to allow scrolling */
[data-testid="stSidebarContent"], 
[data-testid="stSidebarUserContent"], 
section[data-testid="stSidebar"] > div {
  overflow-y: auto !important;
  max-height: 100vh !important;
}

/* Customizable scrollbar for sleek cyber look */
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
   PULSING SIGNAL BADGE (reactbits: Glowing Ring)
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
   SECTION HEADERS (Scanning light animation)
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
   HEADER HERO STRIP (reactbits: StarBorder vibe)
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

/* ═══════════════════════════════════════════════
   SLIDER, BUTTONS, SELECTIONS (Clean aesthetic adjustments)
═══════════════════════════════════════════════ */
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
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, #10204d 0%, #1e3a8a 100%) !important;
  border-color: rgba(59,130,246,0.60) !important;
  color: #93c5fd !important;
}

/* Miscellaneous corrections */
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
        warnings_out.append("TSLA_1.csv unified dataset not found in app folder.")
        return pd.DataFrame(), warnings_out

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
# ║  Return-space recursion + Hurst regime detection +       ║
# ║  Ornstein-Uhlenbeck mean-reversion damping               ║
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
        return dict(hurst=0.5, trend_slope=0.0, mean_price=prices[-1],
                    vol=0.02, ou_speed=0.05)

    log_rets  = np.diff(np.log(prices))
    vol       = float(np.std(log_rets)) if len(log_rets) > 1 else 0.02
    H         = _hurst_exponent(prices)

    x          = np.arange(len(prices))
    log_prices = np.log(prices)
    slope      = float(np.polyfit(x, log_prices, 1)[0]) * 252  # annualised

    ema_span   = min(200, len(prices))
    weights    = np.exp(np.linspace(-1, 0, ema_span))
    weights   /= weights.sum()
    mean_price = float(np.convolve(prices, weights[::-1], mode='valid')[-1])

    try:
        if len(log_rets) > 10:
            phi    = np.corrcoef(log_rets[:-1], log_rets[1:])[0, 1]
            phi    = np.clip(phi, -0.99, 0.99)
            ou_speed = float(-np.log(abs(phi))) if abs(phi) < 1.0 else 0.05
        else:
            ou_speed = 0.05
    except Exception:
        ou_speed = 0.05

    return dict(hurst=H, trend_slope=slope, mean_price=mean_price,
                vol=vol, ou_speed=ou_speed)

def _single_step_forecast(model, scaler, lookback_context: list) -> float:
    x_sc  = scaler.transform(np.array(lookback_context[-LOOKBACK:]).reshape(-1, 1)).flatten()
    x_in  = np.array(x_sc, dtype=np.float32).reshape(1, LOOKBACK, 1)
    raw   = float(np.clip(model.predict(x_in, verbose=0)[0, 0], 0.0, 1.0))
    return float(scaler.inverse_transform([[raw]])[0, 0])

def _revise_prediction(
    raw_pred:       float,
    prev_price:     float,
    anchor_price:   float,
    regime:         dict,
    step:           int,
    total_steps:    int,
) -> float:
    H           = regime["hurst"]
    mean_price  = regime["mean_price"]
    ou_speed    = regime["ou_speed"]
    vol         = regime["vol"]

    if prev_price <= 0 or anchor_price <= 0:
        return raw_pred

    raw_log_ret = np.log(max(raw_pred, 1e-6) / max(prev_price, 1e-6))
    cumulative_log_ret = np.log(max(prev_price, 1e-6) / max(anchor_price, 1e-6))

    drift_budget = 2.0 * vol * np.sqrt(step)
    drift_excess = abs(cumulative_log_ret) - drift_budget
    if drift_excess > 0:
        correction_sign     = -np.sign(cumulative_log_ret)
        drift_correction    = correction_sign * drift_excess * 0.15
    else:
        drift_correction = 0.0

    price_gap_pct = abs(prev_price - mean_price) / mean_price
    if price_gap_pct > 0.05:
        ou_pull = -ou_speed * np.log(max(prev_price, 1e-6) / max(mean_price, 1e-6))
        ou_scale = float(np.interp(H, [0.35, 0.50, 0.65], [0.35, 0.20, 0.08]))
    else:
        ou_pull  = 0.0
        ou_scale = 0.0

    final_log_ret = raw_log_ret + drift_correction + (ou_scale * ou_pull)
    final_log_ret = float(np.clip(final_log_ret, -4.0 * vol, 4.0 * vol))

    revised_price = prev_price * np.exp(final_log_ret)
    revised_price = float(np.clip(revised_price, mean_price * 0.20, mean_price * 3.50))
    return revised_price

def dynamic_timeline_forecasting(
    model, scaler, df: pd.DataFrame, start_date: pd.Timestamp, n_days: int
) -> tuple:
    db_max_date  = df.index.max()
    target_start = pd.Timestamp(start_date)
    biz_dates    = pd.bdate_range(start=target_start, periods=n_days)

    recent_ret = df["DailyReturn"].replace([np.inf, -np.inf], np.nan).dropna().tail(60)
    daily_vol  = (recent_ret.std() / 100) if len(recent_ret) >= 5 else 0.02

    preds_prices: list = []
    bridge_dates, bridge_prices, bridge_lo, bridge_hi = [], [], [], []

    # ── PATH A: start_date is within or at the database boundary ─────────────
    if target_start <= db_max_date:
        anchor_price_a = safe_float(df["Adj Close"].iloc[-1])
        future_step    = 0

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
                    history_slice = df.head(LOOKBACK)

                lookback_context = history_slice.tail(LOOKBACK)["Adj Close"].tolist()
                idx_offset = len(preds_prices) - 1
                while len(lookback_context) < LOOKBACK and idx_offset >= 0:
                    lookback_context.insert(0, preds_prices[idx_offset])
                    idx_offset -= 1

                raw_pred = _single_step_forecast(model, scaler, lookback_context)
                regime   = _compute_regime(history_slice.tail(252)["Adj Close"])
                prev_p   = lookback_context[-1]
                revised  = _revise_prediction(
                    raw_pred, prev_p, anchor_price_a, regime, future_step, n_days
                )
                preds_prices.append(revised)

    # ── PATH B: start_date is strictly beyond the database boundary ───────────
    else:
        working_df = df[["Adj Close"]].copy()
        gap_range  = pd.bdate_range(
            start=db_max_date + pd.Timedelta(days=1),
            end=target_start - pd.Timedelta(days=1),
        )

        if len(gap_range) > 0:
            regime_bridge  = _compute_regime(working_df.tail(252)["Adj Close"])
            anchor_bridge  = safe_float(working_df["Adj Close"].iloc[-1])
            total_bridge   = len(gap_range)

            for b_step, g_date in enumerate(gap_range, start=1):
                seed_vals = working_df.tail(LOOKBACK)["Adj Close"].tolist()
                raw_pred  = _single_step_forecast(model, scaler, seed_vals)
                prev_p    = seed_vals[-1]
                revised   = _revise_prediction(
                    raw_pred, prev_p, anchor_bridge, regime_bridge, b_step, total_bridge
                )

                working_df.loc[g_date, "Adj Close"] = revised
                bridge_dates.append(g_date)
                bridge_prices.append(revised)

                band_frac = np.clip(daily_vol * np.sqrt(b_step), 0, 0.45)
                bridge_lo.append(revised * (1 - band_frac))
                bridge_hi.append(revised * (1 + band_frac))

        regime_target = _compute_regime(working_df.tail(252)["Adj Close"])
        anchor_target = safe_float(working_df["Adj Close"].iloc[-1])
        for t_step, b_date in enumerate(biz_dates, start=1):
            seed_vals = working_df.tail(LOOKBACK)["Adj Close"].tolist()
            raw_pred  = _single_step_forecast(model, scaler, seed_vals)
            prev_p    = seed_vals[-1]
            revised   = _revise_prediction(
                raw_pred, prev_p, anchor_target, regime_target, t_step, n_days
            )
            preds_prices.append(revised)
            working_df.loc[b_date, "Adj Close"] = revised

    # ── Confidence bands ─────────────────────────────────────────────────────
    preds_prices = np.array(preds_prices, dtype=np.float32)
    lower_bounds, upper_bounds = [], []

    bridge_steps_count = len(bridge_prices)
    for idx in range(len(preds_prices)):
        total_depth = bridge_steps_count + idx + 1
        band_frac   = np.clip(daily_vol * np.sqrt(total_depth), 0, 0.45)
        lower_bounds.append(preds_prices[idx] * (1 - band_frac))
        upper_bounds.append(preds_prices[idx] * (1 + band_frac))

    return (
        biz_dates,
        preds_prices,
        np.array(lower_bounds),
        np.array(upper_bounds),
        pd.DatetimeIndex(bridge_dates),
        np.array(bridge_prices),
        np.array(bridge_lo),
        np.array(bridge_hi),
    )

# ╔══════════════════════════════════════════════════════════╗
# ║                    LOAD INITIALIZER                      ║
# ╚══════════════════════════════════════════════════════════╝

with st.spinner("Decoding dataset configuration matrix…"):
    df, data_warnings = load_data()

if df.empty:
    st.error("⛔ Dataset parsing exception. Check structure status.")
    st.stop()

for w in data_warnings:
    st.warning(f"⚠️ {w}")

current_price  = safe_float(df["Close"].iloc[-1])
prev_price     = safe_float(df["Close"].iloc[-2], fallback=current_price)
price_change   = current_price - prev_price
price_change_p = (price_change / prev_price * 100) if prev_price != 0 else 0.0

try:
    scaler = build_scaler(df)
    scaler_ok = True
except Exception as _scaler_err:
    scaler_ok = False
    st.warning(f"⚠️ Scaler core fault: {_scaler_err}")

# ╔══════════════════════════════════════════════════════════╗
# ║                    SIDEBAR CONTROLS                      ║
# ╚══════════════════════════════════════════════════════════╝

# Critical configuration sidebar with optimized scrolling styling and interactive nodes
with st.sidebar:
    st.markdown("## <span class='shiny-text'>⚡ TSLA TRANSCEIVER</span>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<p class="section-header">Model Engine Control</p>', unsafe_allow_html=True)
    gdrive_url = st.text_input(
        "Network Cloud Model Link", value=secret_url, type="password",
        help="Model artifact download link"
    )
    load_btn = st.button("⬇ INITIALIZE CORE ENGINE", type="primary")

    model_status_slot = st.empty()
    if st.session_state.get("model_loaded", False):
        model_status_slot.success("🟢 CORE PROTOCOLS ONLINE")
    else:
        model_status_slot.info("⏳ Core offline. Connect transceivers.")

    st.markdown("---")
    st.markdown('<p class="section-header">Forecast Space parameters</p>', unsafe_allow_html=True)
    chosen_start_date = st.date_input("Anchor Execution Date", value=df.index[-1].date())
    forecast_days     = st.slider("Forecast Temporal Reach", 5, 60, 30, 5)

    st.markdown("---")
    st.markdown('<p class="section-header">Alpha Risk matrix</p>', unsafe_allow_html=True)
    entry_price  = st.number_input("Target Entry Level ($)", min_value=0.0, value=0.0, step=0.01)
    position_qty = st.number_input("Unit Target Density", min_value=1, value=10, step=1)
    risk_pct     = st.slider("Risk Tolerance Clip (%)", 1, 20, 5)

if not st.session_state.get("model_loaded", False):
    url_to_load = gdrive_url.strip() if gdrive_url else secret_url.strip()
    if url_to_load:
        f_id = extract_gdrive_id(url_to_load)
        if f_id:
            try:
                st.session_state["model_obj"] = load_model_cached(f_id)
                st.session_state["model_loaded"] = True
                model_status_slot.success("🟢 CORE PROTOCOLS ONLINE")
            except Exception:
                st.session_state["model_loaded"] = False

if load_btn:
    url_clean = gdrive_url.strip() if gdrive_url else ""
    if url_clean:
        f_id = extract_gdrive_id(url_clean)
        if f_id:
            with st.sidebar:
                with st.spinner("Downloading dynamic core matrix..."):
                    try:
                        st.session_state["model_obj"] = load_model_cached(f_id)
                        st.session_state["model_loaded"] = True
                        model_status_slot.success("✅ DEPLOYMENT STABILIZED")
                    except Exception as _me:
                        st.error(f"❌ {_me}")

model = st.session_state.get("model_obj", None)
eff_entry = entry_price if entry_price > 0.0 else current_price

f_dates, f_prices, f_lower, f_upper = None, None, None, None
b_dates, b_prices, b_lower, b_upper = None, None, None, None

if model is not None and scaler_ok:
    with st.spinner("Executing structural timeline simulations..."):
        try:
            f_dates, f_prices, f_lower, f_upper, b_dates, b_prices, b_lower, b_upper = dynamic_timeline_forecasting(
                model, scaler, df, pd.Timestamp(chosen_start_date), n_days=forecast_days
            )
        except Exception as _err:
            st.error(f"Matrix strategy fault: {_err}")

# ╔══════════════════════════════════════════════════════════╗
# ║                    MAIN DASHBOARD HERO                   ║
# ╚══════════════════════════════════════════════════════════╝

st.markdown("""
<div class="app-header-banner">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 class="app-title"><span class="shiny-text">TSLA HYBRID FORECAST TERMINAL</span></h1>
            <p class="app-subtitle">CNN-GRU NEURAL PREDICTIONS • HURST EXPOSURE REGIME COMPLIANCE • RECTIFIED DECAY SYSTEM</p>
        </div>
        <div style="text-align: right;">
            <div style="background: rgba(59,130,246,0.12); border: 1px solid rgba(59,130,246,0.30); padding: 4px 12px; border-radius: 6px;">
                <span class="live-dot"></span><span style="font-family: 'JetBrains Mono', monospace; font-size: 0.70rem; color: #60a5fa; letter-spacing: 0.1em; font-weight: 600;">TRANSCEIVER LIVE</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════╗
# ║                    KPI METRICS BANNER                    ║
# ╚══════════════════════════════════════════════════════════╝

col_t, col_p, col_d, col_v, col_sp = st.columns([2.5, 2, 2, 2, 2.5])
with col_t:
    st.markdown(metric_card("Database Terminal Target", "TSLA (NASDAQ)", f"Sync Anchor: {df.index[-1].strftime('%d %b %Y')}", "metric-muted"), unsafe_allow_html=True)
with col_p:
    d_cls = "metric-delta-up" if price_change >= 0 else "metric-delta-down"
    arrow = "▲" if price_change >= 0 else "▼"
    st.markdown(metric_card("Last Trading Close", f"${current_price:.2f}", f"{arrow} ${abs(price_change):.2f} ({price_change_p:+.2f}%)", d_cls), unsafe_allow_html=True)
with col_d:
    st.markdown(metric_card("52-Week High Threshold", f"${safe_float(df['High'].tail(252).max()):.2f}"), unsafe_allow_html=True)
with col_v:
    st.markdown(metric_card("52-Week Low Threshold", f"${safe_float(df['Low'].tail(252).min()):.2f}"), unsafe_allow_html=True)
with col_sp:
    v_val = df["Volume"].tail(20).mean()
    st.markdown(metric_card("20-Day Mean Volume", f"{v_val/1e6:.2f}M" if v_val>=1e6 else f"{v_val/1e3:.0f}K"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📡 SIGNAL RECEPTOR", "🔮 QUANT FORECAST ENGINE", "📊 TELEMETRY MATRIX"])

# ════════════════════════════════════════════════════════════
#  TAB 1 — SIGNAL COMPILER
# ════════════════════════════════════════════════════════════
with tab1:
    rsi_now  = safe_float(df["RSI"].dropna().iloc[-1] if df["RSI"].dropna().any() else np.nan)
    macd_now = safe_float(df["MACD"].dropna().iloc[-1] if df["MACD"].dropna().any() else np.nan)
    sig_now  = safe_float(df["MACDSig"].dropna().iloc[-1] if df["MACDSig"].dropna().any() else np.nan)
    ma30_now = safe_float(df["MA30"].dropna().iloc[-1] if df["MA30"].dropna().any() else np.nan)
    ma90_now = safe_float(df["MA90"].dropna().iloc[-1] if df["MA90"].dropna().any() else np.nan)
    bb_up    = safe_float(df["BB_Upper"].dropna().iloc[-1] if df["BB_Upper"].dropna().any() else np.nan)
    bb_lo    = safe_float(df["BB_Lower"].dropna().iloc[-1] if df["BB_Lower"].dropna().any() else np.nan)

    eff_entry = entry_price if entry_price > 0.0 else current_price

    tech_scores = {
        "RSI Indicator (14)": 1 if rsi_now < 35 else (-1 if rsi_now > 65 else 0),
        "MACD Convergence Zone": 1 if macd_now > sig_now else -1,
        "MA-30/90 Structural Crossing": 1 if ma30_now > ma90_now else -1,
        "BB Volatility Bounds": 1 if current_price < bb_lo else (-1 if current_price > bb_up else 0)
    }

    if f_prices is not None:
        model_target = safe_float(f_prices[min(4, len(f_prices)-1)])
        model_pct = (model_target - eff_entry) / eff_entry * 100
        tech_scores["NN Neural Forecast Drift"] = 1 if model_pct > 1.2 else (-1 if model_pct < -1.2 else 0)

    valuation_premium = (eff_entry - current_price) / current_price * 100
    if valuation_premium > 5.0:
        tech_scores["Structural Entry Guard"] = -2 
    elif valuation_premium < -5.0:
        tech_scores["Structural Entry Guard"] = 1
    else:
        tech_scores["Structural Entry Guard"] = 0

    total_score = sum(tech_scores.values())
    if total_score >= 2:    signal_label, signal_css = "BUY",  "signal-buy"
    elif total_score <= -2: signal_label, signal_css = "SELL", "signal-sell"
    else:                   signal_label, signal_css = "HOLD", "signal-hold"

    stop_loss    = eff_entry * (1 - risk_pct / 100)
    take_profit  = eff_entry + max(eff_entry - stop_loss, 0.01) * 2.0

    left, right = st.columns([1, 1.8], gap="large")
    with left:
        st.markdown('<p class="section-header">Consolidated Alpha Vector</p>', unsafe_allow_html=True)
        st.markdown(f'<div style="text-align:center;padding:24px 0 16px"><div class="{signal_css}">{signal_label}</div></div>', unsafe_allow_html=True)
        
        st.markdown('<p class="section-header">Consensus Matrix Details</p>', unsafe_allow_html=True)
        for name, sc in tech_scores.items():
            if sc >= 1:
                st.markdown(f'<div class="indicator-row"><span class="ind-icon">🟢</span><span class="ind-name">{name}</span><span class="ind-badge badge-bull">ACCELERATIVE</span></div>', unsafe_allow_html=True)
            elif sc == -1:
                st.markdown(f'<div class="indicator-row"><span class="ind-icon">🔴</span><span class="ind-name">{name}</span><span class="ind-badge badge-bear">DEPRESSIVE</span></div>', unsafe_allow_html=True)
            elif sc <= -2:
                st.markdown(f'<div class="indicator-row"><span class="ind-icon">⚠️</span><span class="ind-name">{name}</span><span class="ind-badge badge-warn">OUTLIER LOCKOUT</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="indicator-row"><span class="ind-icon">🟡</span><span class="ind-name">{name}</span><span class="ind-badge badge-neut">STABILIZED</span></div>', unsafe_allow_html=True)

    with right:
        st.markdown('<p class="section-header">Dynamic Execution Space Map</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.markdown(metric_card("Calibrated Entry", f"${eff_entry:.2f}"), unsafe_allow_html=True)
        c2.markdown(metric_card("Stop Threshold", f"${stop_loss:.2f}", f"−{risk_pct}%", "metric-delta-down"), unsafe_allow_html=True)
        c3.markdown(metric_card("Reward Target (1:2)", f"${take_profit:.2f}", f"+{risk_pct*2}%", "metric-delta-up"), unsafe_allow_html=True)
        
        fig_t = go.Figure()
        ctx_df = df.tail(90)
        fig_t.add_trace(go.Scatter(x=ctx_df.index, y=ctx_df["Close"], name="Close Core", line=dict(color=ACCENT, width=1.8)))
        fig_t.add_trace(go.Scatter(x=ctx_df.index, y=ctx_df["MA30"], name="MA30 Node", line=dict(color=BLUE, width=1, dash="dash")))
        fig_t.add_trace(go.Scatter(x=ctx_df.index, y=ctx_df["BB_Upper"], name="BB Upper Band", line=dict(color=MUTED, width=0.8, dash="dot")))
        fig_t.add_trace(go.Scatter(x=ctx_df.index, y=ctx_df["BB_Lower"], name="BB Lower Band", line=dict(color=MUTED, width=0.8, dash="dot"), fill="tonexty", fillcolor="rgba(100,116,139,0.02)"))
        
        fig_t.add_hline(y=eff_entry, line_color=ACCENT, line_width=1.2, line_dash="solid", annotation_text="Calculated Entry Target", annotation_position="top left", annotation_font=dict(color=ACCENT, size=9, family="Space Grotesk"))
        fig_t.add_hline(y=take_profit, line_color=GREEN, line_width=1.2, line_dash="dash", annotation_text="Take Profit Threshold (1:2)", annotation_position="top left", annotation_font=dict(color=GREEN, size=9, family="Space Grotesk"))
        fig_t.add_hline(y=stop_loss, line_color=RED, line_width=1.2, line_dash="dash", annotation_text="Risk Stop Loss Line", annotation_position="bottom left", annotation_font=dict(color=RED, size=9, family="Space Grotesk"))

        fig_t.update_layout(**base_layout(280, "Price Action Vector vs Boundary Limits", override_yaxis=dict(tickprefix="$")))
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig_t, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  TAB 2 — FORECAST STRATEGY
# ════════════════════════════════════════════════════════════
with tab2:
    if model is None:
        empty_state("🔮 NETWORK OFFLINE", "Model telemetry connection pending. Initialize the neural transceiver from the control console.")
    elif f_prices is None:
        empty_state("⛔ INTERFERENCE FAULT", "Timeline core compile mismatch.")
    else:
        st.markdown(f'<div style="background: rgba(59,130,246,0.06); border: 1px solid rgba(59,130,246,0.22); border-radius: 8px; padding: 12px 18px; font-family: \'JetBrains Mono\', monospace; font-size: 0.78rem; color: #a1b0cb; margin-bottom: 16px;">🌐 <strong>TEMPORAL SYNCHRONIZATION POINT:</strong> {pd.Timestamp(chosen_start_date).strftime("%A, %d %b %Y")} (UTC)</div>', unsafe_allow_html=True)
        
        f_end = safe_float(f_prices[-1])
        f_chg = ((f_end - current_price) / current_price * 100)

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("Simulation Anchor Price", f"${current_price:.2f}"), unsafe_allow_html=True)
        c2.markdown(metric_card("Terminal Horizon Price", f"${f_end:.2f}", f"{f_chg:+.2f}%", "metric-delta-up" if f_chg>=0 else "metric-delta-down"), unsafe_allow_html=True)
        c3.markdown(metric_card("Timeline Peak Bound", f"${safe_float(f_prices.max()):.2f}"), unsafe_allow_html=True)
        c4.markdown(metric_card("Timeline Trough Bound", f"${safe_float(f_prices.min()):.2f}"), unsafe_allow_html=True)

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], name="Historical Real-Time Core", line=dict(color=ACCENT, width=1.8)))
        
        if b_dates is not None and len(b_dates) > 0:
            b_x = list(b_dates) + list(b_dates[::-1])
            b_y = list(b_upper) + list(b_lower[::-1])
            fig_fc.add_trace(go.Scatter(x=b_x, y=b_y, fill="toself", fillcolor="rgba(168,85,247,0.05)", line=dict(color="rgba(0,0,0,0)"), name="Bridge Uncertainty Band"))
            fig_fc.add_trace(go.Scatter(x=b_dates, y=b_prices, name="Context Bridge Trajectory", line=dict(color=PURPLE, width=1.5, dash="dash")))

        fx = list(f_dates) + list(f_dates[::-1])
        fy = list(f_upper) + list(f_lower[::-1])
        fig_fc.add_trace(go.Scatter(x=fx, y=fy, fill="toself", fillcolor="rgba(59,130,246,0.10)", line=dict(color="rgba(0,0,0,0)"), name="Forecast Uncertainty Band"))
        fig_fc.add_trace(go.Scatter(x=f_dates, y=f_prices, name="Model Hybrid Projection", line=dict(color=BLUE, width=2.0, dash="dashdot"), mode="lines+markers"))

        view_start = pd.Timestamp(chosen_start_date) - pd.DateOffset(months=2)
        view_end   = pd.Timestamp(f_dates[-1]) + pd.DateOffset(months=2)

        hist_in_view = df.loc[(df.index >= view_start) & (df.index <= view_end), "Adj Close"].dropna()
        all_visible_prices = list(hist_in_view.values) + list(f_prices)
        if b_prices is not None and len(b_prices) > 0:
            all_visible_prices += list(b_prices)
        all_visible_prices += list(f_lower) + list(f_upper)
        if b_lower is not None and len(b_lower) > 0:
            all_visible_prices += list(b_lower) + list(b_upper)

        all_visible_prices = [v for v in all_visible_prices if np.isfinite(v) and v > 0]
        if all_visible_prices:
            y_min   = min(all_visible_prices)
            y_max   = max(all_visible_prices)
            y_pad   = (y_max - y_min) * 0.08
            y_range = [y_min - y_pad, y_max + y_pad]
        else:
            y_range = None

        base_ly_params = base_layout(
            440,
            "Neural Engine Simulation Space Continuum",
            override_yaxis=dict(tickprefix="$", range=y_range) if y_range else dict(tickprefix="$"),
        )
        base_ly_params["xaxis"].update(dict(range=[view_start, view_end]))

        fig_fc.update_layout(**base_ly_params)
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig_fc, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  TAB 3 — TELEMETRY CONTROL MATRIX
# ════════════════════════════════════════════════════════════
with tab3:
    df_vis = df.copy()
    
    if b_dates is not None and len(b_dates) > 0:
        df_br = pd.DataFrame(index=b_dates)
        for c in ["Close", "Open", "High", "Low"]: df_br[c] = b_prices
        df_br["Volume"] = df["Volume"].iloc[-1]
        df_vis = pd.concat([df_vis, df_br])
        
    if f_prices is not None:
        df_f = pd.DataFrame(index=f_dates)
        df_f["Close"] = f_prices; df_f["Open"] = f_prices
        df_f["High"] = f_upper; df_f["Low"] = f_lower
        df_f["Volume"] = df["Volume"].iloc[-1]
        df_vis = pd.concat([df_vis, df_f])
        
    df_vis = df_vis[~df_vis.index.duplicated(keep='first')].sort_index()
    
    df_vis["Spread"] = df_vis["High"] - df_vis["Low"]
    df_vis["MA30"]   = df_vis["Close"].rolling(30).mean()
    df_vis["MA90"]   = df_vis["Close"].rolling(90).mean()
    df_vis["BB_Mid"] = df_vis["Close"].rolling(20).mean()
    v_std            = df_vis["Close"].rolling(20).std()
    df_vis["BB_Upper"] = df_vis["BB_Mid"] + 2 * v_std
    df_vis["BB_Lower"] = df_vis["BB_Mid"] - 2 * v_std
    
    df_vis["EMA12"] = df_vis["Close"].ewm(span=12, adjust=False).mean()
    df_vis["EMA26"] = df_vis["Close"].ewm(span=26, adjust=False).mean()
    df_vis["MACD"]  = df_vis["EMA12"] - df_vis["EMA26"]
    df_vis["MACDSig"] = df_vis["MACD"].ewm(span=9, adjust=False).mean()
    df_vis["MACDHist"] = df_vis["MACD"] - df_vis["MACDSig"]
    
    d_v = df_vis["Close"].diff()
    g_v = d_v.clip(lower=0).rolling(14).mean()
    l_v = (-d_v.clip(upper=0)).rolling(14).mean()
    df_vis["RSI"] = 100 - (100 / (1 + (g_v / l_v.replace(0, np.nan))))

    fa, fb = st.columns(2)
    with fa: viz_start = st.date_input("Vector Frame Start", value=df.index[-90].date(), min_value=df.index[0].date(), max_value=df_vis.index[-1].date(), key="v_start")
    with fb: viz_end   = st.date_input("Vector Frame Stop", value=df_vis.index[-1].date(), min_value=df.index[0].date(), max_value=df_vis.index[-1].date(), key="v_end")

    if viz_start >= viz_end:
        st.warning("⚠️ Frame Index Exception. Correct inputs.")
        st.stop()

    dv = df_vis.loc[str(viz_start):str(viz_end)].copy()

    r1a, r1b = st.columns([3, 2])
    with r1a:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=dv.index, y=dv["Close"], name="Unified Close Connection", line=dict(color=ACCENT, width=1.5)))
        for c, col_color in [("MA30", BLUE), ("MA90", PURPLE)]:
            if c in dv.columns and dv[c].notna().any():
                fig1.add_trace(go.Scatter(x=dv.index, y=dv[c], name=c, line=dict(color=col_color, width=1, dash="dot")))
        fig1.update_layout(**base_layout(300, "Continuous Close Pricing Sequence"))
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with r1b:
        v_cols = [GREEN if i==0 else (GREEN if dv["Close"].iloc[i]>=dv["Close"].iloc[i-1] else RED) for i in range(len(dv))]
        fig2 = go.Figure(go.Bar(x=dv.index, y=dv["Volume"], marker_color=v_cols, name="Volume Stream"))
        fig2.update_layout(**base_layout(300, "Segmented Bar Volume Distribution"))
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    r2a, r2b = st.columns([3, 2])
    with r2a:
        fig3 = go.Figure(go.Candlestick(x=dv.index, open=dv["Open"], high=dv["High"], low=dv["Low"], close=dv["Close"], increasing_line_color=GREEN, decreasing_line_color=RED, name="OHLC Candlestick"))
        if dv["BB_Upper"].notna().any():
            fig3.add_trace(go.Scatter(x=dv.index, y=dv["BB_Upper"], name="Volatility Cell Upper", line=dict(color=MUTED, width=0.8, dash="dash")))
            fig3.add_trace(go.Scatter(x=dv.index, y=dv["BB_Lower"], fill="tonexty", fillcolor="rgba(100,116,139,0.02)", name="Volatility Cell Lower", line=dict(color=MUTED, width=0.8, dash="dash")))
        fig3.update_layout(**base_layout(300, "Holographic Structural Candlestick Envelope"))
        fig3.update_layout(xaxis_rangeslider_visible=False)
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with r2b:
        fig4 = go.Figure(go.Scatter(x=dv.index, y=dv["Spread"], fill="tozeroy", fillcolor="rgba(255,204,0,0.06)", line=dict(color=ACCENT, width=1.0)))
        fig4.update_layout(**base_layout(300, "Intraday Dispersion Bounds (High - Low Variance)"))
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    r3a, r3b = st.columns(2)
    with r3a:
        if dv["MACD"].notna().any():
            fig5 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.05)
            fig5.add_trace(go.Scatter(x=dv.index, y=dv["MACD"], name="MACD Vector", line=dict(color=BLUE, width=1.2)), row=1, col=1)
            fig5.add_trace(go.Scatter(x=dv.index, y=dv["MACDSig"], name="MACD Signal", line=dict(color=ACCENT, width=1.2)), row=1, col=1)
            h_colors = [GREEN if val >= 0 else RED for val in dv["MACDHist"].fillna(0)]
            fig5.add_trace(go.Bar(x=dv.index, y=dv["MACDHist"], name="Histogram Matrix", marker_color=h_colors), row=2, col=1)
            fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(12,16,28,0.40)", font_color=FONT_COL, height=300, margin=dict(l=40,r=20,t=30,b=20), title=dict(text="Momentum Convergence/Divergence Oscillator (12, 26, 9)", font=dict(size=11, color=MUTED, family="Space Grotesk")), showlegend=False)
            fig5.update_xaxes(gridcolor=GRID_COL); fig5.update_yaxes(gridcolor=GRID_COL)
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            st.plotly_chart(fig5, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else: empty_state("📉", "Initializing oscillator datasets...")
    with r3b:
        if dv["RSI"].notna().any():
            fig6 = go.Figure()
            fig6.add_trace(go.Scatter(x=dv.index, y=dv["RSI"], name="RSI Engine", line=dict(color=PURPLE, width=1.2)))
            fig6.add_hrect(y0=70, y1=100, fillcolor="rgba(244,63,94,0.03)", line_width=0)
            fig6.add_hrect(y0=0,  y1=30,  fillcolor="rgba(16,185,129,0.03)", line_width=0)
            fig6.add_hline(y=70, line_color=RED, line_dash="dash", line_width=0.8)
            fig6.add_hline(y=30, line_color=GREEN, line_dash="dash", line_width=0.8)
            fig6.update_layout(**base_layout(300, "Relative Strength Velocity Zone RSI (14)", override_yaxis=dict(range=[0, 100])))
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            st.plotly_chart(fig6, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else: empty_state("📉", "Initializing velocity datasets...")

    r4a, r4b = st.columns(2)
    with r4a:
        yearly = dv.groupby(dv.index.year)["Close"].mean().reset_index()
        fig7 = go.Figure(go.Bar(x=yearly.iloc[:, 0].astype(str), y=yearly["Close"], marker_color=ACCENT))
        fig7.update_layout(**base_layout(300, "Macro Annualized Core Price Assets"))
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig7, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with r4b:
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        fig8 = go.Figure()
        for m_idx, m_name in enumerate(months, 1):
            sub = dv[dv.index.month == m_idx]["Close"].dropna()
            if not sub.empty: fig8.add_trace(go.Box(y=sub, name=m_name, marker_color=BLUE, line_color=BLUE, fillcolor="rgba(59,130,246,0.12)"))
        fig8.update_layout(**base_layout(300, "Seasonality Structural Distribution Matrices"), showlegend=False)
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig8, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<p class="section-header">Cross-Sectional Attribute Correlation Matrix Heatmap</p>', unsafe_allow_html=True)
    corr_cols = [c for c in ["Open", "High", "Low", "Close", "Volume", "Spread"] if c in dv.columns]
    corr_data = dv[corr_cols].dropna()
    
    if len(corr_data) >= 5 and len(corr_cols) >= 2:
        c_mat = corr_data.corr().round(3)
        fig11 = go.Figure(go.Heatmap(z=c_mat.values, x=corr_cols, y=corr_cols, colorscale="RdBu", zmid=0, zmin=-1, zmax=1, text=c_mat.values, texttemplate="%{text:.2f}", showscale=True))
        fig11.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(12,16,28,0.40)", font_color=FONT_COL, height=360, margin=dict(l=40, r=20, t=10, b=40))
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig11, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        empty_state("📊", "Attribute matrices lack sufficient spatial alignment dimensions.")
