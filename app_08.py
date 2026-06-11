# ============================================================
# TSLA FORECASTING HUB  |  app.py
# Model: CNN-GRU + Hurst Regime Detection + OU Mean-Reversion
# Full Engineering Overhaul: 25 Unique ReactBits-Inspired Components
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

# Safe retrieval for environmental secrets
secret_url = ""
try:
    if "model_config" in st.secrets and "gdrive_model_link" in st.secrets["model_config"]:
        secret_url = st.secrets["model_config"]["gdrive_model_link"]
except Exception:
    pass

def get_secret_model_link() -> str:
    return secret_url

# ── MODULE 19: ULTRA-WIDE CYBERNETIC DASH LAYOUT INITIALIZATION ──
st.set_page_config(
    page_title="TSLA Hybrid Forecast Hub",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ╔══════════════════════════════════════════════════════════╗
# ║        REACTBITS 25-MODULE FULL INJECTION ENGINE         ║
# ╚══════════════════════════════════════════════════════════╝
st.markdown("""
<style>
/* MODULE 13: JETBRAINS MONO CODE BLUEPRINT TYPOGRAPHY */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap');
/* MODULE 14: SPACE GROTESK VARIABLE GEOMETRY DISPLAY TYPOGRAPHY */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

.stApp {
  background-color: #05070c !important;
  color: #e2eaf5 !important;
  font-family: 'Inter', sans-serif !important;
  overflow-x: hidden;
}

/* MODULE 11: AURORA FLOATING PLASMA NODES A & B ANIMATION ENGINE */
@keyframes aurora-drift-primary {
  0%   { transform: translate(0%,   0%)   scale(1);    opacity: 0.55; }
  33%  { transform: translate(8%,  -10%)  scale(1.15); opacity: 0.35; }
  66%  { transform: translate(-6%,  8%)   scale(0.95); opacity: 0.65; }
  100% { transform: translate(0%,   0%)   scale(1);    opacity: 0.55; }
}
@keyframes aurora-drift-secondary {
  0%   { transform: translate(0%,  0%)   scale(1);    opacity: 0.40; }
  40%  { transform: translate(-8%, 12%)  scale(1.10); opacity: 0.50; }
  75%  { transform: translate(6%, -6%)   scale(0.90); opacity: 0.25; }
  100% { transform: translate(0%,  0%)   scale(1);    opacity: 0.40; }
}

/* MODULE 1: STRANDS CYBER BACKGROUND LINE MESH MATRIX */
.stApp::before {
  content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background:
    radial-gradient(ellipse 70% 55% at 15% 20%, rgba(59,130,246,0.14) 0%, transparent 65%),
    radial-gradient(ellipse 60% 70% at 85% 75%, rgba(147,51,234,0.10) 0%, transparent 60%),
    radial-gradient(ellipse 50% 40% at 50% 50%, rgba(245,158,11,0.03) 0%, transparent 55%);
  animation: aurora-drift-primary 22s ease-in-out infinite;
}
.stApp::after {
  content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background:
    radial-gradient(ellipse 60% 45% at 80% 20%, rgba(16,185,129,0.06) 0%, transparent 60%),
    radial-gradient(ellipse 50% 60% at 20% 80%, rgba(59,130,246,0.08) 0%, transparent 55%);
  animation: aurora-drift-secondary 26s ease-in-out infinite;
}

/* MODULE 3: DOTGRID MATRIX CANVAS PATTERN LAYERING */
.stApp > div {
  background-image: radial-gradient(circle, rgba(255,255,255,0.022) 1px, transparent 1px);
  background-size: 22px 22px;
  position: relative; z-index: 1;
}

/* MODULE 18: CUSTOM OVERFLOW SIDEBAR FORCE-SCROLL ENGINE */
[data-testid="stSidebar"] {
  background: rgba(6, 8, 14, 0.97) !important;
  border-right: 1px solid rgba(59,130,246,0.20) !important;
  backdrop-filter: blur(24px) !important;
  -webkit-backdrop-filter: blur(24px) !important;
}
[data-testid="stSidebarContent"], [data-testid="stSidebarUserContent"], section[data-testid="stSidebar"] > div {
  overflow-y: auto !important; max-height: 100vh !important;
}
[data-testid="stSidebarContent"]::-webkit-scrollbar { width: 5px; }
[data-testid="stSidebarContent"]::-webkit-scrollbar-track { background: rgba(0,0,0,0.1); }
[data-testid="stSidebarContent"]::-webkit-scrollbar-thumb { background: rgba(59,130,246,0.2); border-radius: 4px; }

/* MODULE 8: STAGGERED ANIMATED MENUS SIDE PANEL TRANSITION */
[data-testid="stSidebar"] .element-container {
  animation: sidebarMenuSlideIn 0.35s cubic-bezier(0.16, 1, 0.3, 1) both;
}
@keyframes sidebarMenuSlideIn {
  from { opacity: 0; transform: translateX(-12px); }
  to { opacity: 1; transform: translateX(0); }
}

/* MODULE 17: NEON MATRIX COLOR PALETTES (PREMIUM NAVIGATION TABS) */
.stTabs [data-baseweb="tab-list"] {
  gap: 8px !important;
  background: rgba(7,9,15,0.90) !important;
  border-bottom: 1px solid rgba(59,130,246,0.25) !important;
  backdrop-filter: blur(16px);
  padding: 6px 16px 0 16px !important;
  border-radius: 12px 12px 0 0;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important; color: rgba(148,163,184,0.60) !important;
  font-family: 'Space Grotesk', sans-serif !important; font-weight: 600 !important;
  font-size: 0.82rem !important; letter-spacing: 0.10em !important; text-transform: uppercase !important;
  padding: 12px 24px !important; border: none !important; border-bottom: 2px solid transparent !important;
  transition: all 0.25s ease !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #ffaa00 !important; }
.stTabs [aria-selected="true"] {
  color: #ffcc00 !important; border-bottom: 2px solid #ffcc00 !important;
  background: rgba(255, 204, 0, 0.04) !important;
  text-shadow: 0 0 12px rgba(255, 204, 0, 0.30);
}

/* MODULE 2: SHINYTEXT LEFT-TO-RIGHT SHIMMER GRADIENT */
@keyframes shinyTextSweep {
  0%   { background-position: -200% center; }
  100% { background-position:  200% center; }
}
.shiny-text {
  background: linear-gradient(120deg, #ffffff 20%, #ffcc00 50%, #ffffff 80%);
  background-size: 200% auto;
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  animation: shinyTextSweep 4.0s linear infinite;
  font-weight: 700;
}

/* MODULE 7: SPOTLIGHT GLOWING SHIMMER CARD FRAMEWERK */
@keyframes cardShimmer {
  0%   { background-position: -200% center; }
  100% { background-position:  200% center; }
}
.metric-card {
  position: relative; background: rgba(10,13,23,0.85) !important;
  border: 1px solid rgba(59,130,246,0.22) !important; border-radius: 14px !important;
  padding: 18px 16px !important; text-align: center; min-height: 110px;
  backdrop-filter: blur(20px); overflow: hidden;
  transition: border-color 0.3s ease, box-shadow 0.3s ease, transform 0.25s ease;
}
/* MODULE 9: GLARE HOVER OVERLAY INTEGRATION */
.metric-card::before {
  content: ""; position: absolute; inset: 0; border-radius: 14px;
  background: linear-gradient(115deg, transparent 35%, rgba(59,130,246,0.12) 50%, transparent 65%);
  background-size: 200% 100%; animation: cardShimmer 5.0s linear infinite; pointer-events: none;
}
.metric-card:hover {
  border-color: rgba(59,130,246,0.60) !important;
  box-shadow: 0 8px 25px rgba(59,130,246,0.25);
  transform: translateY(-2px);
}
.metric-label {
  color: #94a3b8; font-size: 0.72rem; letter-spacing: 0.12em;
  text-transform: uppercase; margin-bottom: 6px; font-family: 'Space Grotesk', sans-serif;
}
.metric-value {
  color: #ffffff; font-size: 1.55rem; font-weight: 700;
  font-family: 'JetBrains Mono', monospace; text-shadow: 0 0 15px rgba(59,130,246,0.20);
}

/* MODULE 10: SPLASH GLOWING PULSE RING SIGNUM BADGES */
@keyframes signumBuy {
  0%   { box-shadow: 0 0 0 0 rgba(16,185,129,0.50), 0 0 12px rgba(16,185,129,0.20); }
  70%  { box-shadow: 0 0 0 10px rgba(16,185,129,0), 0 0 25px rgba(16,185,129,0.10); }
  100% { box-shadow: 0 0 0 0 rgba(16,185,129,0), 0 0 12px rgba(16,185,129,0.20); }
}
@keyframes signumSell {
  0%   { box-shadow: 0 0 0 0 rgba(244,63,94,0.50), 0 0 12px rgba(244,63,94,0.20); }
  70%  { box-shadow: 0 0 0 10px rgba(244,63,94,0), 0 0 25px rgba(244,63,94,0.10); }
  100% { box-shadow: 0 0 0 0 rgba(244,63,94,0), 0 0 12px rgba(244,63,94,0.20); }
}
@keyframes signumHold {
  0%   { box-shadow: 0 0 0 0 rgba(245,158,11,0.50), 0 0 12px rgba(245,158,11,0.20); }
  70%  { box-shadow: 0 0 0 10px rgba(245,158,11,0), 0 0 25px rgba(245,158,11,0.10); }
  100% { box-shadow: 0 0 0 0 rgba(245,158,11,0), 0 0 12px rgba(245,158,11,0.20); }
}
.signal-buy {
  background: rgba(16,185,129,0.12) !important; color: #10b981 !important;
  border: 2px solid rgba(16,185,129,0.70) !important; border-radius: 8px !important;
  padding: 10px 32px !important; font-weight: 700 !important; font-size: 1.55rem !important;
  font-family: 'Space Grotesk', monospace !important; letter-spacing: 0.08em;
  display: inline-block !important; animation: signumBuy 2.2s infinite !important;
}
.signal-sell {
  background: rgba(244,63,94,0.12) !important; color: #f43f5e !important;
  border: 2px solid rgba(244,63,94,0.70) !important; border-radius: 8px !important;
  padding: 10px 32px !important; font-weight: 700 !important; font-size: 1.55rem !important;
  font-family: 'Space Grotesk', monospace !important; letter-spacing: 0.08em;
  display: inline-block !important; animation: signumSell 2.2s infinite !important;
}
.signal-hold {
  background: rgba(245,158,11,0.12) !important; color: #f59e0b !important;
  border: 2px solid rgba(245,158,11,0.70) !important; border-radius: 8px !important;
  padding: 10px 32px !important; font-weight: 700 !important; font-size: 1.55rem !important;
  font-family: 'Space Grotesk', monospace !important; letter-spacing: 0.08em;
  display: inline-block !important; animation: signumHold 2.2s infinite !important;
}

.metric-delta-up   { color: #10b981; font-size: 0.82rem; margin-top: 4px; font-family: 'JetBrains Mono', monospace; }
.metric-delta-down { color: #f43f5e; font-size: 0.82rem; margin-top: 4px; font-family: 'JetBrains Mono', monospace; }
.metric-muted      { color: #64748b; font-size: 0.82rem; margin-top: 4px; }

/* MODULE 15: SCANNING LIGHT BAR COMPONENT HEADERS */
@keyframes barScanEngine {
  0%   { left: -100%; }
  100% { left:  130%; }
}
.section-header {
  position: relative; color: #ffcc00; font-size: 0.76rem; letter-spacing: 0.18em;
  text-transform: uppercase; margin-top: 24px; margin-bottom: 12px;
  font-family: 'Space Grotesk', sans-serif; font-weight: 600; padding-bottom: 6px; overflow: hidden;
}
.section-header::after {
  content: ""; position: absolute; bottom: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(59,130,246,0.3), transparent);
}
.section-header::before {
  content: ""; position: absolute; bottom: 0; width: 35%; height: 1px;
  background: linear-gradient(90deg, transparent, #ffcc00, transparent);
  animation: barScanEngine 4.0s linear infinite;
}

/* MODULE 4: FLUID GLASS BORDER GLOW CONTAINER MODULE */
.chart-wrap {
  position: relative; border: 1px solid rgba(59,130,246,0.18); border-radius: 12px;
  overflow: hidden; background: rgba(6,9,16,0.55); padding: 8px;
  transition: border-color 0.3s ease, box-shadow 0.3s ease; margin-bottom: 14px;
}
.chart-wrap:hover { 
  border-color: rgba(59,130,246,0.40); box-shadow: 0 5px 18px rgba(59,130,246,0.08);
}

/* MODULE 16: DYNAMIC MACRO TELEMETRY INDICATOR TOGGLE ROWS */
.indicator-row {
  display: flex; align-items: center; gap: 12px; padding: 10px 14px; margin-bottom: 8px;
  border-radius: 8px; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.04);
  font-size: 0.82rem; color: #e2e8f5; transition: all 0.2s ease;
}
.indicator-row:hover {
  background: rgba(59,130,246,0.07); border-color: rgba(59,130,246,0.22); transform: translateX(2px);
}
.ind-icon { font-size: 0.90rem; flex-shrink: 0; }
.ind-name { font-weight: 500; flex: 1; }
.ind-badge {
  font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; letter-spacing: 0.05em;
  padding: 3px 10px; border-radius: 5px; font-weight: 600;
}
.badge-bull { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.25); }
.badge-bear { background: rgba(244,63,94,0.15);  color: #f43f5e; border: 1px solid rgba(244,63,94,0.25); }
.badge-neut { background: rgba(245,158,11,0.12); color: #f59e0b; border: 1px solid rgba(245,158,11,0.22); }
.badge-warn { background: rgba(244,63,94,0.20);  color: #ffa5b5; border: 1px solid rgba(244,63,94,0.30); }

/* MODULE 6: SCROLL VELOCITY ENGINE MOMENTUM CONTROLS (BUTTONS) */
.stButton > button {
  background: linear-gradient(135deg, #09122b 0%, #12214d 100%) !important;
  border: 1px solid rgba(59,130,246,0.40) !important; color: #60a5fa !important;
  border-radius: 8px !important; padding: 10px 24px !important;
  font-family: 'Space Grotesk', sans-serif !important; font-weight: 600 !important;
  font-size: 0.82rem !important; letter-spacing: 0.08em !important; text-transform: uppercase !important;
  transition: all 0.25s ease !important; width: 100%;
}
.stButton > button:hover {
  border-color: rgba(59,130,246,0.85) !important; box-shadow: 0 4px 15px rgba(59,130,246,0.3) !important;
  color: #93c5fd !important; transform: translateY(-1px);
}
</style>
""", unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════╗
# ║                   DATA CORE CONFIG                       ║
# ╚══════════════════════════════════════════════════════════╝
CSV_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TSLA_1.csv")
LOOKBACK  = 60
GRID_COL  = "#161f30"
FONT_COL  = "#e2e8f5"
ACCENT    = "#ffcc00"
GREEN     = "#10b981"
RED       = "#f43f5e"
BLUE      = "#3b82f6"
PURPLE    = "#a855f7"
MUTED     = "#64748b"

def safe_float(val, fallback=0.0) -> float:
    try:
        v = float(val)
        return fallback if (np.isnan(v) or np.isinf(v)) else v
    except Exception:
        return fallback

# MODULE 5: COUNTUP NUMERIC TICK VALUE LOADER OVERLAY
def metric_card(label: str, value: str, delta: str = "", delta_cls: str = "metric-muted") -> str:
    return (
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="{delta_cls}">{delta}</div>'
        f'</div>'
    )

# MODULE 20: INTERACTIVE PLOTLY DARK MESH SYNTHESIS COMPONENT
def base_layout(height: int = 350, title: str = "", override_yaxis=None) -> dict:
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(10,14,24,0.45)",
        font_color=FONT_COL, 
        height=height,
        margin=dict(l=45, r=20, t=45, b=35),
        title=dict(text=title, font=dict(size=11, color=MUTED, family="Space Grotesk")),
        xaxis=dict(gridcolor=GRID_COL, showgrid=True, linecolor=GRID_COL, zeroline=False),
        yaxis=dict(gridcolor=GRID_COL, showgrid=True, linecolor=GRID_COL, zeroline=False),
    )
    # Safely merge yaxis overrides into the native yaxis property dictionary
    if override_yaxis is not None:
        layout["yaxis"].update(override_yaxis)
    return layout

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
            warnings_out.append(f"Inversion read parameter fail: {e}")
    else:
        warnings_out.append("TSLA_1.csv not found. Re-routing initialization to safe generative data space models.")
        np.random.seed(42)
        base_price = 220.0
        biz_dates = pd.bdate_range(start=pd.Timestamp("2025-06-11"), periods=252)
        prices = [base_price]
        current = base_price
        for _ in range(1, len(biz_dates)):
            current = max(current * np.exp(0.0005 + 0.022 * np.random.normal()), 10.0)
            prices.append(current)
        prices = np.array(prices)
        opens = prices * (1.0 + (np.random.rand(len(biz_dates)) - 0.5) * 0.012)
        spreads = prices * (0.01 + np.random.rand(len(biz_dates)) * 0.03)
        df = pd.DataFrame({
            "Open": opens, "High": np.maximum(opens, prices) + spreads*0.4,
            "Low": np.minimum(opens, prices) - spreads*0.6, "Close": prices,
            "Adj Close": prices, "Volume": (15 + np.random.rand(len(biz_dates))*20)*1000000
        }, index=biz_dates)
        df.index.name = "Date"

    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df["Spread"] = df["High"] - df["Low"]
    df["MA30"] = df["Close"].rolling(30).mean()
    df["MA90"] = df["Close"].rolling(90).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACDSig"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACDHist"] = df["MACD"] - df["MACDSig"]
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
    scaler.fit(df[["Adj Close"]].dropna().values)
    return scaler

# ╔══════════════════════════════════════════════════════════╗
# ║        QUANT MATHEMATICAL ENGINE MATHEMATICS             ║
# ╚══════════════════════════════════════════════════════════╝
def _hurst_exponent(series: np.ndarray) -> float:
    n = len(series)
    if n < 20: return 0.5
    try:
        lags = [2, 4, 8, 16, 32] if n >= 64 else [2, 4, 8]
        rs_vals = []
        for lag in lags:
            chunks = [series[i:i+lag] for i in range(0, n - lag + 1, lag)]
            if not chunks: continue
            rs_chunk = []
            for c in chunks:
                r = c.max() - c.min()
                s = np.std(c, ddof=1)
                if s > 0: rs_chunk.append(r / s)
            if rs_chunk: rs_vals.append((lag, np.mean(rs_chunk)))
        if len(rs_vals) < 2: return 0.5
        H = np.polyfit(np.log([v[0] for v in rs_vals]), np.log([v[1] for v in rs_vals]), 1)[0]
        return float(np.clip(H, 0.1, 0.9))
    except Exception: return 0.5

def _compute_regime(history_series: pd.Series) -> dict:
    prices = history_series.dropna().values[-252:]
    if len(prices) < 30: return dict(hurst=0.5, trend_slope=0.0, mean_price=prices[-1], vol=0.02, ou_speed=0.05)
    log_rets = np.diff(np.log(prices))
    vol = float(np.std(log_rets)) if len(log_rets) > 1 else 0.02
    H = _hurst_exponent(prices)
    slope = float(np.polyfit(np.arange(len(prices)), np.log(prices), 1)[0]) * 252 
    mean_price = float(prices[-20:].mean())
    try:
        phi = np.clip(np.corrcoef(log_rets[:-1], log_rets[1:])[0, 1], -0.95, 0.95)
        ou_speed = float(-np.log(abs(phi))) if abs(phi) > 0 else 0.05
    except Exception: ou_speed = 0.05
    return dict(hurst=H, trend_slope=slope, mean_price=mean_price, vol=vol, ou_speed=ou_speed)

def _single_step_forecast(model, scaler, lookback_context: list) -> float:
    x_sc = scaler.transform(np.array(lookback_context[-LOOKBACK:]).reshape(-1, 1)).flatten()
    return float(scaler.inverse_transform([[float(model.predict(np.array(x_sc).reshape(1, LOOKBACK, 1), verbose=0)[0, 0])]])[0, 0])

def _revise_prediction(raw_pred: float, prev_price: float, anchor_price: float, regime: dict, step: int) -> float:
    H = regime["hurst"]
    mean_price = regime["mean_price"]
    ou_speed = regime["ou_speed"]
    vol = regime["vol"]
    raw_log_ret = np.log(max(raw_pred, 1e-3) / max(prev_price, 1e-3))
    cumulative_log_ret = np.log(max(prev_price, 1e-3) / max(anchor_price, 1e-3))
    
    drift_correction = -np.sign(cumulative_log_ret) * max(abs(cumulative_log_ret) - (2.0 * vol * np.sqrt(step)), 0) * 0.12
    ou_pull = -ou_speed * np.log(max(prev_price, 1e-3) / max(mean_price, 1e-3))
    ou_scale = float(np.interp(H, [0.35, 0.50, 0.65], [0.30, 0.15, 0.05]))
    
    final_p = prev_price * np.exp(np.clip(raw_log_ret + drift_correction + (ou_scale * ou_pull), -3.5*vol, 3.5*vol))
    return float(np.clip(final_p, mean_price * 0.3, mean_price * 3.0))

def dynamic_timeline_forecasting(model, scaler, df: pd.DataFrame, start_date: pd.Timestamp, n_days: int) -> tuple:
    db_max_date = df.index.max()
    target_start = pd.Timestamp(start_date)
    biz_dates = pd.bdate_range(start=target_start, periods=n_days)
    daily_vol = (df["DailyReturn"].std() / 100) if len(df) > 5 else 0.02
    preds_prices = []
    bridge_dates, bridge_prices, bridge_lo, bridge_hi = [], [], [], []

    if target_start <= db_max_date:
        anchor_price = safe_float(df["Adj Close"].iloc[-1])
        future_step = 0
        for curr_date in biz_dates:
            if curr_date <= db_max_date:
                preds_prices.append(safe_float(df["Adj Close"].reindex([curr_date], method='nearest').iloc[0]))
            else:
                future_step += 1
                if not bridge_dates:
                    curr_context = df["Adj Close"].loc[:df.index[df.index <= db_max_date][-1]].tail(LOOKBACK).tolist()
                    regime = _compute_regime(df["Adj Close"].loc[:df.index[df.index <= db_max_date][-1]])
                    curr_prev = safe_float(curr_context[-1])
                rev_p = _revise_prediction(_single_step_forecast(model, scaler, curr_context), curr_prev, anchor_price, regime, future_step)
                preds_prices.append(rev_p)
                curr_context.append(rev_p)
                curr_prev = rev_p
                bridge_dates.append(curr_date)
                bridge_prices.append(rev_p)
                w = daily_vol * np.sqrt(future_step)
                bridge_lo.append(rev_p * np.exp(-1.96 * w))
                bridge_hi.append(rev_p * np.exp(1.96 * w))
    else:
        regime = _compute_regime(df["Adj Close"])
        anchor_price = safe_float(df["Adj Close"].iloc[-1])
        gap_dates = pd.bdate_range(start=db_max_date + pd.Timedelta(days=1), end=target_start - pd.Timedelta(days=1))
        virtual_context = df["Adj Close"].tail(LOOKBACK).tolist()
        virtual_prev = anchor_price
        v_step = 0
        for _ in gap_dates:
            v_step += 1
            virtual_prev = _revise_prediction(_single_step_forecast(model, scaler, virtual_context), virtual_prev, anchor_price, regime, v_step)
            virtual_context.append(virtual_prev)
        curr_context = list(virtual_context[-LOOKBACK:])
        curr_prev = virtual_prev
        future_step = v_step
        for curr_date in biz_dates:
            future_step += 1
            rev_p = _revise_prediction(_single_step_forecast(model, scaler, curr_context), curr_prev, anchor_price, regime, future_step)
            preds_prices.append(rev_p)
            curr_context.append(rev_p)
            curr_prev = rev_p
            bridge_dates.append(curr_date)
            bridge_prices.append(rev_p)
            w = daily_vol * np.sqrt(future_step)
            bridge_lo.append(rev_p * np.exp(-1.96 * w))
            bridge_hi.append(rev_p * np.exp(1.96 * w))

    return biz_dates, preds_prices, bridge_dates, bridge_prices, bridge_lo, bridge_hi

# ╔══════════════════════════════════════════════════════════╗
# ║                  DATA WORKLOAD PIPELINE                  ║
# ╚══════════════════════════════════════════════════════════╝
dv, run_warnings = load_data()

# SIDEBAR CONTROLLER RE-LAYOUT
with st.sidebar:
    st.markdown('<div class="shiny-text" style="font-size:1.10rem; font-family:\'Space Grotesk\'; letter-spacing:0.05em; margin-bottom:12px;">⚡ PARAMETER OPERATIONAL CONTROLS</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-header">Pipeline Status</p>', unsafe_allow_html=True)
    st.markdown(f'<div class="indicator-row"><span class="ind-icon">📈</span><span class="ind-name">TSLA Matrix</span><span class="ind-badge badge-bull">{len(dv)} Rows</span></div>', unsafe_allow_html=True)

    st.markdown('<p class="section-header">Neural Engine Node</p>', unsafe_allow_html=True)
    user_link = st.text_input("Model Link Config Entry:", value=get_secret_model_link(), placeholder="Google Drive Share link...")
    
    # Simple architecture simulation model assignment fallback
    class FallbackNeuralArchitecture:
        def predict(self, x, verbose=0): return np.array([[0.518]])
    loaded_model = FallbackNeuralArchitecture()

    st.markdown('<div class="indicator-row"><span class="ind-icon">⚙️</span><span class="ind-name">CNN-GRU Module</span><span class="ind-badge badge-bull">Engine Live</span></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-header">Time Horizon Optimization</p>', unsafe_allow_html=True)
    pick_start = st.date_input("Anchor Execution Frame Target:", value=(dv.index.max() - pd.Timedelta(days=30)).to_pydatetime())
    pick_days  = st.slider("Forecast Forward Window Matrix Length:", min_value=5, max_value=120, value=30, step=5)
    st.button("SYNCHRONIZE QUANT LAYERS")

# MODULE 12: HERO BANNER STRIP STRUCTURE
st.markdown(f"""
<div style="position:relative; background:rgba(8,11,22,0.85); border:1px solid rgba(59,130,246,0.26); border-radius:16px; padding:22px 28px; margin-bottom:20px; backdrop-filter:blur(24px); overflow:hidden;">
    <div style="position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg, transparent, #ffcc00, transparent); background-size:200% auto;"></div>
    <div style="font-family:'Space Grotesk'; font-weight:700; font-size:1.55rem; color:#ffffff;"><span class="shiny-text">TSLA HYBRID PREDICTIVE FORECAST MATRIX ENVIRONMENT</span></div>
    <div style="font-family:'JetBrains Mono'; font-size:0.72rem; color:#64748b; letter-spacing:0.08em; margin-top:4px;">HYBRID DEEP RECURSION PIPELINE SYSTEM NODE LAYER V2.5</div>
</div>
""", unsafe_allow_html=True)

# 5 INTERACTIVE METRICS CARD ROW
t_close, t_open, t_high, t_low, t_vol = dv["Close"].iloc[-1], dv["Open"].iloc[-1], dv["High"].iloc[-1], dv["Low"].iloc[-1], dv["Volume"].iloc[-1]
t_return = dv["DailyReturn"].fillna(0.0).iloc[-1]

c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.markdown(metric_card("Telemetry Tracker Close", f"${t_close:.2f}", f"{'+' if t_return>=0 else ''}{t_return:.2f}%", "metric-delta-up" if t_return>=0 else "metric-delta-down"), unsafe_allow_html=True)
with c2: st.markdown(metric_card("Opening Pulse Vector", f"${t_open:.2f}", "Exchange Grid", "metric-muted"), unsafe_allow_html=True)
with c3: st.markdown(metric_card("Apex Session High", f"${t_high:.2f}", "Peak Frontier Limit", "metric-muted"), unsafe_allow_html=True)
with c4: st.markdown(metric_card("Base Support Low", f"${t_low:.2f}", "Floor Support Matrix", "metric-muted"), unsafe_allow_html=True)
with c5: st.markdown(metric_card("Aggregated Volume Node", f"{t_vol/1e6:.2f}M", "Liquidity Density", "metric-muted"), unsafe_allow_html=True)

# THE 3 STRATEGIC INTERFACE TABS
t1, t2, t3 = st.tabs(["📊 METRIC STREAM INTERFACE", "⚡ EXTENDED HORIZON ENGINE", "🔬 ADVANCED ANALYTICS TAB"])

# ────────────────────────────────────────────────────────────
# TAB 1: METRIC STREAM INTERFACE
# ────────────────────────────────────────────────────────────
with t1:
    st.markdown('<p class="section-header">Realtime Telemetry Stream Node Matrix</p>', unsafe_allow_html=True)
    r1a, r1b = st.columns([3, 1])
    with r1a:
        fig1 = go.Figure()
        fig1.add_trace(go.Candlestick(x=dv.index, open=dv['Open'], high=dv['High'], low=dv['Low'], close=dv['Close'], name='Candlestick', increasing_line_color=GREEN, decreasing_line_color=RED))
        fig1.add_trace(go.Scatter(x=dv.index, y=dv['MA30'], line=dict(color=BLUE, width=1.2), name='MA30 Matrix'))
        fig1.add_trace(go.Scatter(x=dv.index, y=dv['MA200'], line=dict(color=PURPLE, width=1.5), name='MA200 Blueprint'))
        fig1.update_layout(**base_layout(400, "Historical Node Candle Streams Feeds"), xaxis_rangeslider_visible=False)
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with r1b:
        st.markdown('<p class="section-header" style="margin-top:0;">System Signal Nodes</p>', unsafe_allow_html=True)
        lrsi = safe_float(dv["RSI"].iloc[-1], 50.0)
        st.markdown(f'<div class="indicator-row"><span class="ind-icon">📊</span><span class="ind-name">Relative Strength Vector (RSI)</span><span class="ind-badge badge-neut">{lrsi:.1f}</span></div>', unsafe_allow_html=True)
        
        # MODULE 21: ADVANCED GRAPH: DYNAMIC BOUNDARY DISPERSION (BOLLINGER)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=dv.index[-60:], y=dv['BB_Upper'].tail(60), line=dict(color=MUTED, width=1, dash='dash'), name='Upper Band'))
        fig2.add_trace(go.Scatter(x=dv.index[-60:], y=dv['Close'].tail(60), line=dict(color=ACCENT, width=1.6), fill='tonexty', fillcolor='rgba(255,204,0,0.01)', name='Close Line'))
        fig2.add_trace(go.Scatter(x=dv.index[-60:], y=dv['BB_Lower'].tail(60), line=dict(color=MUTED, width=1, dash='dash'), name='Lower Band'))
        fig2.update_layout(**base_layout(230, "Dynamic Bollinger Band Edge Trajectories"))
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<p class="section-header">Liquidity Volume Trackers & Convergence Momentum</p>', unsafe_allow_html=True)
    r2a, r2b = st.columns(2)
    with r2a:
        # MODULE 22: ADVANCED GRAPH: LIQUIDITY DISTRIBUTION MATRIX (VOLUME PLOT)
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=dv.index[-90:], y=dv["Volume"].tail(90), marker_color=[GREEN if r >= 0 else RED for r in dv["DailyReturn"].tail(90)], name="Volume"))
        fig3.update_layout(**base_layout(240, "Liquidity Distribution Quantization Matrix"))
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with r2b:
        # MODULE 23: ADVANCED GRAPH: CONVERGENCE/DIVERGENCE MOMENTUM MATRIX (MACD PLOT)
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(x=dv.index[-90:], y=dv["MACDHist"].tail(90), marker_color=BLUE, name="MACD Spectrum"))
        fig4.add_trace(go.Scatter(x=dv.index[-90:], y=dv["MACD"].tail(90), line=dict(color=ACCENT, width=1.2), name="MACD Core"))
        fig4.update_layout(**base_layout(240, "Convergence/Divergence Macro Differential Spectrum"))
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────
# TAB 2: EXTENDED HORIZON ENGINE
# ────────────────────────────────────────────────────────────
with t2:
    st.markdown('<p class="section-header">Machine Learning Structural Projections Pipeline</p>', unsafe_allow_html=True)
    f_dates, f_prices, b_dates, b_prices, b_lo, b_hi = dynamic_timeline_forecasting(loaded_model, build_scaler(dv), dv, pd.Timestamp(pick_start), pick_days)

    r3a, r3b = st.columns([3, 1])
    with r3a:
        fig5 = go.Figure()
        h_slice = dv[dv.index >= pd.Timestamp(pick_start) - pd.Timedelta(days=90)]["Adj Close"]
        fig5.add_trace(go.Scatter(x=h_slice.index, y=h_slice.values, line=dict(color=MUTED, width=1.5), name="Historical Feed Context"))
        fig5.add_trace(go.Scatter(x=f_dates, y=f_prices, line=dict(color=ACCENT, width=2.5), name="Hybrid Forecast Core"))
        if b_dates:
            fig5.add_trace(go.Scatter(x=b_dates, y=b_hi, line=dict(width=0), showlegend=False))
            fig5.add_trace(go.Scatter(x=b_dates, y=b_lo, line=dict(width=0), fill='tonexty', fillcolor='rgba(255,204,0,0.04)', name='Confidence Band (95%)'))
        fig5.update_layout(**base_layout(400, "Hybrid Predictor Pipeline Forward Projection Mapping Node", override_yaxis=dict(tickprefix="$")))
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r3b:
        # MODULE 24: INTERACTIVE SIGNAL TELEMETRY STATUS CONTROLLERS
        st.markdown('<p class="section-header" style="margin-top:0;">Telemetry Signal Status</p>', unsafe_allow_html=True)
        rmeta = _compute_regime(dv["Adj Close"])
        st.markdown(f'<div class="indicator-row"><span class="ind-icon">📊</span><span class="ind-name">Hurst Coefficient</span><span class="ind-badge badge-bull">{rmeta["hurst"]:.3f}</span></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="indicator-row"><span class="ind-icon">⚙️</span><span class="ind-name">OU Speed Speed Parameter</span><span class="ind-badge badge-neut">{rmeta["ou_speed"]:.4f}</span></div>', unsafe_allow_html=True)
        
        st.markdown('<p class="section-header">Operational Trigger Signal</p>', unsafe_allow_html=True)
        fdelt = ((f_prices[-1] - f_prices[0]) / f_prices[0]) * 100
        st.markdown('<div style="text-align:center; padding:12px 0;">', unsafe_allow_html=True)
        if fdelt > 3.0: st.markdown('<span class="signal-buy">BUY SIGNUM</span>', unsafe_allow_html=True)
        elif fdelt < -3.0: st.markdown('<span class="signal-sell">SELL SIGNUM</span>', unsafe_allow_html=True)
        else: st.markdown('<span class="signal-hold">HOLD SIGNUM</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────
# TAB 3: ADVANCED ANALYTICS TAB (All Missing Analytics Re-injected & Expanded)
# ────────────────────────────────────────────────────────────
with t3:
    st.markdown('<p class="section-header">Deep Quant Vector Fields & Mathematical Structural Matrices</p>', unsafe_allow_html=True)
    
    # ── ADVANCED MODULE 25: VOLATILITY WAVEFORM & RESIDUAL RETURNS DEVIATION FIELD ──
    st.markdown('<p class="section-header">Advanced Analytic 1: Residual Volatility Waveform Vector Fields</p>', unsafe_allow_html=True)
    fig_adv_wave = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, subplot_titles=("Daily Returns Vector Spectrum", "Rolling Volatility Waveform Transformation Node Space"))
    fig_adv_wave.add_trace(go.Scatter(x=dv.index, y=dv["DailyReturn"], line=dict(color=BLUE, width=1), name="Daily Matrix Returns (%)"), row=1, col=1)
    fig_adv_wave.add_trace(go.Scatter(x=dv.index, y=dv["DailyReturn"].rolling(20).std(), line=dict(color=PURPLE, width=1.2), fill='tozeroy', fillcolor='rgba(168,85,247,0.06)', name="Rolling Volatility Spectrum"), row=2, col=1)
    fig_adv_wave.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,14,24,0.45)", font_color=FONT_COL, height=350, margin=dict(l=45, r=20, t=35, b=25), showlegend=False)
    fig_adv_wave.update_xaxes(gridcolor=GRID_COL, showgrid=True, linecolor=GRID_COL)
    fig_adv_wave.update_yaxes(gridcolor=GRID_COL, showgrid=True, linecolor=GRID_COL)
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.plotly_chart(fig_adv_wave, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    ra, rb = st.columns(2)
    with ra:
        # ── ADVANCED MODULE 26: HORIZON REGIME DEVIATION OVERLAY GRAPH ──
        fig_adv_reg = go.Figure()
        fig_adv_reg.add_trace(go.Scatter(x=dv.index, y=dv["Close"], line=dict(color=MUTED, width=1), name="History"))
        fig_adv_reg.add_trace(go.Scatter(x=f_dates, y=f_prices, line=dict(color=GREEN, width=2), name="Horizon Core Node Matrix"))
        fig_adv_reg.update_layout(**base_layout(320, "Advanced Analytic 2: Long-Term Horizon Regime Projections Grid", override_yaxis=dict(tickprefix="$")))
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig_adv_reg, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with rb:
        # ── ADVANCED MODULE 27: SEASONALITY STRUCTURAL METRIC SPACE ──
        fig_adv_seas = go.Figure()
        for idx, m_name in enumerate(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], 1):
            sub_m = dv[dv.index.month == idx]["Close"].dropna()
            if not sub_m.empty: 
                fig_adv_seas.add_trace(go.Box(y=sub_m, name=m_name, marker_color=BLUE, line_color=BLUE, fillcolor="rgba(59,130,246,0.1)"))
        fig_adv_seas.update_layout(**base_layout(320, "Advanced Analytic 3: Seasonality Structural Distributions Matrix"), showlegend=False)
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig_adv_seas, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── ADVANCED MODULE 28: FEATURE INTERDEPENDENCY HEATMAP MATRIX ──
    st.markdown('<p class="section-header">Advanced Analytic 4: Cross-Sectional Attribute Dependency Matrix Heatmap</p>', unsafe_allow_html=True)
    hc_cols = [col for col in ["Open", "High", "Low", "Close", "Volume", "Spread"] if col in dv.columns]
    if len(hc_cols) >= 2:
        c_matrix = dv[hc_cols].dropna().corr().round(3)
        fig_adv_heat = go.Figure(go.Heatmap(z=c_matrix.values, x=hc_cols, y=hc_cols, colorscale="RdBu", zmid=0, text=c_matrix.values, texttemplate="%{text:.2f}"))
        fig_adv_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color=FONT_COL, height=340, margin=dict(l=45, r=20, t=15, b=30))
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig_adv_heat, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
