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
   FONTS & BASE
═══════════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

/* ═══════════════════════════════════════════════
   AURORA BACKGROUND  (reactbits: Soft Aurora)
   Two slow-drifting radial blobs + base dark bg
═══════════════════════════════════════════════ */
@keyframes aurora-drift-a {
  0%   { transform: translate(0%,   0%)   scale(1);    opacity: 0.55; }
  33%  { transform: translate(8%,  -12%)  scale(1.15); opacity: 0.45; }
  66%  { transform: translate(-6%,  10%)  scale(0.95); opacity: 0.60; }
  100% { transform: translate(0%,   0%)   scale(1);    opacity: 0.55; }
}
@keyframes aurora-drift-b {
  0%   { transform: translate(0%,  0%)   scale(1);    opacity: 0.40; }
  40%  { transform: translate(-9%, 14%)  scale(1.10); opacity: 0.50; }
  75%  { transform: translate(7%, -8%)   scale(1.05); opacity: 0.35; }
  100% { transform: translate(0%,  0%)   scale(1);    opacity: 0.40; }
}
@keyframes aurora-drift-c {
  0%   { transform: translate(0%, 0%)  scale(1);   }
  50%  { transform: translate(5%, 6%)  scale(1.08);}
  100% { transform: translate(0%, 0%)  scale(1);   }
}

.stApp {
  background-color: #080a10 !important;
  color: #e2e4f0 !important;
  font-family: 'Inter', sans-serif !important;
  overflow-x: hidden;
}

/* Aurora blobs injected behind all content */
.stApp::before {
  content: "";
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background:
    radial-gradient(ellipse 80% 60% at 15% 20%,  rgba(91,141,238,0.18) 0%, transparent 70%),
    radial-gradient(ellipse 60% 80% at 85% 75%,  rgba(160,110,225,0.14) 0%, transparent 65%),
    radial-gradient(ellipse 50% 40% at 50% 50%,  rgba(232,200,74,0.06) 0%, transparent 60%);
  animation: aurora-drift-a 18s ease-in-out infinite;
}
.stApp::after {
  content: "";
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background:
    radial-gradient(ellipse 70% 55% at 75% 15%,  rgba(38,212,124,0.10) 0%, transparent 65%),
    radial-gradient(ellipse 55% 70% at 20% 85%,  rgba(91,141,238,0.12) 0%, transparent 60%);
  animation: aurora-drift-b 24s ease-in-out infinite;
}

/* Dot-field grid overlay (reactbits: Dot Field) */
.stApp > div {
  background-image: radial-gradient(circle, rgba(255,255,255,0.035) 1px, transparent 1px);
  background-size: 28px 28px;
  position: relative; z-index: 1;
}

/* ═══════════════════════════════════════════════
   SIDEBAR
═══════════════════════════════════════════════ */
[data-testid="stSidebar"] {
  background: rgba(10, 12, 20, 0.92) !important;
  border-right: 1px solid rgba(91,141,238,0.18) !important;
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
}
[data-testid="stSidebar"]::before {
  content: "";
  position: absolute; inset: 0; pointer-events: none; z-index: 0;
  background: linear-gradient(180deg, rgba(91,141,238,0.06) 0%, transparent 40%);
}

/* ═══════════════════════════════════════════════
   TABS  (reactbits: StarBorder-style active glow)
═══════════════════════════════════════════════ */
@keyframes tab-glow-pulse {
  0%, 100% { box-shadow: 0 2px 12px rgba(232,200,74,0.30); }
  50%       { box-shadow: 0 2px 20px rgba(232,200,74,0.55); }
}
.stTabs [data-baseweb="tab-list"] {
  gap: 0px !important;
  background: rgba(10,12,20,0.80) !important;
  border-bottom: 1px solid rgba(91,141,238,0.15) !important;
  backdrop-filter: blur(12px);
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: rgba(160,168,200,0.65) !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.78rem !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  padding: 16px 32px !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  margin-bottom: -1px !important;
  transition: color 0.25s ease, border-color 0.25s ease !important;
  position: relative;
}
.stTabs [data-baseweb="tab"]:hover {
  color: rgba(232,200,74,0.80) !important;
}
.stTabs [aria-selected="true"] {
  color: #e8c84a !important;
  border-bottom: 2px solid #e8c84a !important;
  background: transparent !important;
  animation: tab-glow-pulse 2.5s ease-in-out infinite;
}

/* ═══════════════════════════════════════════════
   METRIC CARDS  (reactbits: Glowing Border + shimmer)
═══════════════════════════════════════════════ */
@keyframes shimmer-sweep {
  0%   { background-position: -200% center; }
  100% { background-position:  200% center; }
}
@keyframes card-fade-in {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0);   }
}
@keyframes border-glow-spin {
  0%   { opacity: 0.4; }
  50%  { opacity: 0.9; }
  100% { opacity: 0.4; }
}

.metric-card {
  position: relative;
  background: rgba(15,18,30,0.75) !important;
  border: 1px solid rgba(91,141,238,0.20) !important;
  border-radius: 12px !important;
  padding: 20px 18px !important;
  text-align: center;
  min-height: 92px;
  backdrop-filter: blur(16px);
  overflow: hidden;
  animation: card-fade-in 0.5s ease forwards;
  transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.metric-card::before {
  content: "";
  position: absolute; inset: 0; border-radius: 12px;
  background: linear-gradient(105deg,
    transparent 35%,
    rgba(91,141,238,0.07) 50%,
    transparent 65%);
  background-size: 200% 100%;
  animation: shimmer-sweep 3.5s linear infinite;
  pointer-events: none;
}
.metric-card:hover {
  border-color: rgba(91,141,238,0.50) !important;
  box-shadow: 0 0 20px rgba(91,141,238,0.15), 0 0 40px rgba(91,141,238,0.06);
}
.metric-label {
  color: rgba(160,168,200,0.60);
  font-size: 0.68rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  margin-bottom: 8px;
  font-family: 'Inter', sans-serif;
}
.metric-value {
  color: #e2e4f0;
  font-size: 1.45rem;
  font-weight: 700;
  font-family: 'JetBrains Mono', monospace;
  word-break: break-all;
  text-shadow: 0 0 20px rgba(91,141,238,0.25);
}
.metric-delta-up   { color: #26d47c; font-size: 0.80rem; margin-top: 5px; font-family: 'JetBrains Mono', monospace; }
.metric-delta-down { color: #f05252; font-size: 0.80rem; margin-top: 5px; font-family: 'JetBrains Mono', monospace; }
.metric-muted      { color: rgba(160,168,200,0.45); font-size: 0.80rem; margin-top: 5px; }

/* ═══════════════════════════════════════════════
   SIGNAL BADGE  (reactbits: pulsing glow ring)
═══════════════════════════════════════════════ */
@keyframes pulse-ring-buy {
  0%   { box-shadow: 0 0 0  0   rgba(38,212,124,0.70), 0 0 24px rgba(38,212,124,0.30); }
  70%  { box-shadow: 0 0 0  14px rgba(38,212,124,0),   0 0 32px rgba(38,212,124,0.15); }
  100% { box-shadow: 0 0 0  0   rgba(38,212,124,0),    0 0 24px rgba(38,212,124,0.30); }
}
@keyframes pulse-ring-sell {
  0%   { box-shadow: 0 0 0  0   rgba(240,82,82,0.70),  0 0 24px rgba(240,82,82,0.30); }
  70%  { box-shadow: 0 0 0  14px rgba(240,82,82,0),    0 0 32px rgba(240,82,82,0.15); }
  100% { box-shadow: 0 0 0  0   rgba(240,82,82,0),     0 0 24px rgba(240,82,82,0.30); }
}
@keyframes pulse-ring-hold {
  0%   { box-shadow: 0 0 0  0   rgba(232,200,74,0.60),  0 0 24px rgba(232,200,74,0.25); }
  70%  { box-shadow: 0 0 0  14px rgba(232,200,74,0),    0 0 32px rgba(232,200,74,0.12); }
  100% { box-shadow: 0 0 0  0   rgba(232,200,74,0),     0 0 24px rgba(232,200,74,0.25); }
}
.signal-buy {
  background: rgba(13,51,38,0.85);
  color: #26d47c;
  border: 1.5px solid #26d47c;
  border-radius: 8px;
  padding: 12px 40px;
  font-weight: 800;
  font-size: 1.9rem;
  font-family: 'JetBrains Mono', monospace;
  display: inline-block;
  letter-spacing: 0.12em;
  animation: pulse-ring-buy 2s ease-out infinite;
  backdrop-filter: blur(10px);
}
.signal-sell {
  background: rgba(53,13,13,0.85);
  color: #f05252;
  border: 1.5px solid #f05252;
  border-radius: 8px;
  padding: 12px 40px;
  font-weight: 800;
  font-size: 1.9rem;
  font-family: 'JetBrains Mono', monospace;
  display: inline-block;
  letter-spacing: 0.12em;
  animation: pulse-ring-sell 2s ease-out infinite;
  backdrop-filter: blur(10px);
}
.signal-hold {
  background: rgba(42,42,16,0.85);
  color: #e8c84a;
  border: 1.5px solid #e8c84a;
  border-radius: 8px;
  padding: 12px 40px;
  font-weight: 800;
  font-size: 1.9rem;
  font-family: 'JetBrains Mono', monospace;
  display: inline-block;
  letter-spacing: 0.12em;
  animation: pulse-ring-hold 2s ease-out infinite;
  backdrop-filter: blur(10px);
}

/* ═══════════════════════════════════════════════
   SECTION HEADERS  (scanning line effect)
═══════════════════════════════════════════════ */
@keyframes scan-line {
  0%   { left: -100%; }
  100% { left:  110%; }
}
.section-header {
  position: relative;
  color: #e8c84a;
  font-size: 0.65rem;
  letter-spacing: 0.20em;
  text-transform: uppercase;
  margin-bottom: 12px;
  margin-top: 24px;
  font-family: 'Inter', sans-serif;
  font-weight: 600;
  overflow: hidden;
  padding-bottom: 6px;
}
.section-header::after {
  content: "";
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(232,200,74,0.5), transparent);
}
.section-header::before {
  content: "";
  position: absolute;
  bottom: 0;
  width: 40%;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(232,200,74,0.9), transparent);
  animation: scan-line 3s linear infinite;
}

/* ═══════════════════════════════════════════════
   HEADER BANNER  (reactbits: Beams-style top strip)
═══════════════════════════════════════════════ */
@keyframes beam-sweep {
  0%   { transform: translateX(-100%) skewX(-15deg); opacity: 0; }
  20%  { opacity: 1; }
  80%  { opacity: 1; }
  100% { transform: translateX(600%) skewX(-15deg); opacity: 0; }
}
.app-header-banner {
  position: relative;
  background: rgba(10,12,22,0.70);
  border: 1px solid rgba(91,141,238,0.18);
  border-radius: 14px;
  padding: 20px 28px 16px;
  margin-bottom: 8px;
  overflow: hidden;
  backdrop-filter: blur(20px);
}
.app-header-banner::before {
  content: "";
  position: absolute;
  top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg,
    transparent 0%, rgba(91,141,238,0.8) 30%,
    rgba(232,200,74,0.8) 60%, transparent 100%);
  animation: beam-sweep 5s ease-in-out infinite 1s;
}
.app-header-banner::after {
  content: "";
  position: absolute;
  top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg,
    transparent 0%, rgba(160,110,225,0.6) 40%,
    rgba(38,212,124,0.6) 70%, transparent 100%);
  animation: beam-sweep 5s ease-in-out infinite 3.5s;
}
.app-title {
  font-family: 'Inter', sans-serif;
  font-weight: 700;
  font-size: 1.35rem;
  color: #e2e4f0;
  letter-spacing: 0.02em;
  margin: 0;
}
.app-subtitle {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.68rem;
  color: rgba(160,168,200,0.55);
  letter-spacing: 0.10em;
  margin-top: 3px;
}
.live-dot {
  display: inline-block;
  width: 7px; height: 7px;
  background: #26d47c;
  border-radius: 50%;
  margin-right: 7px;
  vertical-align: middle;
  animation: pulse-ring-buy 1.8s ease-out infinite;
  box-shadow: 0 0 6px rgba(38,212,124,0.8);
}

/* ═══════════════════════════════════════════════
   EMPTY STATE
═══════════════════════════════════════════════ */
.empty-state {
  background: rgba(15,18,30,0.60);
  border: 1px dashed rgba(91,141,238,0.25);
  border-radius: 12px;
  padding: 40px 28px;
  text-align: center;
  color: rgba(160,168,200,0.55);
  backdrop-filter: blur(10px);
}
.empty-state-icon { font-size: 2.2rem; margin-bottom: 12px; }
.empty-state-msg  { font-size: 0.88rem; line-height: 1.7; font-family: 'Inter', sans-serif; }

/* ═══════════════════════════════════════════════
   INDICATOR ROWS  (breakdown list)
═══════════════════════════════════════════════ */
@keyframes row-slide-in {
  from { opacity: 0; transform: translateX(-10px); }
  to   { opacity: 1; transform: translateX(0); }
}
.indicator-row {
  display: flex; align-items: center; gap: 10px;
  padding: 8px 12px;
  margin-bottom: 4px;
  border-radius: 8px;
  background: rgba(255,255,255,0.025);
  border: 1px solid rgba(255,255,255,0.04);
  font-family: 'Inter', sans-serif;
  font-size: 0.82rem;
  color: #c8ccd8;
  animation: row-slide-in 0.3s ease forwards;
  transition: background 0.2s ease;
}
.indicator-row:hover { background: rgba(91,141,238,0.06); }
.ind-icon  { font-size: 0.85rem; flex-shrink: 0; }
.ind-name  { font-weight: 600; flex: 1; }
.ind-badge {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.68rem;
  letter-spacing: 0.08em;
  padding: 2px 8px;
  border-radius: 4px;
  font-weight: 600;
}
.badge-bull { background: rgba(38,212,124,0.12); color: #26d47c; border: 1px solid rgba(38,212,124,0.25); }
.badge-bear { background: rgba(240,82,82,0.12);  color: #f05252; border: 1px solid rgba(240,82,82,0.25); }
.badge-neut { background: rgba(232,200,74,0.10); color: #e8c84a; border: 1px solid rgba(232,200,74,0.20); }
.badge-warn { background: rgba(240,82,82,0.18);  color: #ff7070; border: 1px solid rgba(240,82,82,0.35); }

/* ═══════════════════════════════════════════════
   FORECAST INFO BANNER
═══════════════════════════════════════════════ */
@keyframes info-slide {
  from { opacity: 0; transform: translateY(-6px); }
  to   { opacity: 1; transform: translateY(0); }
}
.forecast-info-bar {
  background: rgba(91,141,238,0.07);
  border: 1px solid rgba(91,141,238,0.22);
  border-radius: 10px;
  padding: 12px 18px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem;
  color: rgba(160,168,200,0.85);
  letter-spacing: 0.04em;
  margin-bottom: 16px;
  animation: info-slide 0.4s ease forwards;
  display: flex; align-items: center; gap: 10px;
}
.forecast-info-bar .fi-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: #5b8dee;
  box-shadow: 0 0 8px rgba(91,141,238,0.8);
  flex-shrink: 0;
  animation: border-glow-spin 2s ease-in-out infinite;
}

/* ═══════════════════════════════════════════════
   CHART WRAPPER  (glowing border + scan line)
═══════════════════════════════════════════════ */
@keyframes chart-scan {
  0%   { top: 0%; opacity: 0.6; }
  100% { top: 100%; opacity: 0; }
}
.chart-wrap {
  position: relative;
  border: 1px solid rgba(91,141,238,0.15);
  border-radius: 12px;
  overflow: hidden;
  background: rgba(8,10,16,0.50);
  padding: 4px;
  transition: border-color 0.3s ease;
}
.chart-wrap:hover { border-color: rgba(91,141,238,0.35); }
.chart-wrap::after {
  content: "";
  position: absolute;
  left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(91,141,238,0.35), transparent);
  animation: chart-scan 6s linear infinite;
  pointer-events: none;
}

/* ═══════════════════════════════════════════════
   SIDEBAR INPUTS
═══════════════════════════════════════════════ */
.stTextInput input, .stNumberInput input {
  background: rgba(10,12,22,0.80) !important;
  border: 1px solid rgba(91,141,238,0.22) !important;
  color: #e2e4f0 !important;
  border-radius: 8px !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.85rem !important;
  transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
  border-color: rgba(91,141,238,0.55) !important;
  box-shadow: 0 0 0 3px rgba(91,141,238,0.10) !important;
  outline: none !important;
}
.stSelectbox > div > div {
  background: rgba(10,12,22,0.80) !important;
  border: 1px solid rgba(91,141,238,0.22) !important;
  border-radius: 8px !important;
}
.stSlider [data-baseweb="slider"] { padding: 0 4px; }
[data-testid="stSliderThumb"] { background: #5b8dee !important; box-shadow: 0 0 8px rgba(91,141,238,0.6) !important; }

/* ═══════════════════════════════════════════════
   SIDEBAR BUTTONS
═══════════════════════════════════════════════ */
@keyframes btn-shimmer {
  0%   { background-position: -200% center; }
  100% { background-position:  200% center; }
}
.stButton > button {
  background: linear-gradient(105deg, #0f1a3a 0%, #1a2a5e 50%, #0f1a3a 100%) !important;
  background-size: 200% auto !important;
  border: 1px solid rgba(91,141,238,0.40) !important;
  color: #7ab0ff !important;
  border-radius: 8px !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.80rem !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  transition: all 0.3s ease !important;
}
.stButton > button:hover {
  animation: btn-shimmer 1.5s linear infinite !important;
  border-color: rgba(91,141,238,0.70) !important;
  box-shadow: 0 0 18px rgba(91,141,238,0.25) !important;
  color: #aad0ff !important;
}
.stButton > button[kind="primary"] {
  background: linear-gradient(105deg, #122040 0%, #1e3a7a 50%, #122040 100%) !important;
  background-size: 200% auto !important;
  border-color: rgba(91,141,238,0.55) !important;
  color: #90c0ff !important;
}

/* ═══════════════════════════════════════════════
   MISC
═══════════════════════════════════════════════ */
hr { border-color: rgba(91,141,238,0.12) !important; }
#MainMenu, footer { visibility: hidden; }
.stAlert > div { border-radius: 10px !important; backdrop-filter: blur(10px); }

/* Streamlit default caption / markdown refinements */
.stMarkdown p, .stCaption { color: rgba(160,168,200,0.70) !important; font-family: 'Inter', sans-serif !important; }
label, .stWidgetLabel { color: rgba(160,168,200,0.80) !important; font-family: 'Inter', sans-serif !important; font-size: 0.80rem !important; }
</style>
""", unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════╗
# ║                    CONSTANTS                             ║
# ╚══════════════════════════════════════════════════════════╝

CSV_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TSLA_1.csv")
LOOKBACK  = 60
PLOT_BG   = "#0d0f14"
GRID_COL  = "#1a1d28"
FONT_COL  = "#e8eaf0"
ACCENT    = "#e8c84a"
GREEN     = "#26d47c"
RED       = "#f05252"
BLUE      = "#5b8dee"
PURPLE    = "#a06ee1"
MUTED     = "#7a8099"

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
        f'<div class="empty-state">'
        f'<div class="empty-state-icon">{icon}</div>'
        f'<div class="empty-state-msg">{msg}</div>'
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
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font_color=FONT_COL, height=height,
        margin=dict(l=0, r=0, t=36, b=0),
        title=dict(text=title, font=dict(size=13, color=MUTED)),
        xaxis=dict(gridcolor=GRID_COL, showgrid=True),
        yaxis=dict(gridcolor=GRID_COL, showgrid=True),
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
    """
    Estimate Hurst exponent via R/S analysis on a price series.
    H < 0.45  → mean-reverting  (anti-persistent)
    H ≈ 0.50  → random walk
    H > 0.55  → trending        (persistent)
    Returns 0.5 if estimation fails.
    """
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
    """
    Analyse the last 252 days of prices and return regime parameters:
      - hurst:       Hurst exponent H
      - trend_slope: annualised log-return slope (positive = up, negative = down)
      - mean_price:  long-run equilibrium (200d EMA)
      - vol:         daily log-return std dev
      - ou_speed:    Ornstein-Uhlenbeck mean-reversion speed (κ)
    """
    prices = history_series.dropna().values[-252:]
    if len(prices) < 30:
        return dict(hurst=0.5, trend_slope=0.0, mean_price=prices[-1],
                    vol=0.02, ou_speed=0.05)

    log_rets  = np.diff(np.log(prices))
    vol       = float(np.std(log_rets)) if len(log_rets) > 1 else 0.02
    H         = _hurst_exponent(prices)

    # Trend slope: linear fit on log-prices, annualised
    x          = np.arange(len(prices))
    log_prices = np.log(prices)
    slope      = float(np.polyfit(x, log_prices, 1)[0]) * 252  # annualised

    # Long-run mean: 200-day EMA of the full window (or all available)
    ema_span   = min(200, len(prices))
    weights    = np.exp(np.linspace(-1, 0, ema_span))
    weights   /= weights.sum()
    mean_price = float(np.convolve(prices, weights[::-1], mode='valid')[-1])

    # OU mean-reversion speed κ: estimated from AR(1) on log-returns
    # κ ≈ -ln(AR1 coefficient) — higher κ = faster reversion
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
    """Run one CNN-GRU step. Returns inverse-transformed USD price."""
    x_sc  = scaler.transform(np.array(lookback_context[-LOOKBACK:]).reshape(-1, 1)).flatten()
    x_in  = np.array(x_sc, dtype=np.float32).reshape(1, LOOKBACK, 1)
    raw   = float(np.clip(model.predict(x_in, verbose=0)[0, 0], 0.0, 1.0))
    return float(scaler.inverse_transform([[raw]])[0, 0])


def _revise_prediction(
    raw_pred:       float,
    prev_price:     float,
    anchor_price:   float,   # price at step 0 (start of forecast horizon)
    regime:         dict,
    step:           int,
    total_steps:    int,
) -> float:
    """
    Revise a single CNN-GRU raw prediction.

    Design
    ──────
    The previous version applied per-step return magnitude damping which
    suppressed the CNN-GRU's natural step-to-step variation and caused the
    flatline.  The correct approach is:

    1. Trust the CNN-GRU's step-to-step return fully — this preserves the
       natural ups and downs (daily variation) that the model learned.

    2. Apply a CUMULATIVE drift correction only — if after `step` steps the
       total log-return from anchor_price has grown beyond what is plausible
       given historical volatility, nudge the excess back toward zero.
       This prevents 60-step compounding without flattening individual steps.

    3. Apply a light OU pull proportional to how far the current price has
       drifted from the long-run EMA.  Scaled so it only activates when the
       price is >10% away from the EMA.

    4. Hurst weighting: in a trending market (H>0.55) apply less OU pull;
       in a mean-reverting market (H<0.45) apply more.

    5. Hard cap per step at ±4σ so single outlier predictions don't corrupt
       the lookback window for subsequent steps.
    """
    H           = regime["hurst"]
    mean_price  = regime["mean_price"]
    ou_speed    = regime["ou_speed"]
    vol         = regime["vol"]

    if prev_price <= 0 or anchor_price <= 0:
        return raw_pred

    # ── 1. CNN-GRU raw step return — kept fully intact ────────────────────────
    raw_log_ret = np.log(max(raw_pred, 1e-6) / max(prev_price, 1e-6))

    # ── 2. Cumulative drift correction ────────────────────────────────────────
    # How far has the price drifted from anchor in total log-return terms?
    cumulative_log_ret = np.log(max(prev_price, 1e-6) / max(anchor_price, 1e-6))

    # Expected max absolute drift at this step depth: ±2σ√t (random-walk bound)
    drift_budget = 2.0 * vol * np.sqrt(step)

    # Correction: if cumulative drift exceeds budget, apply a restoring force
    # proportional to the excess — NOT capping the step return, just nudging
    drift_excess = abs(cumulative_log_ret) - drift_budget
    if drift_excess > 0:
        correction_sign     = -np.sign(cumulative_log_ret)   # oppose the drift
        drift_correction    = correction_sign * drift_excess * 0.15  # gentle 15%
    else:
        drift_correction = 0.0

    # ── 3. Hurst-scaled OU pull ────────────────────────────────────────────────
    # Only apply when price is meaningfully away from EMA (>5% gap)
    price_gap_pct = abs(prev_price - mean_price) / mean_price
    if price_gap_pct > 0.05:
        ou_pull = -ou_speed * np.log(max(prev_price, 1e-6) / max(mean_price, 1e-6))
        # Hurst scaling: H=0.55 → ou_scale=0.10 (trending, weak pull)
        #                H=0.50 → ou_scale=0.20
        #                H=0.40 → ou_scale=0.35 (mean-reverting, strong pull)
        ou_scale = float(np.interp(H, [0.35, 0.50, 0.65], [0.35, 0.20, 0.08]))
    else:
        ou_pull  = 0.0
        ou_scale = 0.0

    # ── 4. Compose final log-return ───────────────────────────────────────────
    # CNN-GRU drives the step; drift correction and OU are additive adjustments
    final_log_ret = raw_log_ret + drift_correction + (ou_scale * ou_pull)

    # Per-step hard cap at ±4σ — outlier guard only, not a routine constraint
    final_log_ret = float(np.clip(final_log_ret, -4.0 * vol, 4.0 * vol))

    revised_price = prev_price * np.exp(final_log_ret)

    # Absolute sanity bounds: price cannot fall below 20% or above 350% of EMA
    revised_price = float(np.clip(revised_price, mean_price * 0.20, mean_price * 3.50))
    return revised_price


def dynamic_timeline_forecasting(
    model, scaler, df: pd.DataFrame, start_date: pd.Timestamp, n_days: int
) -> tuple:
    """
    Hybrid forecast pipeline.

    Revision strategy (no Prophet / no SARIMAX)
    ─────────────────────────────────────────────
    Root cause of the perpetual-decline bug:
      The CNN-GRU is recursive. When the last 60 real prices form a declining
      sequence, each prediction feeds a lower value back in, compounding the
      decline over 60 steps into a smooth, unrealistic downtrend.

    Fix:
      After each raw CNN-GRU step we call `_revise_prediction` which:
        (a) Converts the raw price to a log-return and applies exponential
            magnitude damping so momentum cannot compound across 60 steps.
        (b) Uses the Hurst exponent of the last 252 days to detect whether the
            market is trending or mean-reverting and weights the CNN-GRU
            signal accordingly.
        (c) Applies an Ornstein-Uhlenbeck pull toward the long-run EMA,
            preventing the forecast from drifting to extremes.
      The revised price is then fed back as the next step's input — so the
      lookback window is populated with realistic, regime-aware values rather
      than a compounding one-directional slope.
    """
    db_max_date  = df.index.max()
    target_start = pd.Timestamp(start_date)
    biz_dates    = pd.bdate_range(start=target_start, periods=n_days)

    recent_ret = df["DailyReturn"].replace([np.inf, -np.inf], np.nan).dropna().tail(60)
    daily_vol  = (recent_ret.std() / 100) if len(recent_ret) >= 5 else 0.02

    preds_prices: list = []
    bridge_dates, bridge_prices, bridge_lo, bridge_hi = [], [], [], []

    # ── PATH A: start_date is within or at the database boundary ─────────────
    if target_start <= db_max_date:
        # anchor = last real price before forecast steps begin
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

        # Bridge: fill gap between DB end and chosen start
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

        # Target horizon: regime computed on full history including bridge
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

with st.spinner("Loading TSLA data…"):
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

with st.sidebar:
    st.markdown("## ⚡ TSLA Hub")
    st.markdown("---")

    st.markdown('<p class="section-header">Model Engine Status</p>', unsafe_allow_html=True)
    gdrive_url = st.text_input(
        "Model Resource Node", value=secret_url, type="password",
        help="Pre-loaded securely via secrets file."
    )
    load_btn = st.button("⬇ Load Model Core", width='stretch', type="primary")

    model_status_slot = st.empty()
    if st.session_state.get("model_loaded", False):
        model_status_slot.success("🟢 Systems Operational (Model Live)")
    else:
        model_status_slot.info("⏳ Initializing Deep Learning Core...")

    st.markdown("---")
    st.markdown('<p class="section-header">Forecast Matrix Parameters</p>', unsafe_allow_html=True)
    chosen_start_date = st.date_input("Forecast Start Boundary Point", value=df.index[-1].date())
    forecast_days     = st.slider("Trading Days Horizon Out", 5, 60, 30, 5)

    st.markdown("---")
    st.markdown('<p class="section-header">Trade Matrix Configuration</p>', unsafe_allow_html=True)
    entry_price  = st.number_input("Position Acquisition Entry ($)", min_value=0.0, value=0.0, step=0.01)
    position_qty = st.number_input("Asset Share Target Volume", min_value=1, value=10, step=1)
    risk_pct     = st.slider("Risk Exposure Cut-off (%)", 1, 20, 5)

if not st.session_state.get("model_loaded", False):
    url_to_load = gdrive_url.strip() if gdrive_url else secret_url.strip()
    if url_to_load:
        f_id = extract_gdrive_id(url_to_load)
        if f_id:
            try:
                st.session_state["model_obj"] = load_model_cached(f_id)
                st.session_state["model_loaded"] = True
                model_status_slot.success("🟢 Systems Operational (Model Live)")
            except Exception:
                st.session_state["model_loaded"] = False

if load_btn:
    url_clean = gdrive_url.strip() if gdrive_url else ""
    if url_clean:
        f_id = extract_gdrive_id(url_clean)
        if f_id:
            with st.sidebar:
                with st.spinner("Downloading model layer..."):
                    try:
                        st.session_state["model_obj"] = load_model_cached(f_id)
                        st.session_state["model_loaded"] = True
                        model_status_slot.success("✅ Model Ready")
                    except Exception as _me:
                        st.error(f"❌ {_me}")

model = st.session_state.get("model_obj", None)
eff_entry = entry_price if entry_price > 0.0 else current_price

f_dates, f_prices, f_lower, f_upper = None, None, None, None
b_dates, b_prices, b_lower, b_upper = None, None, None, None

if model is not None and scaler_ok:
    with st.spinner("Executing Double-Engine Structural Matrix..."):
        try:
            f_dates, f_prices, f_lower, f_upper, b_dates, b_prices, b_lower, b_upper = dynamic_timeline_forecasting(
                model, scaler, df, pd.Timestamp(chosen_start_date), n_days=forecast_days
            )
        except Exception as _err:
            st.error(f"Hybrid Matrix Strategy Core fault: {_err}")

# ╔══════════════════════════════════════════════════════════╗
# ║                    KPI METRICS BANNER                    ║
# ╚══════════════════════════════════════════════════════════╝

col_t, col_p, col_d, col_v, col_sp = st.columns([3, 2, 2, 2, 2])
with col_t:
    st.markdown("### ⚡ Tesla, Inc. &nbsp;`TSLA`")
    st.caption(f"Last database sync marker: {df.index[-1].strftime('%d %b %Y')}")
with col_p:
    d_cls = "metric-delta-up" if price_change >= 0 else "metric-delta-down"
    arrow = "▲" if price_change >= 0 else "▼"
    st.markdown(metric_card("Last Close", f"${current_price:.2f}", f"{arrow} ${abs(price_change):.2f} ({price_change_p:+.2f}%)", d_cls), unsafe_allow_html=True)
with col_d:
    st.markdown(metric_card("52-Week High", f"${safe_float(df['High'].tail(252).max()):.2f}"), unsafe_allow_html=True)
with col_v:
    st.markdown(metric_card("52-Week Low", f"${safe_float(df['Low'].tail(252).min()):.2f}"), unsafe_allow_html=True)
with col_sp:
    v_val = df["Volume"].tail(20).mean()
    st.markdown(metric_card("Avg Volume (20d)", f"{v_val/1e6:.1f}M" if v_val>=1e6 else f"{v_val/1e3:.0f}K"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📡  Trade Signals", "🔮  Forecast Engine", "📊  Advanced Visualizations"])

# ════════════════════════════════════════════════════════════
#  TAB 1 — TRADE SIGNALS
# ════════════════════════════════════════════════════════════
with tab1:
    rsi_now  = safe_float(df["RSI"].dropna().iloc[-1] if df["RSI"].dropna().any() else np.nan)
    macd_now = safe_float(df["MACD"].dropna().iloc[-1] if df["MACD"].dropna().any() else np.nan)
    sig_now  = safe_float(df["MACDSig"].dropna().iloc[-1] if df["MACDSig"].dropna().any() else np.nan)
    ma30_now = safe_float(df["MA30"].dropna().iloc[-1] if df["MA30"].dropna().any() else np.nan)
    ma90_now = safe_float(df["MA90"].dropna().iloc[-1] if df["MA90"].dropna().any() else np.nan)
    bb_up    = safe_float(df["BB_Upper"].dropna().iloc[-1] if df["BB_Upper"].dropna().any() else np.nan)
    bb_lo    = safe_float(df["BB_Lower"].dropna().iloc[-1] if df["BB_Lower"].dropna().any() else np.nan)

    # Ensure entry price reflects either custom input or falls back gracefully to current market value
    eff_entry = entry_price if entry_price > 0.0 else current_price

    tech_scores = {
        "RSI (14)": 1 if rsi_now < 30 else (-1 if rsi_now > 70 else 0),
        "MACD vs Signal": 1 if macd_now > sig_now else -1,
        "MA30 vs MA90": 1 if ma30_now > ma90_now else -1,
        "Bollinger Band": 1 if current_price < bb_lo else (-1 if current_price > bb_up else 0)
    }

    if f_prices is not None:
        # Evaluate model forecast trajectory against your SPECIFIC execution entry point rather than current close
        model_target = safe_float(f_prices[min(4, len(f_prices)-1)])
        model_pct = (model_target - eff_entry) / eff_entry * 100
        tech_scores["Model (5-day)"] = 1 if model_pct > 1.5 else (-1 if model_pct < -1.5 else 0)

    # Entry Valuation Strain Factor
    # If a user sets an entry price significantly above the current market price, penalize the entry strategy heavily.
    valuation_premium = (eff_entry - current_price) / current_price * 100
    if valuation_premium > 5.0:
        # Slashing score if buying over a 5% premium to prevent chasing an overpriced execution entry
        tech_scores["Entry Premium Guard"] = -2 
    elif valuation_premium < -5.0:
        # Extra points if buying at a deep discount relative to live market prints
        tech_scores["Entry Premium Guard"] = 1
    else:
        tech_scores["Entry Premium Guard"] = 0

    total_score = sum(tech_scores.values())
    if total_score >= 2:    signal_label, signal_css = "BUY",  "signal-buy"
    elif total_score <= -2: signal_label, signal_css = "SELL", "signal-sell"
    else:                   signal_label, signal_css = "HOLD", "signal-hold"

    stop_loss    = eff_entry * (1 - risk_pct / 100)
    take_profit  = eff_entry + max(eff_entry - stop_loss, 0.01) * 2.0

    left, right = st.columns([1, 2], gap="large")
    with left:
        st.markdown('<p class="section-header">Composite Signal Alignment</p>', unsafe_allow_html=True)
        st.markdown(f'<div style="text-align:center;padding:28px 0 16px"><div class="{signal_css}">{signal_label}</div></div>', unsafe_allow_html=True)
        
        st.markdown('<p class="section-header">Breakdown Parameters</p>', unsafe_allow_html=True)
        for name, sc in tech_scores.items():
            # FIXED: Handles the custom -2 score gracefully without broken text tags
            if sc >= 1:
                lbl, icon = "Bullish", "🟢"
            elif sc == -1:
                lbl, icon = "Bearish", "🔴"
            elif sc <= -2:
                lbl, icon = "Overpriced Target Penalty", "🚨"
            else:
                lbl, icon = "Neutral", "🟡"
                
            st.markdown(f"{icon} &nbsp;**{name}** — {lbl}")

    with right:
        st.markdown('<p class="section-header">Execution Risk Calculator Matrix</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.markdown(metric_card("Calculated Entry", f"${eff_entry:.2f}"), unsafe_allow_html=True)
        c2.markdown(metric_card("Stop Loss Threshold", f"${stop_loss:.2f}", f"−{risk_pct}%", "metric-delta-down"), unsafe_allow_html=True)
        c3.markdown(metric_card("Take Target (1:2)", f"${take_profit:.2f}", f"+{risk_pct*2}%", "metric-delta-up"), unsafe_allow_html=True)
        
        fig_t = go.Figure()
        ctx_df = df.tail(90)
        fig_t.add_trace(go.Scatter(x=ctx_df.index, y=ctx_df["Close"], name="Close Core", line=dict(color=ACCENT, width=2)))
        fig_t.add_trace(go.Scatter(x=ctx_df.index, y=ctx_df["MA30"], name="MA30 Node", line=dict(color=BLUE, width=1, dash="dash")))
        fig_t.add_trace(go.Scatter(x=ctx_df.index, y=ctx_df["BB_Upper"], name="BB Upper Band", line=dict(color=MUTED, width=0.8, dash="dot")))
        fig_t.add_trace(go.Scatter(x=ctx_df.index, y=ctx_df["BB_Lower"], name="BB Lower Band", line=dict(color=MUTED, width=0.8, dash="dot"), fill="tonexty", fillcolor="rgba(122,128,153,0.03)"))
        
        fig_t.add_hline(y=eff_entry, line_color=ACCENT, line_width=1.2, line_dash="solid", annotation_text="Calculated Entry Target", annotation_position="top left")
        fig_t.add_hline(y=take_profit, line_color=GREEN, line_width=1.2, line_dash="dash", annotation_text="Take Profit Threshold (1:2)", annotation_position="top left")
        fig_t.add_hline(y=stop_loss, line_color=RED, line_width=1.2, line_dash="dash", annotation_text="Risk Stop Loss Line", annotation_position="bottom left")

        fig_t.update_layout(**base_layout(290, "Trailing Trend Baseline Metrics Context Overlay", override_yaxis=dict(tickprefix="$")))
        st.plotly_chart(fig_t, width='stretch')

# ════════════════════════════════════════════════════════════
#  TAB 2 — FORECAST ENGINE (GAP-FREE CONTEXT WITH FIXED VIEW RANGE)
# ════════════════════════════════════════════════════════════
with tab2:
    if model is None:
        empty_state("🔮", "Paste model tracking link inside sidebar to initialize Forecast Matrix engines.")
    elif f_prices is None:
        empty_state("⛔", "Timeline core compilation fault.")
    else:
        st.info(f"📅 **Forecasting Starting Boundary Execution Node:** `{pd.Timestamp(chosen_start_date).strftime('%A, %d %B %Y')}`")
        
        f_end = safe_float(f_prices[-1])
        f_chg = ((f_end - current_price) / current_price * 100)

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("Anchor Close", f"${current_price:.2f}"), unsafe_allow_html=True)
        c2.markdown(metric_card("Horizon Output End", f"${f_end:.2f}", f"{f_chg:+.2f}%", "metric-delta-up" if f_chg>=0 else "metric-delta-down"), unsafe_allow_html=True)
        c3.markdown(metric_card("Horizon Ceiling Peak", f"${safe_float(f_prices.max()):.2f}"), unsafe_allow_html=True)
        c4.markdown(metric_card("Horizon Floor Trough", f"${safe_float(f_prices.min()):.2f}"), unsafe_allow_html=True)

        fig_fc = go.Figure()
        
        # Full historical line (needed for seamless pan/zoom)
        fig_fc.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], name="Database Historical Core", line=dict(color=ACCENT, width=2)))
        
        if b_dates is not None and len(b_dates) > 0:
            b_x = list(b_dates) + list(b_dates[::-1])
            b_y = list(b_upper) + list(b_lower[::-1])
            fig_fc.add_trace(go.Scatter(x=b_x, y=b_y, fill="toself", fillcolor="rgba(160,110,225,0.06)", line=dict(color="rgba(0,0,0,0)"), name="Bridge Confidence Variance"))
            fig_fc.add_trace(go.Scatter(x=b_dates, y=b_prices, name="Implicit Gap Forecast Bridge", line=dict(color=PURPLE, width=1.5, dash="dash")))

        fx = list(f_dates) + list(f_dates[::-1])
        fy = list(f_upper) + list(f_lower[::-1])
        fig_fc.add_trace(go.Scatter(x=fx, y=fy, fill="toself", fillcolor="rgba(91,141,238,0.12)", line=dict(color="rgba(0,0,0,0)"), name="Forecast Confidence Variance"))
        fig_fc.add_trace(go.Scatter(x=f_dates, y=f_prices, name="Target Horizon Output (Hybrid Model)", line=dict(color=BLUE, width=2.2, dash="dot"), mode="lines+markers"))

        # ── X-axis viewport: 2 months before start → 2 months after horizon end ──
        view_start = pd.Timestamp(chosen_start_date) - pd.DateOffset(months=2)
        view_end   = pd.Timestamp(f_dates[-1]) + pd.DateOffset(months=2)

        # ── Y-axis: tight range derived from data in the visible window only ─────
        # Collect all price values that fall within the x viewport
        hist_in_view = df.loc[
            (df.index >= view_start) & (df.index <= view_end), "Adj Close"
        ].dropna()
        all_visible_prices = list(hist_in_view.values) + list(f_prices)
        if b_prices is not None and len(b_prices) > 0:
            all_visible_prices += list(b_prices)
        # Include confidence band extremes so bands are never clipped
        all_visible_prices += list(f_lower) + list(f_upper)
        if b_lower is not None and len(b_lower) > 0:
            all_visible_prices += list(b_lower) + list(b_upper)

        all_visible_prices = [v for v in all_visible_prices if np.isfinite(v) and v > 0]
        if all_visible_prices:
            y_min   = min(all_visible_prices)
            y_max   = max(all_visible_prices)
            y_pad   = (y_max - y_min) * 0.08   # 8% padding above and below
            y_range = [y_min - y_pad, y_max + y_pad]
        else:
            y_range = None

        base_ly_params = base_layout(
            440,
            "Dynamic Continuity Multi-Step Simulation Chart Core",
            override_yaxis=dict(tickprefix="$", range=y_range) if y_range else dict(tickprefix="$"),
        )
        base_ly_params["xaxis"].update(dict(range=[view_start, view_end]))

        fig_fc.update_layout(**base_ly_params)
        st.plotly_chart(fig_fc, width='stretch')

# ════════════════════════════════════════════════════════════
#  TAB 3 — ADVANCED VISUALIZATIONS
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
    with fa: viz_start = st.date_input("Filter Range Start", value=df.index[-90].date(), min_value=df.index[0].date(), max_value=df_vis.index[-1].date(), key="v_start")
    with fb: viz_end   = st.date_input("Filter Range Stop", value=df_vis.index[-1].date(), min_value=df.index[0].date(), max_value=df_vis.index[-1].date(), key="v_end")

    if viz_start >= viz_end:
        st.warning("⚠️ Logic Exception: Start date must be before stop date.")
        st.stop()

    dv = df_vis.loc[str(viz_start):str(viz_end)].copy()

    r1a, r1b = st.columns([3, 2])
    with r1a:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=dv.index, y=dv["Close"], name="Unified Close (Concat)", line=dict(color=ACCENT, width=1.8)))
        for c, col_color in [("MA30", BLUE), ("MA90", PURPLE)]:
            if c in dv.columns and dv[c].notna().any():
                fig1.add_trace(go.Scatter(x=dv.index, y=dv[c], name=c, line=dict(color=col_color, width=1, dash="dot")))
        fig1.update_layout(**base_layout(340, "Continuous Close Pricing & Trailing Moving Averages", override_yaxis=dict(tickprefix="$")))
        st.plotly_chart(fig1, width='stretch')
    with r1b:
        v_cols = [GREEN if i==0 else (GREEN if dv["Close"].iloc[i]>=dv["Close"].iloc[i-1] else RED) for i in range(len(dv))]
        fig2 = go.Figure(go.Bar(x=dv.index, y=dv["Volume"], marker_color=v_cols, name="Volume Node"))
        fig2.update_layout(**base_layout(340, "Unified Timeline Volume Distribution Matrix"))
        st.plotly_chart(fig2, width='stretch')

    r2a, r2b = st.columns([3, 2])
    with r2a:
        fig3 = go.Figure(go.Candlestick(x=dv.index, open=dv["Open"], high=dv["High"], low=dv["Low"], close=dv["Close"], increasing_line_color=GREEN, decreasing_line_color=RED, name="OHLC Layer"))
        if dv["BB_Upper"].notna().any():
            fig3.add_trace(go.Scatter(x=dv.index, y=dv["BB_Upper"], name="Volatility Cap", line=dict(color=MUTED, width=0.8, dash="dash")))
            fig3.add_trace(go.Scatter(x=dv.index, y=dv["BB_Lower"], fill="tonexty", fillcolor="rgba(122,128,153,0.04)", name="Volatility Floor", line=dict(color=MUTED, width=0.8, dash="dash")))
        fig3.update_layout(**base_layout(340, "High-Frequency Structural Candlestick + Variance Channels", override_yaxis=dict(tickprefix="$")))
        fig3.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig3, width='stretch')
    with r2b:
        fig4 = go.Figure(go.Scatter(x=dv.index, y=dv["Spread"], fill="tozeroy", fillcolor="rgba(232,200,74,0.12)", line=dict(color=ACCENT, width=1.2)))
        fig4.update_layout(**base_layout(340, "Intraday Risk Dispersion (High − Low Volatility Variance)", override_yaxis=dict(tickprefix="$")))
        st.plotly_chart(fig4, width='stretch')

    r3a, r3b = st.columns(2)
    with r3a:
        if dv["MACD"].notna().any():
            fig5 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.05)
            fig5.add_trace(go.Scatter(x=dv.index, y=dv["MACD"], name="MACD Core", line=dict(color=BLUE, width=1.5)), row=1, col=1)
            fig5.add_trace(go.Scatter(x=dv.index, y=dv["MACDSig"], name="Signal Line", line=dict(color=ACCENT, width=1.5)), row=1, col=1)
            h_colors = [GREEN if val >= 0 else RED for val in dv["MACDHist"].fillna(0)]
            fig5.add_trace(go.Bar(x=dv.index, y=dv["MACDHist"], name="Histogram", marker_color=h_colors), row=2, col=1)
            fig5.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG, font_color=FONT_COL, height=340, margin=dict(l=0,r=0,t=30,b=0), title=dict(text="Momentum MACD Oscillator Structure (12, 26, 9)", font=dict(size=12, color=MUTED)), showlegend=False)
            fig5.update_xaxes(gridcolor=GRID_COL); fig5.update_yaxes(gridcolor=GRID_COL)
            st.plotly_chart(fig5, width='stretch')
        else: empty_state("📉", "MACD sequence arrays initializing...")
    with r3b:
        if dv["RSI"].notna().any():
            fig6 = go.Figure()
            fig6.add_trace(go.Scatter(x=dv.index, y=dv["RSI"], name="RSI", line=dict(color=PURPLE, width=1.5)))
            fig6.add_hrect(y0=70, y1=100, fillcolor="rgba(240,82,82,0.06)", line_width=0)
            fig6.add_hrect(y0=0,  y1=30,  fillcolor="rgba(38,212,124,0.06)", line_width=0)
            fig6.add_hline(y=70, line_color=RED, line_dash="dash", line_width=0.8)
            fig6.add_hline(y=30, line_color=GREEN, line_dash="dash", line_width=0.8)
            fig6.update_layout(**base_layout(340, "Relative Strength Velocity Index RSI (14)", override_yaxis=dict(range=[0, 100])))
            st.plotly_chart(fig6, width='stretch')
        else: empty_state("📉", "Velocity arrays initializing...")

    r4a, r4b = st.columns(2)
    with r4a:
        yearly = dv.groupby(dv.index.year)["Close"].mean().reset_index()
        fig7 = go.Figure(go.Bar(x=yearly.iloc[:, 0].astype(str), y=yearly["Close"], marker_color=ACCENT))
        fig7.update_layout(**base_layout(340, "Macro Annual Mean Close Allocation Values (Extended Horizon)", override_yaxis=dict(tickprefix="$")))
        st.plotly_chart(fig7, width='stretch')
    with r4b:
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        fig8 = go.Figure()
        for m_idx, m_name in enumerate(months, 1):
            sub = dv[dv.index.month == m_idx]["Close"].dropna()
            if not sub.empty: fig8.add_trace(go.Box(y=sub, name=m_name, marker_color=BLUE, line_color=BLUE, fillcolor="rgba(91,141,238,0.18)"))
        fig8.update_layout(**base_layout(340, "Seasonality Structural Distribution Matrices"), showlegend=False)
        st.plotly_chart(fig8, width='stretch')

    st.markdown('<p class="section-header">Cross-Sectional Attribute Correlation Matrix Heatmap</p>', unsafe_allow_html=True)
    corr_cols = [c for c in ["Open", "High", "Low", "Close", "Volume", "Spread"] if c in dv.columns]
    corr_data = dv[corr_cols].dropna()
    
    if len(corr_data) >= 5 and len(corr_cols) >= 2:
        c_mat = corr_data.corr().round(3)
        fig11 = go.Figure(go.Heatmap(z=c_mat.values, x=corr_cols, y=corr_cols, colorscale="RdBu", zmid=0, zmin=-1, zmax=1, text=c_mat.values, texttemplate="%{text:.2f}", showscale=True))
        fig11.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG, font_color=FONT_COL, height=360, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig11, width='stretch')
    else:
        empty_state("📊", "Attribute matrices lack sufficient spatial alignment dimensions.")
