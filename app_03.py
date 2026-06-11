# ============================================================
# TSLA FORECASTING HUB  |  app.py
# Model: Hybrid CNN-GRU + Statistical Trend Alignment
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

# Global Safe Secret Declaration Engine
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

# ── Theme / CSS ───────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0d0f14; color: #e8eaf0; }
  [data-testid="stSidebar"] { background-color: #111318; border-right: 1px solid #1e2130; }

  .stTabs [data-baseweb="tab-list"] {
      gap: 0px; background: #111318; border-bottom: 2px solid #1e2130;
  }
  .stTabs [data-baseweb="tab"] {
      background: transparent; color: #7a8099; font-weight: 600;
      font-size: 0.85rem; letter-spacing: 0.06em; text-transform: uppercase;
      padding: 14px 28px; border: none;
      border-bottom: 2px solid transparent; margin-bottom: -2px;
  }
  .stTabs [aria-selected="true"] {
      color: #e8c84a !important;
      border-bottom: 2px solid #e8c84a !important;
      background: transparent !important;
  }
  .metric-card {
      background: #151820; border: 1px solid #1e2130;
      border-radius: 10px; padding: 18px 22px; text-align: center;
      min-height: 90px;
  }
  .metric-label {
      color: #7a8099; font-size: 0.72rem; letter-spacing: 0.08em;
      text-transform: uppercase; margin-bottom: 6px;
  }
  .metric-value {
      color: #e8eaf0; font-size: 1.5rem; font-weight: 700;
      font-family: 'Courier New', monospace; word-break: break-all;
  }
  .metric-delta-up   { color: #26d47c; font-size: 0.82rem; margin-top: 4px; }
  .metric-delta-down { color: #f05252; font-size: 0.82rem; margin-top: 4px; }
  .metric-muted      { color: #7a8099; font-size: 0.82rem; margin-top: 4px; }

  .signal-buy  { background:#0d3326; color:#26d47c; border:1px solid #26d47c; border-radius:6px; padding:10px 32px; font-weight:700; font-size:1.8rem; display:inline-block; letter-spacing:0.08em; }
  .signal-sell { background:#350d0d; color:#f05252; border:1px solid #f05252; border-radius:6px; padding:10px 32px; font-weight:700; font-size:1.8rem; display:inline-block; letter-spacing:0.08em; }
  .signal-hold { background:#2a2a10; color:#e8c84a; border:1px solid #e8c84a; border-radius:6px; padding:10px 32px; font-weight:700; font-size:1.8rem; display:inline-block; letter-spacing:0.08em; }

  .section-header {
      color:#e8c84a; font-size:0.68rem; letter-spacing:0.14em;
      text-transform:uppercase; margin-bottom:10px; margin-top:22px; opacity:0.75;
  }
  .empty-state {
      background:#151820; border:1px dashed #2a2f42; border-radius:10px;
      padding:36px 24px; text-align:center; color:#7a8099;
  }
  .empty-state-icon { font-size:2rem; margin-bottom:10px; }
  .empty-state-msg  { font-size:0.9rem; line-height:1.6; }

  hr { border-color: #1e2130; }
  #MainMenu, footer { visibility: hidden; }

  .stTextInput input, .stNumberInput input {
      background: #151820 !important; border: 1px solid #2a2f42 !important;
      color: #e8eaf0 !important; border-radius: 6px !important;
  }
  .stSelectbox > div > div {
      background: #151820 !important; border: 1px solid #2a2f42 !important;
  }
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

    # Derived technical columns
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

    # RSI (14)
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
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

def dynamic_timeline_forecasting(model, scaler, df: pd.DataFrame, start_date: pd.Timestamp, n_days: int) -> tuple:
    """
    Advanced Hybrid Strategy Engine.
    Combines deep learning models with SARIMAX seasonality filters to prevent decay flatlining.
    """
    # Defensive Runtime Import Injection to isolate failures if environment is misconfigured
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        raise RuntimeError("The 'statsmodels' library is missing from your deployment runtime. Please add statsmodels to requirements.txt")

    today_marker = pd.Timestamp.now().normalize()
    db_max_date = df.index.max()
    target_start = pd.Timestamp(start_date)
    
    biz_dates = pd.bdate_range(start=target_start, periods=n_days)
    recent_ret = df["DailyReturn"].replace([np.inf, -np.inf], np.nan).dropna().tail(60)
    daily_vol = (recent_ret.std() / 100) if len(recent_ret) >= 5 else 0.02
    
    preds_prices, lower_bounds, upper_bounds = [], [], []
    bridge_dates, bridge_prices, bridge_lo, bridge_hi = [], [], [], []
    
    # ─── STRATEGY 1: START DATE IS BEFORE TODAY ───
    if target_start <= today_marker:
        june_10_marker = pd.Timestamp("2026-06-10")
        if june_10_marker > db_max_date:
            june_10_marker = db_max_date
            
        one_year_ago = june_10_marker - pd.Timedelta(days=365)
        hist_1y = df.loc[one_year_ago:june_10_marker, "Adj Close"].resample('B').ffill().dropna()
        
        try:
            sarimax_core = SARIMAX(hist_1y.values, order=(1,1,1), seasonal_order=(1,0,0,5))
            sarimax_res = sarimax_core.fit(disp=False)
            seasonality_deltas = sarimax_res.forecast(steps=n_days) - hist_1y.iloc[-1]
        except Exception:
            seasonality_deltas = np.zeros(n_days)

        for idx, curr_date in enumerate(biz_dates):
            if curr_date <= db_max_date:
                hist_match = df.loc[:curr_date]
                actual_val = float(hist_match["Adj Close"].iloc[-1]) if not hist_match.empty else float(df["Adj Close"].iloc[0])
                preds_prices.append(actual_val)
            else:
                lookback_slice = df[df.index < curr_date].tail(LOOKBACK)
                scaled_seed = list(scaler.transform(lookback_slice[["Adj Close"]].values).flatten())
                
                for step_idx in range(max(1, idx)):
                    if biz_dates[step_idx] > db_max_date and biz_dates[step_idx] < curr_date:
                        scaled_val = scaler.transform([[preds_prices[step_idx]]])[0, 0]
                        scaled_seed.append(scaled_val)
                        
                x = np.array(scaled_seed[-LOOKBACK:], dtype=np.float32).reshape(1, LOOKBACK, 1)
                out_scaled = np.clip(float(model.predict(x, verbose=0)[0, 0]), 0.0, 1.0)
                pred_val = float(scaler.inverse_transform([[out_scaled]])[0,0])
                preds_prices.append(pred_val)

        preds_prices = np.array(preds_prices, dtype=np.float32)
        for idx in range(len(preds_prices)):
            if idx < len(seasonality_deltas):
                preds_prices[idx] += float(seasonality_deltas[idx] * 0.45)
            
            band_frac = np.clip(daily_vol * np.sqrt(idx + 1), 0, 0.45)
            lower_bounds.append(preds_prices[idx] * (1 - band_frac))
            upper_bounds.append(preds_prices[idx] * (1 + band_frac))

    # ─── STRATEGY 2: START DATE IS AFTER TODAY ───
    else:
        one_year_before_today = today_marker - pd.Timedelta(days=365)
        hist_1y = df.loc[one_year_before_today:min(today_marker, db_max_date), "Adj Close"].resample('B').ffill().dropna()
        
        try:
            sarimax_core = SARIMAX(hist_1y.values, order=(1,1,1), seasonal_order=(1,0,0,5))
            sarimax_res = sarimax_core.fit(disp=False)
            seasonality_deltas = sarimax_res.forecast(steps=n_days + 30) - hist_1y.iloc[-1]
        except Exception:
            seasonality_deltas = np.zeros(n_days + 30)

        gap_range = pd.bdate_range(start=db_max_date + pd.Timedelta(days=1), end=target_start - pd.Timedelta(days=1))
        history_slice = df.tail(LOOKBACK)
        scaled_seed = list(scaler.transform(history_slice[["Adj Close"]].values).flatten())
        
        bridge_step = 1
        for g_date in gap_range:
            x = np.array(scaled_seed[-LOOKBACK:], dtype=np.float32).reshape(1, LOOKBACK, 1)
            out_scaled = np.clip(float(model.predict(x, verbose=0)[0, 0]), 0.0, 1.0)
            pred_val = float(scaler.inverse_transform([[out_scaled]])[0,0])
            scaled_seed.append(out_scaled)
            
            if bridge_step < len(seasonality_deltas):
                pred_val += float(seasonality_deltas[bridge_step] * 0.45)
                
            band_frac = np.clip(daily_vol * np.sqrt(bridge_step), 0, 0.45)
            bridge_dates.append(g_date)
            bridge_prices.append(pred_val)
            bridge_lo.append(pred_val * (1 - band_frac))
            bridge_hi.append(pred_val * (1 + band_frac))
            bridge_step += 1
            
        for step in range(1, n_days + 1):
            x = np.array(scaled_seed[-LOOKBACK:], dtype=np.float32).reshape(1, LOOKBACK, 1)
            out_scaled = np.clip(float(model.predict(x, verbose=0)[0, 0]), 0.0, 1.0)
            pred_val = float(scaler.inverse_transform([[out_scaled]])[0,0])
            scaled_seed.append(out_scaled)
            
            s_idx = bridge_step + step
            if s_idx < len(seasonality_deltas):
                pred_val += float(seasonality_deltas[s_idx] * 0.45)
                
            band_frac = np.clip(daily_vol * np.sqrt(bridge_step), 0, 0.45)
            preds_prices.append(pred_val)
            lower_bounds.append(pred_val * (1 - band_frac))
            upper_bounds.append(pred_val * (1 + band_frac))
            bridge_step += 1
            
    return (biz_dates, np.array(preds_prices), np.array(lower_bounds), np.array(upper_bounds),
            pd.DatetimeIndex(bridge_dates), np.array(bridge_prices), np.array(bridge_lo), np.array(bridge_hi))

# ╔══════════════════════════════════════════════════════════╗
# ║                    LOAD INITIALIZER                      ║
# ╚══════════════════════════════════════════════════════════╗

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
    load_btn = st.button("⬇ Load Model Core", use_container_width=True, type="primary")

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

    tech_scores = {
        "RSI (14)": 1 if rsi_now < 30 else (-1 if rsi_now > 70 else 0),
        "MACD vs Signal": 1 if macd_now > sig_now else -1,
        "MA30 vs MA90": 1 if ma30_now > ma90_now else -1,
        "Bollinger Band": 1 if current_price < bb_lo else (-1 if current_price > bb_up else 0)
    }

    if f_prices is not None:
        model_target = safe_float(f_prices[min(4, len(f_prices)-1)])
        model_pct = (model_target - current_price) / current_price * 100
        tech_scores["Model (5-day)"] = 1 if model_pct > 1.5 else (-1 if model_pct < -1.5 else 0)

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
            lbl = "Bullish" if sc == 1 else ("Bearish" if sc == -1 else "Neutral")
            icon = "🟢" if sc == 1 else ("🔴" if sc == -1 else "🟡")
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
        st.plotly_chart(fig_t, use_container_width=True)

# ════════════════════════════════════════════════════════════
#  TAB 2 — FORECAST ENGINE
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
        fig_fc.add_trace(go.Scatter(x=df.index[-90:], y=df["Adj Close"].tail(90), name="Database Historical Core", line=dict(color=ACCENT, width=2)))
        
        if b_dates is not None and len(b_dates) > 0:
            b_x = list(b_dates) + list(b_dates[::-1])
            b_y = list(b_upper) + list(b_lower[::-1])
            fig_fc.add_trace(go.Scatter(x=b_x, y=b_y, fill="toself", fillcolor="rgba(160,110,225,0.06)", line=dict(color="rgba(0,0,0,0)"), name="Bridge Confidence Variance"))
            fig_fc.add_trace(go.Scatter(x=b_dates, y=b_prices, name="Implicit Gap Forecast Bridge", line=dict(color=PURPLE, width=1.5, dash="dash")))

        fx = list(f_dates) + list(f_dates[::-1])
        fy = list(f_upper) + list(f_lower[::-1])
        fig_fc.add_trace(go.Scatter(x=fx, y=fy, fill="toself", fillcolor="rgba(91,141,238,0.12)", line=dict(color="rgba(0,0,0,0)"), name="Forecast Confidence Variance"))
        fig_fc.add_trace(go.Scatter(x=f_dates, y=f_prices, name="Target Horizon Output (Hybrid Model)", line=dict(color=BLUE, width=2.2, dash="dot"), mode="lines+markers"))
        
        fig_fc.update_layout(**base_layout(440, "Dynamic Continuity Multi-Step Simulation Chart Core", override_yaxis=dict(tickprefix="$")))
        st.plotly_chart(fig_fc, use_container_width=True)

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
        st.plotly_chart(fig1, use_container_width=True)
    with r1b:
        v_cols = [GREEN if i==0 else (GREEN if dv["Close"].iloc[i]>=dv["Close"].iloc[i-1] else RED) for i in range(len(dv))]
        fig2 = go.Figure(go.Bar(x=dv.index, y=dv["Volume"], marker_color=v_cols, name="Volume Node"))
        fig2.update_layout(**base_layout(340, "Unified Timeline Volume Distribution Matrix"))
        st.plotly_chart(fig2, use_container_width=True)

    r2a, r2b = st.columns([3, 2])
    with r2a:
        fig3 = go.Figure(go.Candlestick(x=dv.index, open=dv["Open"], high=dv["High"], low=dv["Low"], close=dv["Close"], increasing_line_color=GREEN, decreasing_line_color=RED, name="OHLC Layer"))
        if dv["BB_Upper"].notna().any():
            fig3.add_trace(go.Scatter(x=dv.index, y=dv["BB_Upper"], name="Volatility Cap", line=dict(color=MUTED, width=0.8, dash="dash")))
            fig3.add_trace(go.Scatter(x=dv.index, y=dv["BB_Lower"], fill="tonexty", fillcolor="rgba(122,128,153,0.04)", name="Volatility Floor", line=dict(color=MUTED, width=0.8, dash="dash")))
        fig3.update_layout(**base_layout(340, "High-Frequency Structural Candlestick + Variance Channels", override_yaxis=dict(tickprefix="$")))
        fig3.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig3, use_container_width=True)
    with r2b:
        fig4 = go.Figure(go.Scatter(x=dv.index, y=dv["Spread"], fill="tozeroy", fillcolor="rgba(232,200,74,0.12)", line=dict(color=ACCENT, width=1.2)))
        fig4.update_layout(**base_layout(340, "Intraday Risk Dispersion (High − Low Volatility Variance)", override_yaxis=dict(tickprefix="$")))
        st.plotly_chart(fig4, use_container_width=True)

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
            st.plotly_chart(fig5, use_container_width=True)
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
            st.plotly_chart(fig6, use_container_width=True)
        else: empty_state("📉", "Velocity arrays initializing...")

    r4a, r4b = st.columns(2)
    with r4a:
        yearly = dv.groupby(dv.index.year)["Close"].mean().reset_index()
        fig7 = go.Figure(go.Bar(x=yearly.iloc[:, 0].astype(str), y=yearly["Close"], marker_color=ACCENT))
        fig7.update_layout(**base_layout(340, "Macro Annual Mean Close Allocation Values (Extended Horizon)", override_yaxis=dict(tickprefix="$")))
        st.plotly_chart(fig7, use_container_width=True)
    with r4b:
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        fig8 = go.Figure()
        for m_idx, m_name in enumerate(months, 1):
            sub = dv[dv.index.month == m_idx]["Close"].dropna()
            if not sub.empty: fig8.add_trace(go.Box(y=sub, name=m_name, marker_color=BLUE, line_color=BLUE, fillcolor="rgba(91,141,238,0.18)"))
        fig8.update_layout(**base_layout(340, "Seasonality Structural Distribution Matrices"), showlegend=False)
        st.plotly_chart(fig8, use_container_width=True)

    st.markdown('<p class="section-header">Cross-Sectional Attribute Correlation Matrix Heatmap</p>', unsafe_allow_html=True)
    corr_cols = [c for c in ["Open", "High", "Low", "Close", "Volume", "Spread"] if c in dv.columns]
    corr_data = dv[corr_cols].dropna()
    
    if len(corr_data) >= 5 and len(corr_cols) >= 2:
        c_mat = corr_data.corr().round(3)
        fig11 = go.Figure(go.Heatmap(z=c_mat.values, x=corr_cols, y=corr_cols, colorscale="RdBu", zmid=0, zmin=-1, zmax=1, text=c_mat.values, texttemplate="%{text:.2f}", showscale=True))
        fig11.update_layout(paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG, font_color=FONT_COL, height=360, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig11, use_container_width=True)
    else:
        empty_state("📊", "Attribute matrices lack sufficient spatial alignment dimensions.")
