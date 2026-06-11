# ============================================================
# TSLA FORECASTING HUB  |  app.py
# Model: CNN-GRU (Model III) | Data: TSLA.csv + yfinance API
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

# Define fallback for secret_url to ensure no NameError occurs
secret_url = ""
try:
    if "model_config" in st.secrets and "gdrive_model_link" in st.secrets["model_config"]:
        secret_url = st.secrets["model_config"]["gdrive_model_link"]
except Exception:
    pass

def get_secret_model_link() -> str:
    """Safely retrieves the model link from Streamlit secrets framework."""
    return secret_url

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="TSLA Forecast Hub",
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
    """Convert to float without raising; return fallback on NaN/None/inf."""
    try:
        v = float(val)
        return fallback if (np.isnan(v) or np.isinf(v)) else v
    except Exception:
        return fallback


def empty_state(icon: str, msg: str):
    """Render a clean empty-state card instead of blank space."""
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
    """
    Returns (df, warnings_list).
    Never raises — all errors are collected and surfaced to the UI.
    """
    warnings_out = []

    # ── Load from unified master dataset TSLA_1.csv ────────────
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

    # Require minimum usable columns
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing  = required - set(df.columns)
    if missing:
        return pd.DataFrame(), [f"Dataset is missing columns: {', '.join(missing)}"]

    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]

    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # ── Derived columns ───────────────────────────────────────
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

    # Bollinger Bands (20, 2σ)
    df["BB_Mid"]   = df["Close"].rolling(20).mean()
    bb_std         = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["BB_Mid"] + 2 * bb_std
    df["BB_Lower"] = df["BB_Mid"] - 2 * bb_std

    return df, warnings_out


def build_scaler(df: pd.DataFrame):
    """Fit MinMaxScaler on Adj Close — mirrors notebook exactly."""
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
    """Extract file ID from any common Google Drive URL format."""
    url = url.strip()
    for pattern in [
        r"/file/d/([a-zA-Z0-9_-]{20,})",
        r"[?&]id=([a-zA-Z0-9_-]{20,})",
        r"/d/([a-zA-Z0-9_-]{20,})/",
        r"^([a-zA-Z0-9_-]{20,})$",   # bare file ID
    ]:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    return None


@st.cache_resource(show_spinner=False)
def load_model_cached(file_id: str):
    """Download and load the .keras model. Cached by file_id."""
    try:
        import gdown
    except ImportError:
        raise RuntimeError("'gdown' is not installed. Add it to requirements.txt.")
    try:
        import tensorflow as tf
    except ImportError:
        raise RuntimeError("'tensorflow' is not installed. Add it to requirements.txt.")

    download_url = f"https://drive.google.com/uc?id={file_id}"
    tmp_path     = os.path.join(tempfile.gettempdir(), f"tsla_model_{file_id[:8]}.keras")

    if not os.path.exists(tmp_path):
        result = gdown.download(download_url, tmp_path, quiet=True)
        if result is None or not os.path.exists(tmp_path):
            raise RuntimeError("Download returned nothing. Check verification share parameters.")

    try:
        model = tf.keras.models.load_model(tmp_path)
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"Model file downloaded but could not be loaded: {e}")

    return model


def dynamic_timeline_forecasting(model, scaler, df: pd.DataFrame, start_date: pd.Timestamp, n_days: int) -> tuple:
    """
    Advanced multi-step timeline management.
    Handles actual data mapping up to June 10, 2026 with 100% confidence bands, 
    and bridges extended future horizons recursively via neural network loops.
    """
    db_max_date = df.index.max()
    target_start = pd.Timestamp(start_date)
    
    # Pre-generate chronological date range starting from user's chosen start point
    biz_dates = pd.bdate_range(start=target_start, periods=n_days)
    
    # Calculate rolling standard deviation for forecast volatility bounds
    recent_ret = df["DailyReturn"].replace([np.inf, -np.inf], np.nan).dropna().tail(60)
    daily_vol = (recent_ret.std() / 100) if len(recent_ret) >= 5 else 0.02
    
    preds_prices = []
    lower_bounds = []
    upper_bounds = []
    
    # ── CASE 1: Start date is inside our known historical matrix (<= June 10, 2026) ──
    if target_start <= db_max_date:
        # We need a continuous simulation block running from historical data up to the final forecast point
        sim_dates = pd.bdate_range(start=target_start, end=biz_dates[-1])
        
        # Pre-seed running list of scaled values for lookback context window
        history_slice = df[df.index < target_start].tail(LOOKBACK)
        scaled_seed = list(scaler.transform(history_slice[["Adj Close"]].values).flatten())
        
        sim_cache = {}
        forecast_step_counter = 1
        
        for curr_date in sim_dates:
            if curr_date <= db_max_date:
                # Use actual database data
                actual_val = float(df.loc[curr_date, "Adj Close"])
                sim_cache[curr_date] = (actual_val, actual_val, actual_val)
                # Feed the real value back into scaler tracking queue
                scaled_seed.append(float(scaler.transform([[actual_val]])[0,0]))
            else:
                # Missing day beyond June 10 database ceiling -> Recursive Forecast
                x = np.array(scaled_seed[-LOOKBACK:], dtype=np.float32).reshape(1, LOOKBACK, 1)
                out_scaled = float(model.predict(x, verbose=0)[0, 0])
                out_scaled = np.clip(out_scaled, 0.0, 1.0)
                
                pred_val = float(scaler.inverse_transform([[out_scaled]])[0,0])
                scaled_seed.append(out_scaled)
                
                # Apply compounding variance channel expansion
                band_frac = np.clip(daily_vol * np.sqrt(forecast_step_counter), 0, 0.5)
                sim_cache[curr_date] = (pred_val, pred_val * (1 - band_frac), pred_val * (1 + band_frac))
                forecast_step_counter += 1
                
        # Filter down specifically to match the n_days horizon requested
        for d in biz_dates:
            if d in sim_cache:
                p, lo, hi = sim_cache[d]
            else:
                p, lo, hi = sim_cache[db_max_date] # fallback safety
            preds_prices.append(p)
            lower_bounds.append(lo)
            upper_bounds.append(hi)
            
    # ── CASE 2: Start date is way into the future (years ahead gap) ──
    else:
        # Step A: Bridge the entire continuous gap from June 10, 2026 up to user's future start date
        bridge_dates = pd.bdate_range(start=db_max_date + pd.Timedelta(days=1), end=target_start - pd.Timedelta(days=1))
        
        history_slice = df.tail(LOOKBACK)
        scaled_seed = list(scaler.transform(history_slice[["Adj Close"]].values).flatten())
        
        # Run recursive tracking engine through the unmonitored baseline gap
        for _ in bridge_dates:
            x = np.array(scaled_seed[-LOOKBACK:], dtype=np.float32).reshape(1, LOOKBACK, 1)
            out_scaled = float(model.predict(x, verbose=0)[0, 0])
            out_scaled = np.clip(out_scaled, 0.0, 1.0)
            scaled_seed.append(out_scaled)
            
        # Step B: Generate forecast sequence from target_start onward
        for step in range(1, n_days + 1):
            x = np.array(scaled_seed[-LOOKBACK:], dtype=np.float32).reshape(1, LOOKBACK, 1)
            out_scaled = float(model.predict(x, verbose=0)[0, 0])
            out_scaled = np.clip(out_scaled, 0.0, 1.0)
            
            pred_val = float(scaler.inverse_transform([[out_scaled]])[0,0])
            scaled_seed.append(out_scaled)
            
            band_frac = np.clip(daily_vol * np.sqrt(step), 0, 0.5)
            preds_prices.append(pred_val)
            lower_bounds.append(pred_val * (1 - band_frac))
            upper_bounds.append(pred_val * (1 + band_frac))
            
    return biz_dates, np.array(preds_prices), np.array(lower_bounds), np.array(upper_bounds)


# ╔══════════════════════════════════════════════════════════╗
# ║                    LOAD DATA                             ║
# ╚══════════════════════════════════════════════════════════╗

with st.spinner("Loading TSLA data…"):
    df, data_warnings = load_data()

if df.empty:
    st.error("⛔ Dataset instantiation failure. Check TSLA_1.csv status matrix.")
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
    st.warning(f"⚠️ Scaler fault: {_scaler_err}")


# ╔══════════════════════════════════════════════════════════╗
# ║                    SIDEBAR                               ║
# ╚══════════════════════════════════════════════════════════╝

with st.sidebar:
    st.markdown("## ⚡ TSLA Hub")
    st.markdown("---")

    st.markdown('<p class="section-header">Model Engine Status</p>', unsafe_allow_html=True)
    gdrive_url = st.text_input(
        "Model Resource Node",
        value=secret_url,
        type="password",
        help="Pre-loaded securely from Streamlit Cloud Secrets.",
    )
    load_btn = st.button("⬇ Load Model", use_container_width=True, type="primary")

    model_status_slot = st.empty()
    if st.session_state.get("model_loaded", False):
        model_status_slot.success("🟢 Systems Operational (Model Live)")
    else:
        model_status_slot.info("⏳ Initializing Deep Learning Core...")

    st.markdown("---")
    st.markdown('<p class="section-header">Forecast Parameter Matrix</p>', unsafe_allow_html=True)
    
    # New selector option to set chosen start date anchor point
    chosen_start_date = st.date_input(
        "Forecast Execution Start Date",
        value=df.index[-1].date()
    )
    forecast_days = st.slider("Trading days ahead", 5, 60, 30, 5)

    st.markdown("---")
    st.markdown('<p class="section-header">Trade Parameters</p>', unsafe_allow_html=True)
    entry_price  = st.number_input("Your Entry Price ($)", min_value=0.0, value=0.0, step=0.01)
    position_qty = st.number_input("Position Size (shares)", min_value=1, value=10, step=1)
    risk_pct     = st.slider("Risk Tolerance (%)", 1, 20, 5)

    st.markdown("---")
    st.caption(f"Data Source: Live Yahoo Finance Wrapper | Node: Streamlit Cloud Engine")
  
if not st.session_state.get("model_loaded", False):
    url_to_load = gdrive_url.strip() if gdrive_url else secret_url.strip()
    if url_to_load:
        file_id = extract_gdrive_id(url_to_load)
        if file_id:
            try:
                _m = load_model_cached(file_id)
                st.session_state["model_obj"] = _m
                st.session_state["model_loaded"] = True
                model_status_slot.success("🟢 Systems Operational (Model Live)")
            except Exception as e:
                st.session_state["model_loaded"] = False
                st.sidebar.error(f"❌ Core Initialization Fault: {e}")

if load_btn:
    url_clean = gdrive_url.strip() if gdrive_url else ""
    if not url_clean:
        st.sidebar.error("❌ Please paste a Google Drive link first.")
    else:
        file_id = extract_gdrive_id(url_clean)
        if file_id is None:
            st.sidebar.error("❌ Invalid Google Drive URL tracking matrix.")
        else:
            with st.sidebar:
                with st.spinner("Downloading model…"):
                    try:
                        _m = load_model_cached(file_id)
                        st.session_state["model_obj"]    = _m
                        st.session_state["model_loaded"] = True
                        model_status_slot.success("✅ Model ready")
                    except Exception as _me:
                        st.session_state["model_loaded"] = False
                        st.error(f"❌ {_me}")

model = st.session_state.get("model_obj", None)
eff_entry = entry_price if entry_price > 0.0 else current_price


# ╔══════════════════════════════════════════════════════════╗
# ║             AUTOMATED BACKGROUND PIPELINE                ║
# ╚══════════════════════════════════════════════════════════╝

f_dates, f_prices, f_lower, f_upper = None, None, None, None
if model is not None and scaler_ok:
    try:
        f_dates, f_prices, f_lower, f_upper = dynamic_timeline_forecasting(
            model, scaler, df, pd.Timestamp(chosen_start_date), n_days=forecast_days
        )
    except Exception as _e:
        pass


# ╔══════════════════════════════════════════════════════════╗
# ║                    HEADER STRIP                          ║
# ╚══════════════════════════════════════════════════════════╝

col_t, col_p, col_d, col_v, col_sp = st.columns([3, 2, 2, 2, 2])

with col_t:
    st.markdown("### ⚡ Tesla, Inc. &nbsp;`TSLA`")
    st.caption(f"Last database sync marker: {df.index[-1].strftime('%d %b %Y')}")

with col_p:
    delta_cls = "metric-delta-up" if price_change >= 0 else "metric-delta-down"
    arrow     = "▲" if price_change >= 0 else "▼"
    st.markdown(
        metric_card(
            "Last Close", f"${current_price:.2f}",
            f"{arrow} ${abs(price_change):.2f} ({price_change_p:+.2f}%)", delta_cls,
        ),
        unsafe_allow_html=True,
    )

with col_d:
    hi_52 = safe_float(df["High"].tail(252).max())
    st.markdown(metric_card("52-Week High", f"${hi_52:.2f}"), unsafe_allow_html=True)

with col_v:
    lo_52 = safe_float(df["Low"].tail(252).min())
    st.markdown(metric_card("52-Week Low", f"${lo_52:.2f}"), unsafe_allow_html=True)

with col_sp:
    avg_vol = df["Volume"].tail(20).mean()
    vol_str = f"{avg_vol/1e6:.1f}M" if avg_vol >= 1e6 else f"{avg_vol/1e3:.0f}K"
    st.markdown(metric_card("Avg Volume (20d)", vol_str), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════╗
# ║                       TABS                              ║
# ╚══════════════════════════════════════════════════════════╝

tab1, tab2, tab3 = st.tabs(["📡  Trade Signals", "🔮  Forecast Engine", "📊  Advanced Visualizations"])


# ============================================================
#  TAB 1 — TRADE SIGNALS (FIXED LAYOUT CLASH)
# ============================================================
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
    risk_per_sh  = max(eff_entry - stop_loss, 0.01)
    take_profit  = eff_entry + risk_per_sh * 2.0
    max_loss     = risk_per_sh * position_qty
    max_gain     = (take_profit - eff_entry) * position_qty

    left, right = st.columns([1, 2], gap="large")
    with left:
        st.markdown('<p class="section-header">Composite Signal</p>', unsafe_allow_html=True)
        st.markdown(f'<div style="text-align:center;padding:28px 0 16px"><div class="{signal_css}">{signal_label}</div></div>', unsafe_allow_html=True)
        
        st.markdown('<p class="section-header">Indicator Breakdown</p>', unsafe_allow_html=True)
        for name, sc in tech_scores.items():
            lbl = "Bullish" if sc == 1 else ("Bearish" if sc == -1 else "Neutral")
            icon = "🟢" if sc == 1 else ("🔴" if sc == -1 else "🟡")
            st.markdown(f"{icon} &nbsp;**{name}** — {lbl}")

    with right:
        st.markdown('<p class="section-header">Risk / Reward Calculator</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.markdown(metric_card("Entry", f"${eff_entry:.2f}"), unsafe_allow_html=True)
        c2.markdown(metric_card("Stop Loss", f"${stop_loss:.2f}", f"−{risk_pct}%", "metric-delta-down"), unsafe_allow_html=True)
        c3.markdown(metric_card("Take Profit (1:2)", f"${take_profit:.2f}", f"+{risk_pct*2}%", "metric-delta-up"), unsafe_allow_html=True)
        
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=df.index[-90:], y=df["Close"].tail(90), name="Close", line=dict(color=ACCENT)))
        
        # FIX: Passed override_yaxis inside base_layout parameters directly to prevent duplication crash
        fig_t.update_layout(**base_layout(240, "Trailing Context Baseline Evaluation", override_yaxis=dict(tickprefix="$")))
        st.plotly_chart(fig_t, use_container_width=True)


# ============================================================
#  TAB 2 — FORECAST ENGINE (FIXED LAYOUT CLASH)
# ============================================================
with tab2:
    if model is None:
        empty_state("🔮", "Paste model tracking link inside sidebar to initialize Forecast Matrix engines.")
    elif f_prices is None:
        empty_state("⛔", "Timeline processing fault. Check parameters.")
    else:
        # Show exactly the user-selected date when forecasting starts
        st.info(f"📅 **Forecasting Starting Boundary Execution Node:** `{pd.Timestamp(chosen_start_date).strftime('%A, %d %B %Y')}`")
        
        f_end = safe_float(f_prices[-1])
        f_hi  = safe_float(f_prices.max())
        f_lo  = safe_float(f_prices.min())
        f_chg = ((f_end - current_price) / current_price * 100)

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("Anchor Price", f"${current_price:.2f}"), unsafe_allow_html=True)
        c2.markdown(metric_card(f"Horizon End Value", f"${f_end:.2f}", f"{f_chg:+.2f}%", "metric-delta-up" if f_chg>=0 else "metric-delta-down"), unsafe_allow_html=True)
        c3.markdown(metric_card("Horizon High Peak", f"${f_hi:.2f}"), unsafe_allow_html=True)
        c4.markdown(metric_card("Horizon Low Trough", f"${f_lo:.2f}"), unsafe_allow_html=True)

        fig_fc = go.Figure()
        # Historical Trace
        fig_fc.add_trace(go.Scatter(x=df.index[-120:], y=df["Adj Close"].tail(120), name="Historical", line=dict(color=ACCENT, width=2)))
        
        # Confidence Band Shading Container
        bx = list(f_dates) + list(f_dates[::-1])
        by = list(f_upper) + list(f_lower[::-1])
        fig_fc.add_trace(go.Scatter(x=bx, y=by, fill="toself", fillcolor="rgba(91,141,238,0.12)", line=dict(color="rgba(0,0,0,0)"), name="Confidence Interval"))
        
        # Forecast Model Plots
        fig_fc.add_trace(go.Scatter(x=f_dates, y=f_prices, name="Forecast Window Matrix", line=dict(color=BLUE, width=2, dash="dot"), mode="lines+markers"))
        
        # FIX: Passed override_yaxis inside base_layout parameters directly to prevent duplication crash
        fig_fc.update_layout(**base_layout(440, "Dynamic Multi-Step Simulation Curve Plots", override_yaxis=dict(tickprefix="$")))
        st.plotly_chart(fig_fc, use_container_width=True)

        with st.expander("📋 View full forecast table"):
            fc_rows = [{
                "Day": i+1, "Date": d.strftime("%d %b %Y"), "Forecast ($)": f"${p:.2f}",
                "Lower Bound ($)": f"${lo:.2f}", "Upper Bound ($)": f"${hi:.2f}"
            } for i, (d, p, lo, hi) in enumerate(zip(f_dates, f_prices, f_lower, f_upper))]
            st.dataframe(pd.DataFrame(fc_rows), use_container_width=True, hide_index=True)


# ============================================================
#  TAB 3 — ADVANCED VISUALIZATIONS (REVISED COMPLETELY)
# ============================================================
with tab3:
    # 1. Build complete combined historical + future projection dataset matrix
    df_vis = df.copy()
    if f_prices is not None:
        df_future = pd.DataFrame(index=f_dates)
        df_future["Close"] = f_prices
        df_future["Open"] = f_prices
        df_future["High"] = f_upper
        df_future["Low"] = f_lower
        df_future["Volume"] = df["Volume"].iloc[-1]  # project forward last known volume node
        df_future["Spread"] = df_future["High"] - df_future["Low"]
        df_future["BB_Upper"] = f_upper
        df_future["BB_Lower"] = f_lower
        df_future["MA30"] = np.nan
        df_future["MA90"] = np.nan
        df_future["MA200"] = np.nan
        df_future["MACD"] = np.nan
        df_future["MACDSig"] = np.nan
        df_future["MACDHist"] = np.nan
        df_future["RSI"] = np.nan
        df_future["DailyReturn"] = df_future["Close"].pct_change() * 100
        
        df_vis = pd.concat([df_vis, df_future])
        df_vis = df_vis[~df_vis.index.duplicated(keep='first')]
        df_vis.sort_index(inplace=True)

    # 2. Horizon Selection Interfaces
    fa, fb = st.columns(2)
    with fa:
        viz_start = st.date_input(
            "From Range Marker", 
            value=df.index[-90].date(), 
            min_value=df.index[0].date(), 
            max_value=df_vis.index[-1].date(), 
            key="v_start"
        )
    with fb:
        viz_end = st.date_input(
            "To Range Marker (Includes Forecast Horizon)", 
            value=df_vis.index[-1].date(), 
            min_value=df.index[0].date(), 
            max_value=df_vis.index[-1].date(), 
            key="v_end"
        )

    if viz_start >= viz_end:
        st.warning("⚠️ Timeline validation error: Start index must be before stop marker.")
        st.stop()

    # Create filtered layout array slice
    dv = df_vis.loc[str(viz_start):str(viz_end)].copy()

    # 📊 Row 1: Unified Pricing Lines + Volumes
    r1a, r1b = st.columns([3, 2])
    with r1a:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=dv.index, y=dv["Close"], 
            name="Unified Close (History + Forecast)", 
            line=dict(color=ACCENT, width=1.8)
        ))
        for col, col_color in [("MA30", BLUE), ("MA90", PURPLE), ("MA200", RED)]:
            if col in dv.columns and dv[col].notna().any():
                fig1.add_trace(go.Scatter(
                    x=dv.index, y=dv[col], 
                    name=col, 
                    line=dict(color=col_color, width=1, dash="dot")
                ))
        fig1.update_layout(**base_layout(360, "Systemic Pricing Vectors & Forecast Concat", override_yaxis=dict(tickprefix="$")))
        st.plotly_chart(fig1, use_container_width=True)

    with r1b:
        vol_colors = [GREEN if idx == 0 else (GREEN if dv["Close"].iloc[idx] >= dv["Close"].iloc[idx-1] else RED) for idx in range(len(dv))]
        fig2 = go.Figure(go.Bar(x=dv.index, y=dv["Volume"], marker_color=vol_colors, name="Volume Node"))
        fig2.update_layout(**base_layout(360, "Volume Profile Distribution Metrics"))
        st.plotly_chart(fig2, use_container_width=True)

    # 📊 Row 2: Structural Candlesticks + Intraday High/Low Dispersion Spread
    r2a, r2b = st.columns([3, 2])
    with r2a:
        fig3 = go.Figure(go.Candlestick(
            x=dv.index, open=dv["Open"], high=dv["High"], 
            low=dv["Low"], close=dv["Close"], 
            increasing_line_color=GREEN, decreasing_line_color=RED, 
            name="OHLC Layer"
        ))
        # FIX: Explicitly updated config variables safely passed as an additional update dictionary call
        fig3.update_layout(**base_layout(360, "High-Frequency Structural Candlestick + Variance Channels (Forecast Extended)", override_yaxis=dict(tickprefix="$")))
        fig3.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig3, use_container_width=True)

    with r2b:
        fig4 = go.Figure(go.Scatter(
            x=dv.index, y=dv["Spread"], 
            fill="tozeroy", fillcolor="rgba(232,200,74,0.12)", 
            line=dict(color=ACCENT, width=1.2)
        ))
        fig4.update_layout(**base_layout(360, "Intraday Risk Dispersion (High − Low Variance Channel)", override_yaxis=dict(tickprefix="$")))
        st.plotly_chart(fig4, use_container_width=True)

    # 📊 Row 3: Macro Annual Allocation Bars + Boxplots Seasonality Matrix
    r4a, r4b = st.columns(2)
    with r4a:
        yearly = dv.groupby(dv.index.year)["Close"].mean().reset_index()
        fig7 = go.Figure(go.Bar(x=yearly.iloc[:, 0].astype(str), y=yearly["Close"], marker_color=ACCENT))
        fig7.update_layout(**base_layout(360, "Macro Annual Mean Close Allocation Values (Including Forecast Years)", override_yaxis=dict(tickprefix="$")))
        st.plotly_chart(fig7, use_container_width=True)

    with r4b:
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        fig8 = go.Figure()
        for m_idx, m_name in enumerate(months, 1):
            subset = dv[dv.index.month == m_idx]["Close"].dropna()
            if not subset.empty:
                fig8.add_trace(go.Box(
                    y=subset, name=m_name, 
                    marker_color=BLUE, line_color=BLUE, 
                    fillcolor="rgba(91,141,238,0.18)"
                ))
        fig8.update_layout(**base_layout(360, "Seasonality Structural Distribution Matrices"), showlegend=False)
        st.plotly_chart(fig8, use_container_width=True)
