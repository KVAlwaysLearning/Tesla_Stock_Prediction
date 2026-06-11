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

def get_secret_model_link() -> str:
    """Safely retrieves the model link from Streamlit secrets framework."""
    try:
        if "model_config" in st.secrets and "gdrive_model_link" in st.secrets["model_config"]:
            return st.secrets["model_config"]["gdrive_model_link"]
    except Exception:
        pass
    return ""

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

CSV_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TSLA.csv")
API_START = "2020-02-03"
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


def base_layout(height: int = 350, title: str = "") -> dict:
    return dict(
        paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font_color=FONT_COL, height=height,
        margin=dict(l=0, r=0, t=36, b=0),
        title=dict(text=title, font=dict(size=13, color=MUTED)),
        xaxis=dict(gridcolor=GRID_COL, showgrid=True),
        yaxis=dict(gridcolor=GRID_COL, showgrid=True),
    )


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

    # ── Part 1: CSV ───────────────────────────────────────────
    df_csv = pd.DataFrame()
    if os.path.exists(CSV_PATH):
        try:
            df_csv = pd.read_csv(CSV_PATH, parse_dates=["Date"])
            df_csv.set_index("Date", inplace=True)
            df_csv.sort_index(inplace=True)
            df_csv = df_csv[df_csv.index < pd.Timestamp(API_START)]
            if df_csv.empty:
                warnings_out.append("TSLA.csv contained no rows before 2020-02-03.")
        except Exception as e:
            warnings_out.append(f"Could not read TSLA.csv: {e}")
    else:
        warnings_out.append("TSLA.csv not found in the app folder — using live data only.")

    # ── Part 2: yfinance API ──────────────────────────────────
    df_api = pd.DataFrame()
    try:
        import yfinance as yf
        raw = yf.download("TSLA", start=API_START, progress=False, auto_adjust=False)
        if not raw.empty:
            raw.index.name = "Date"
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [c[0] for c in raw.columns]
            df_api = raw
        else:
            warnings_out.append("yfinance returned empty data for TSLA.")
    except ImportError:
        warnings_out.append("yfinance not installed — live data unavailable.")
    except Exception as e:
        warnings_out.append(f"Live data fetch failed: {e}")

    # ── Part 3: Merge ─────────────────────────────────────────
    frames = [f for f in [df_csv, df_api] if not f.empty]
    if not frames:
        return pd.DataFrame(), ["No data could be loaded. Check TSLA.csv and internet access."]

    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)

    # Require minimum usable columns
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing  = required - set(df.columns)
    if missing:
        return pd.DataFrame(), [f"Dataset is missing columns: {', '.join(missing)}"]

    # Ensure Adj Close exists (fall back to Close if absent)
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]
        warnings_out.append("'Adj Close' not found — using 'Close' as substitute.")

    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # Need at least LOOKBACK + 200 rows for all indicators to be meaningful
    if len(df) < LOOKBACK + 200:
        warnings_out.append(
            f"Only {len(df)} rows of data — some indicators may show as N/A."
        )

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

    # Only re-download if file doesn't exist yet (avoids repeated downloads on re-runs)
    if not os.path.exists(tmp_path):
        result = gdown.download(download_url, tmp_path, quiet=True, fuzzy=True)
        if result is None or not os.path.exists(tmp_path):
            raise RuntimeError(
                "Download returned nothing. Ensure the file is shared as "
                "'Anyone with the link can view' and the link is correct."
            )

    try:
        model = tf.keras.models.load_model(tmp_path)
    except Exception as e:
        # Remove corrupted file so next attempt re-downloads
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"Model file downloaded but could not be loaded: {e}")

    return model


def recursive_forecast(model, scaler, df: pd.DataFrame, n_days: int) -> tuple:
    """
    Recursive multi-step forecast seeded from the last LOOKBACK days.
    Returns (dates, predicted_prices, lower_band, upper_band).
    Raises ValueError with a clear message on any failure.
    """
    adj_col = df[["Adj Close"]].dropna()
    if len(adj_col) < LOOKBACK:
        raise ValueError(
            f"Need at least {LOOKBACK} rows of valid 'Adj Close' data to forecast. "
            f"Only {len(adj_col)} available."
        )

    scaled = scaler.transform(adj_col.values)
    seed   = list(scaled[-LOOKBACK:, 0])

    preds_scaled = []
    for _ in range(n_days):
        x   = np.array(seed[-LOOKBACK:], dtype=np.float32).reshape(1, LOOKBACK, 1)
        out = float(model.predict(x, verbose=0)[0, 0])
        # Clip to [0, 1] — scaled predictions should never escape this range
        out = np.clip(out, 0.0, 1.0)
        preds_scaled.append(out)
        seed.append(out)

    preds_prices = scaler.inverse_transform(
        np.array(preds_scaled, dtype=np.float32).reshape(-1, 1)
    ).flatten()

    # Confidence band: proportional to historical 60-day rolling volatility × √t
    recent_ret  = df["DailyReturn"].replace([np.inf, -np.inf], np.nan).dropna().tail(60)
    daily_vol   = (recent_ret.std() / 100) if len(recent_ret) >= 5 else 0.02
    t_arr       = np.arange(1, n_days + 1)
    band_frac   = np.clip(daily_vol * np.sqrt(t_arr), 0, 0.5)  # cap at 50%
    lower       = preds_prices * (1 - band_frac)
    upper       = preds_prices * (1 + band_frac)

    last_date  = df.index[-1]
    biz_dates  = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_days)

    return biz_dates, preds_prices, lower, upper


# ╔══════════════════════════════════════════════════════════╗
# ║                    LOAD DATA                             ║
# ╚══════════════════════════════════════════════════════════╝

with st.spinner("Loading TSLA data…"):
    df, data_warnings = load_data()

# Fatal: no data at all
if df.empty:
    st.error("⛔ " + (data_warnings[0] if data_warnings else "Unknown data error."))
    st.stop()

# Non-fatal warnings (yfinance offline, missing Adj Close, etc.)
for w in data_warnings:
    st.warning(f"⚠️ {w}")

# Safe price helpers — guard against < 2 rows
if len(df) >= 2:
    current_price  = safe_float(df["Close"].iloc[-1])
    prev_price     = safe_float(df["Close"].iloc[-2], fallback=current_price)
else:
    current_price  = safe_float(df["Close"].iloc[-1])
    prev_price     = current_price

price_change   = current_price - prev_price
price_change_p = (price_change / prev_price * 100) if prev_price != 0 else 0.0

# Build scaler once
try:
    scaler = build_scaler(df)
    scaler_ok = True
except Exception as _scaler_err:
    scaler_ok = False
    st.warning(f"⚠️ Scaler could not be built: {_scaler_err}. Forecast will be unavailable.")


# ╔══════════════════════════════════════════════════════════╗
# ║                    SIDEBAR                               ║
# ╚══════════════════════════════════════════════════════════╝

with st.sidebar:
    st.markdown("## ⚡ TSLA Hub")
    st.markdown("---")

    st.markdown('<p class="section-header">Model Engine Status</p>', unsafe_allow_html=True)
    # Fallback/Override Input: Populated with secret by default so it's hidden/clean
    gdrive_url = st.text_input(
        "Model Resource Node",
        value=secret_url,
        type="password",  # Masks your file ID/link from public viewers
        help="Pre-loaded securely from Streamlit Cloud Secrets. Change only for manual model overrides.",
    )
    load_btn = st.button("⬇ Load Model", use_container_width=True, type="primary")

    # Model status indicator
    # Persistent status layout slot
    model_status_slot = st.empty()
    if st.session_state.get("model_loaded", False):
        model_status_slot.success("🟢 Systems Operational (Model Live)")
    else:
        model_status_slot.info("⏳ Initializing Deep Learning Core...")

    st.markdown("---")
    st.markdown('<p class="section-header">Forecast Horizon</p>', unsafe_allow_html=True)
    forecast_days = st.slider("Trading days ahead", 5, 60, 30, 5)

    st.markdown("---")
    st.markdown('<p class="section-header">Trade Parameters</p>', unsafe_allow_html=True)
    entry_price  = st.number_input("Your Entry Price ($)", min_value=0.0, value=0.0, step=0.01)
    position_qty = st.number_input("Position Size (shares)", min_value=1, value=10, step=1)
    risk_pct     = st.slider("Risk Tolerance (%)", 1, 20, 5)

    st.markdown("---")
    st.caption(f"Data Source: Live Yahoo Finance Wrapper | Node: Streamlit Cloud Engine")
  
# Run automated data/model linkage immediately on page render
if not st.session_state.get("model_loaded", False):
    url_to_load = gdrive_url.strip() if gdrive_url else secret_url.strip()
    if url_to_load:
        file_id = extract_gdrive_id(url_to_load)
        if file_id:
            try:
                # Triggers your cached download and local .keras memory loading
                _m = load_model_cached(file_id)
                st.session_state["model_obj"] = _m
                st.session_state["model_loaded"] = True
                model_status_slot.success("🟢 Systems Operational (Model Live)")
            except Exception as e:
                st.session_state["model_loaded"] = False
                st.sidebar.error(f"❌ Core Initialization Fault: {e}")

# ╔══════════════════════════════════════════════════════════╗
# ║                    MODEL LOADING                         ║
# ╚══════════════════════════════════════════════════════════╝

# Attempt model load when button pressed
if load_btn:
    url_clean = gdrive_url.strip() if gdrive_url else ""
    if not url_clean:
        st.sidebar.error("❌ Please paste a Google Drive link first.")
    else:
        file_id = extract_gdrive_id(url_clean)
        if file_id is None:
            st.sidebar.error(
                "❌ Could not extract a file ID from that URL. "
                "Make sure it's a standard Google Drive sharing link."
            )
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

# Retrieve model from session (persists across reruns)
# Map active binary from session memory state down to your prediction engines
model = st.session_state.get("model_obj", None)
eff_entry = entry_price if entry_price > 0.0 else current_price

# Effective entry price
eff_entry = entry_price if entry_price > 0.0 else current_price


# ╔══════════════════════════════════════════════════════════╗
# ║                    HEADER STRIP                          ║
# ╚══════════════════════════════════════════════════════════╝

col_t, col_p, col_d, col_v, col_sp = st.columns([3, 2, 2, 2, 2])

with col_t:
    st.markdown("### ⚡ Tesla, Inc. &nbsp;`TSLA`")
    st.caption(f"Last trading day: {df.index[-1].strftime('%d %b %Y')}")

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

tab1, tab2, tab3 = st.tabs(["📡  Trade Signals", "🔮  Forecast", "📊  Visualizations"])


# ════════════════════════════════════════════════════════════
#  TAB 1 — TRADE SIGNALS
# ════════════════════════════════════════════════════════════
with tab1:

    # ── Safe indicator reads ─────────────────────────────────
    def last_valid(series, fallback=np.nan):
        s = series.dropna()
        return safe_float(s.iloc[-1], fallback) if not s.empty else fallback

    rsi_now  = last_valid(df["RSI"])
    macd_now = last_valid(df["MACD"])
    sig_now  = last_valid(df["MACDSig"])
    ma30_now = last_valid(df["MA30"])
    ma90_now = last_valid(df["MA90"])
    bb_up    = last_valid(df["BB_Upper"])
    bb_lo    = last_valid(df["BB_Lower"])

    # ── Technical scoring ────────────────────────────────────
    # Each indicator: +1 bullish, -1 bearish, 0 neutral / N/A
    tech_scores: dict[str, int] = {}

    if not np.isnan(rsi_now):
        tech_scores["RSI (14)"] = 1 if rsi_now < 30 else (-1 if rsi_now > 70 else 0)
    else:
        tech_scores["RSI (14)"] = 0

    if not (np.isnan(macd_now) or np.isnan(sig_now)):
        tech_scores["MACD vs Signal"] = 1 if macd_now > sig_now else -1
    else:
        tech_scores["MACD vs Signal"] = 0

    if not (np.isnan(ma30_now) or np.isnan(ma90_now)):
        tech_scores["MA30 vs MA90"] = 1 if ma30_now > ma90_now else -1
    else:
        tech_scores["MA30 vs MA90"] = 0

    if not (np.isnan(bb_up) or np.isnan(bb_lo)):
        tech_scores["Bollinger Band"] = (
            1 if current_price < bb_lo else (-1 if current_price > bb_up else 0)
        )
    else:
        tech_scores["Bollinger Band"] = 0

    # ── Model-based signal (optional) ────────────────────────
    model_target_price: float | None = None
    model_signal_score = 0
    model_signal_error = None

    if model is not None and scaler_ok:
        try:
            _, _p5, _, _ = recursive_forecast(model, scaler, df, n_days=5)
            model_target_price  = safe_float(_p5[-1])
            if model_target_price > 0:
                model_pct          = (model_target_price - current_price) / current_price * 100
                model_signal_score = 1 if model_pct > 1.5 else (-1 if model_pct < -1.5 else 0)
                tech_scores["Model (5-day)"] = model_signal_score
        except Exception as _me:
            model_signal_error = str(_me)

    total_score = sum(tech_scores.values())
    n_signals   = len(tech_scores)

    if   total_score >= 2:  signal_label, signal_css = "BUY",  "signal-buy"
    elif total_score <= -2: signal_label, signal_css = "SELL", "signal-sell"
    else:                   signal_label, signal_css = "HOLD", "signal-hold"

    # ── Risk / Reward maths ──────────────────────────────────
    stop_loss    = eff_entry * (1 - risk_pct / 100)
    risk_per_sh  = max(eff_entry - stop_loss, 0.01)   # guard /0
    take_profit  = eff_entry + risk_per_sh * 2.0       # 1:2 R/R
    max_loss     = risk_per_sh * position_qty
    max_gain     = (take_profit - eff_entry) * position_qty
    model_gain   = ((model_target_price - eff_entry) * position_qty
                    if model_target_price is not None else None)
    implied_rr   = ((model_target_price - eff_entry) / risk_per_sh
                    if model_target_price is not None else None)

    # ── Layout ───────────────────────────────────────────────
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.markdown('<p class="section-header">Composite Signal</p>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="text-align:center;padding:28px 0 16px">'
            f'<div class="{signal_css}">{signal_label}</div>'
            f'<div style="color:{MUTED};font-size:0.78rem;margin-top:10px">'
            f'Score: {total_score:+d} across {n_signals} indicators</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<p class="section-header">Indicator Breakdown</p>', unsafe_allow_html=True)
        label_map = {1: ("🟢", "Bullish"), -1: ("🔴", "Bearish"), 0: ("🟡", "Neutral")}
        for ind_name, score in tech_scores.items():
            icon, lbl = label_map[score]
            st.markdown(f"{icon} &nbsp;**{ind_name}** — {lbl}")

        if model_signal_error:
            st.caption(f"⚠️ Model signal skipped: {model_signal_error}")

        st.markdown("---")
        st.markdown('<p class="section-header">Key Levels</p>', unsafe_allow_html=True)

        def _fmt(v, prefix="$", dec=2):
            return f"{prefix}{v:.{dec}f}" if not np.isnan(v) else "N/A"

        st.markdown(f"RSI: **{_fmt(rsi_now, '', 1)}**")
        st.markdown(f"MACD: **{_fmt(macd_now, '', 3)}** &nbsp;|&nbsp; Signal: **{_fmt(sig_now, '', 3)}**")
        st.markdown(f"BB Upper: **{_fmt(bb_up)}** &nbsp;|&nbsp; Lower: **{_fmt(bb_lo)}**")
        if model_target_price:
            st.markdown(f"Model 5d target: **{_fmt(model_target_price)}**")

    with right:
        st.markdown('<p class="section-header">Risk / Reward Calculator</p>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(metric_card("Entry", f"${eff_entry:.2f}"), unsafe_allow_html=True)
        with c2:
            st.markdown(
                metric_card("Stop Loss", f"${stop_loss:.2f}",
                            f"−{risk_pct}% risk", "metric-delta-down"),
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                metric_card("Take Profit (1:2)", f"${take_profit:.2f}",
                            f"+{risk_pct*2:.0f}% target", "metric-delta-up"),
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        c4, c5, c6 = st.columns(3)
        with c4:
            st.markdown(
                metric_card("Max Loss", f"−${max_loss:.0f}",
                            f"{position_qty} shares", "metric-delta-down"),
                unsafe_allow_html=True,
            )
        with c5:
            st.markdown(
                metric_card("Target Gain", f"+${max_gain:.0f}",
                            "R/R  1 : 2", "metric-delta-up"),
                unsafe_allow_html=True,
            )
        with c6:
            if model_gain is not None and implied_rr is not None:
                g_cls = "metric-delta-up" if model_gain >= 0 else "metric-delta-down"
                g_sgn = "+" if model_gain >= 0 else ""
                st.markdown(
                    metric_card(
                        "Model P&L (5d)", f"{g_sgn}${model_gain:.0f}",
                        f"R/R  1 : {implied_rr:.1f}", g_cls,
                    ),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    metric_card("Model P&L (5d)", "—",
                                "Load model to see", "metric-muted"),
                    unsafe_allow_html=True,
                )

        # ── Price + trade levels chart ───────────────────────
        st.markdown('<p class="section-header">Recent Price vs Trade Levels</p>',
                    unsafe_allow_html=True)

        chart_df = df[["Close", "BB_Upper", "BB_Lower", "MA30", "MA90"]].tail(90).dropna(
            subset=["Close"]
        )

        if chart_df.empty:
            empty_state("📉", "Not enough data to render the trade chart.")
        else:
            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(x=chart_df.index, y=chart_df["Close"],
                                       name="Close", line=dict(color=ACCENT, width=1.8)))
            if chart_df["MA30"].notna().any():
                fig_t.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MA30"],
                                           name="MA30", line=dict(color=BLUE, width=1, dash="dot")))
            if chart_df["MA90"].notna().any():
                fig_t.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MA90"],
                                           name="MA90", line=dict(color=PURPLE, width=1, dash="dot")))
            if chart_df["BB_Upper"].notna().any():
                fig_t.add_trace(go.Scatter(x=chart_df.index, y=chart_df["BB_Upper"],
                                           name="BB Upper",
                                           line=dict(color=MUTED, width=0.8, dash="dash"),
                                           opacity=0.5))
            if chart_df["BB_Lower"].notna().any():
                fig_t.add_trace(go.Scatter(x=chart_df.index, y=chart_df["BB_Lower"],
                                           fill="tonexty",
                                           fillcolor="rgba(122,128,153,0.07)",
                                           name="BB Lower",
                                           line=dict(color=MUTED, width=0.8, dash="dash"),
                                           opacity=0.5))

            for lvl, lbl, col in [
                (eff_entry,   "Entry",  ACCENT),
                (stop_loss,   "Stop",   RED),
                (take_profit, "Target", GREEN),
            ]:
                if lvl > 0:
                    fig_t.add_hline(
                        y=lvl, line_color=col, line_dash="dash", line_width=1.2,
                        annotation_text=f"  {lbl}: ${lvl:.2f}",
                        annotation_font_color=col,
                    )

            fig_t.update_layout(
                **base_layout(330),
                legend=dict(orientation="h", y=-0.18, font_size=10),
                yaxis=dict(gridcolor=GRID_COL, tickprefix="$"),
            )
            st.plotly_chart(fig_t, use_container_width=True)


# ════════════════════════════════════════════════════════════
#  TAB 2 — FORECAST
# ════════════════════════════════════════════════════════════
with tab2:

    if model is None:
        empty_state(
            "🔮",
            "Paste your Google Drive model link in the sidebar and click <strong>⬇ Load Model</strong>.<br>"
            "Once loaded, the CNN-GRU model will forecast up to 60 trading days ahead.",
        )
    elif not scaler_ok:
        empty_state("⚠️", "Scaler could not be initialised. Check data warnings above.")
    else:
        with st.spinner(f"Running {forecast_days}-day recursive forecast…"):
            try:
                f_dates, f_prices, f_lower, f_upper = recursive_forecast(
                    model, scaler, df, n_days=forecast_days
                )
                forecast_ok  = True
                forecast_err = None
            except Exception as _fe:
                forecast_ok  = False
                forecast_err = str(_fe)

        if not forecast_ok:
            empty_state("⛔", f"Forecast failed: {forecast_err}")
        else:
            # Guard against NaN/inf in forecast arrays
            if np.any(~np.isfinite(f_prices)):
                empty_state(
                    "⚠️",
                    "Forecast produced invalid values (NaN / Inf). "
                    "This usually means the model weights are incompatible with the input shape. "
                    "Please verify the model file.",
                )
            else:
                f_end     = safe_float(f_prices[-1])
                f_chg_pct = (f_end - current_price) / current_price * 100 if current_price else 0
                f_hi      = safe_float(f_prices.max())
                f_lo      = safe_float(f_prices.min())
                f_end_lo  = safe_float(f_lower[-1])
                f_end_hi  = safe_float(f_upper[-1])

                # ── Summary cards ─────────────────────────────
                c1, c2, c3, c4 = st.columns(4)
                chg_cls   = "metric-delta-up" if f_chg_pct >= 0 else "metric-delta-down"
                chg_arrow = "▲" if f_chg_pct >= 0 else "▼"

                with c1:
                    st.markdown(metric_card("Current Price", f"${current_price:.2f}"),
                                unsafe_allow_html=True)
                with c2:
                    st.markdown(
                        metric_card(f"Day-{forecast_days} Target", f"${f_end:.2f}",
                                    f"{chg_arrow} {f_chg_pct:+.1f}%", chg_cls),
                        unsafe_allow_html=True,
                    )
                with c3:
                    st.markdown(metric_card("Forecast High", f"${f_hi:.2f}"),
                                unsafe_allow_html=True)
                with c4:
                    st.markdown(metric_card("Forecast Low", f"${f_lo:.2f}"),
                                unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Forecast chart ────────────────────────────
                hist_win = df[["Adj Close"]].tail(120).dropna()

                fig_fc = go.Figure()

                # Historical line
                fig_fc.add_trace(go.Scatter(
                    x=hist_win.index, y=hist_win["Adj Close"],
                    name="Historical", line=dict(color=ACCENT, width=2),
                    hovertemplate="<b>%{x|%d %b %Y}</b><br>Close: $%{y:.2f}<extra></extra>",
                ))

                # Confidence band (polygon fill)
                band_x = list(f_dates) + list(f_dates[::-1])
                band_y = list(f_upper) + list(f_lower[::-1])
                fig_fc.add_trace(go.Scatter(
                    x=band_x, y=band_y,
                    fill="toself", fillcolor="rgba(91,141,238,0.13)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="Confidence Band", hoverinfo="skip",
                ))

                # Forecast line with per-point hover
                fig_fc.add_trace(go.Scatter(
                    x=f_dates, y=f_prices,
                    name="Forecast",
                    line=dict(color=BLUE, width=2, dash="dot"),
                    mode="lines+markers",
                    marker=dict(size=5, color=BLUE),
                    customdata=np.stack([f_lower, f_upper,
                                         (f_prices - current_price) / current_price * 100],
                                        axis=-1),
                    hovertemplate=(
                        "<b>%{x|%d %b %Y}</b><br>"
                        "Forecast: <b>$%{y:.2f}</b><br>"
                        "vs Today: %{customdata[2]:+.1f}%<br>"
                        "Lower bound: $%{customdata[0]:.2f}<br>"
                        "Upper bound: $%{customdata[1]:.2f}"
                        "<extra></extra>"
                    ),
                ))

                # "Today" divider
                fig_fc.add_vline(
                    x=df.index[-1],
                    line_color=MUTED, line_dash="dash", line_width=1,
                    annotation_text="  Today", annotation_font_color=MUTED,
                )

                fig_fc.update_layout(
                    **base_layout(460),
                    legend=dict(orientation="h", y=-0.12, font_size=11),
                    yaxis=dict(gridcolor=GRID_COL, showgrid=True, tickprefix="$"),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_fc, use_container_width=True)

                st.caption(
                    "⚠️ Confidence band = historical 60-day volatility × √t. "
                    "Not financial advice — past performance does not predict future results."
                )

                # ── Forecast table ────────────────────────────
                with st.expander("📋 View full forecast table"):
                    fc_rows = []
                    for i, (d, p, lo, hi) in enumerate(
                        zip(f_dates, f_prices, f_lower, f_upper)
                    ):
                        fc_rows.append({
                            "Day":           i + 1,
                            "Date":          d.strftime("%d %b %Y"),
                            "Forecast ($)":  f"${p:.2f}",
                            "Lower ($)":     f"${lo:.2f}",
                            "Upper ($)":     f"${hi:.2f}",
                            "Δ from Today":  f"{((p-current_price)/current_price)*100:+.2f}%",
                        })
                    st.dataframe(
                        pd.DataFrame(fc_rows), use_container_width=True, hide_index=True
                    )


# ════════════════════════════════════════════════════════════
#  TAB 3 — VISUALIZATIONS
# ════════════════════════════════════════════════════════════
with tab3:

    # ── Date range filter ────────────────────────────────────
    fa, fb = st.columns(2)
    with fa:
        viz_start = st.date_input(
            "From", value=df.index[0].date(),
            min_value=df.index[0].date(), max_value=df.index[-1].date(),
            key="viz_from",
        )
    with fb:
        viz_end = st.date_input(
            "To", value=df.index[-1].date(),
            min_value=df.index[0].date(), max_value=df.index[-1].date(),
            key="viz_to",
        )

    if viz_start >= viz_end:
        st.warning("⚠️ 'From' date must be before 'To' date.")
        st.stop()

    dv = df.loc[str(viz_start):str(viz_end)].copy()

    if dv.empty:
        empty_state("📭", "No data in the selected date range. Adjust the dates above.")
        st.stop()

    if len(dv) < 5:
        st.warning("⚠️ Selected range has fewer than 5 trading days — some charts may look sparse.")

    st.markdown("---")

    # ── Plot helper: skip trace if all-NaN ───────────────────
    def has_data(series) -> bool:
        return series.notna().any()

    # ── Row 1: Price + MAs  |  Volume ────────────────────────
    r1a, r1b = st.columns([3, 2])

    with r1a:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=dv.index, y=dv["Close"],
                                  name="Close", line=dict(color=ACCENT, width=1.8)))
        for col_name, col_color in [("MA30", BLUE), ("MA90", PURPLE), ("MA200", RED)]:
            if has_data(dv[col_name]):
                fig1.add_trace(go.Scatter(
                    x=dv.index, y=dv[col_name], name=col_name,
                    line=dict(color=col_color, width=1, dash="dot"),
                ))
        fig1.update_layout(
            **base_layout(360, "Closing Price + Moving Averages"),
            legend=dict(orientation="h", y=-0.18, font_size=10),
            yaxis=dict(gridcolor=GRID_COL, tickprefix="$"),
        )
        st.plotly_chart(fig1, use_container_width=True)

    with r1b:
        vol_colors = []
        for i in range(len(dv)):
            if i == 0:
                vol_colors.append(GREEN)
            else:
                vol_colors.append(GREEN if dv["Close"].iloc[i] >= dv["Close"].iloc[i-1] else RED)
        fig2 = go.Figure(go.Bar(x=dv.index, y=dv["Volume"],
                                marker_color=vol_colors, name="Volume"))
        fig2.update_layout(**base_layout(360, "Daily Volume"),
                           yaxis=dict(gridcolor=GRID_COL))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 2: Candlestick + BB  |  Spread ───────────────────
    r2a, r2b = st.columns([3, 2])

    with r2a:
        fig3 = go.Figure(go.Candlestick(
            x=dv.index, open=dv["Open"], high=dv["High"],
            low=dv["Low"], close=dv["Close"],
            increasing_line_color=GREEN, decreasing_line_color=RED, name="OHLC",
        ))
        if has_data(dv["BB_Upper"]) and has_data(dv["BB_Lower"]):
            fig3.add_trace(go.Scatter(x=dv.index, y=dv["BB_Upper"], name="BB Upper",
                                      line=dict(color=MUTED, width=0.8, dash="dash")))
            fig3.add_trace(go.Scatter(x=dv.index, y=dv["BB_Lower"],
                                      fill="tonexty", fillcolor="rgba(122,128,153,0.06)",
                                      name="BB Lower",
                                      line=dict(color=MUTED, width=0.8, dash="dash")))
        fig3.update_layout(
            **base_layout(360, "Candlestick + Bollinger Bands"),
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", y=-0.18, font_size=10),
            yaxis=dict(gridcolor=GRID_COL, tickprefix="$"),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with r2b:
        if has_data(dv["Spread"]):
            fig4 = go.Figure(go.Scatter(
                x=dv.index, y=dv["Spread"],
                fill="tozeroy", fillcolor=f"rgba(232,200,74,0.12)",
                line=dict(color=ACCENT, width=1.2), name="Spread",
            ))
            fig4.update_layout(**base_layout(360, "Intraday Volatility (High − Low)"),
                               yaxis=dict(gridcolor=GRID_COL, tickprefix="$"))
            st.plotly_chart(fig4, use_container_width=True)
        else:
            empty_state("📉", "Spread data unavailable for this range.")

    # ── Row 3: MACD  |  RSI ──────────────────────────────────
    r3a, r3b = st.columns(2)

    with r3a:
        if has_data(dv["MACD"]) and has_data(dv["MACDSig"]):
            fig5 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                 row_heights=[0.6, 0.4], vertical_spacing=0.04)
            fig5.add_trace(go.Scatter(x=dv.index, y=dv["MACD"], name="MACD",
                                      line=dict(color=BLUE, width=1.5)), row=1, col=1)
            fig5.add_trace(go.Scatter(x=dv.index, y=dv["MACDSig"], name="Signal",
                                      line=dict(color=ACCENT, width=1.5)), row=1, col=1)
            hist_colors = [GREEN if v >= 0 else RED for v in dv["MACDHist"].fillna(0)]
            fig5.add_trace(go.Bar(x=dv.index, y=dv["MACDHist"],
                                  name="Histogram", marker_color=hist_colors), row=2, col=1)
            fig5.update_layout(
                paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
                font_color=FONT_COL, height=360,
                margin=dict(l=0, r=0, t=36, b=0),
                title=dict(text="MACD (12, 26, 9)", font=dict(size=13, color=MUTED)),
                legend=dict(orientation="h", y=-0.18, font_size=10),
                xaxis=dict(gridcolor=GRID_COL), xaxis2=dict(gridcolor=GRID_COL),
                yaxis=dict(gridcolor=GRID_COL), yaxis2=dict(gridcolor=GRID_COL),
            )
            st.plotly_chart(fig5, use_container_width=True)
        else:
            empty_state("📉", "MACD data not available for this date range.")

    with r3b:
        if has_data(dv["RSI"]):
            fig6 = go.Figure()
            fig6.add_trace(go.Scatter(x=dv.index, y=dv["RSI"], name="RSI",
                                      line=dict(color=PURPLE, width=1.5)))
            fig6.add_hrect(y0=70, y1=100, fillcolor=f"rgba(240,82,82,0.08)", line_width=0)
            fig6.add_hrect(y0=0,  y1=30,  fillcolor=f"rgba(38,212,124,0.08)", line_width=0)
            fig6.add_hline(y=70, line_color=RED,   line_dash="dash", line_width=0.8,
                           annotation_text="  Overbought", annotation_font_color=RED)
            fig6.add_hline(y=30, line_color=GREEN, line_dash="dash", line_width=0.8,
                           annotation_text="  Oversold",   annotation_font_color=GREEN)
            fig6.update_layout(**base_layout(360, "RSI (14)"),
                               yaxis=dict(gridcolor=GRID_COL, range=[0, 100]))
            st.plotly_chart(fig6, use_container_width=True)
        else:
            empty_state("📉", "RSI data not available for this date range.")

    # ── Row 4: Annual avg  |  Monthly box ────────────────────
    r4a, r4b = st.columns(2)

    with r4a:
        yearly = (dv.groupby(dv.index.year)["Close"].mean()
                    .reset_index()
                    .rename(columns={"Date": "Year", "Close": "Avg Close"}))
        if not yearly.empty:
            fig7 = go.Figure(go.Bar(x=yearly.iloc[:, 0].astype(str),
                                    y=yearly["Avg Close"],
                                    marker_color=ACCENT, name="Avg Close"))
            fig7.update_layout(**base_layout(360, "Annual Average Closing Price"),
                               yaxis=dict(gridcolor=GRID_COL, tickprefix="$"),
                               xaxis=dict(gridcolor=GRID_COL))
            st.plotly_chart(fig7, use_container_width=True)
        else:
            empty_state("📅", "Not enough data to plot annual averages.")

    with r4b:
        months     = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        fig8       = go.Figure()
        months_ok  = 0
        for m_idx, m_name in enumerate(months, 1):
            subset = dv[dv.index.month == m_idx]["Close"].dropna()
            if subset.empty:
                continue
            fig8.add_trace(go.Box(y=subset, name=m_name,
                                  marker_color=BLUE, line_color=BLUE,
                                  fillcolor="rgba(91,141,238,0.18)"))
            months_ok += 1
        if months_ok:
            fig8.update_layout(**base_layout(360, "Monthly Price Distribution"),
                               yaxis=dict(gridcolor=GRID_COL, tickprefix="$"),
                               showlegend=False)
            st.plotly_chart(fig8, use_container_width=True)
        else:
            empty_state("📅", "Not enough monthly data to draw box plots.")

    # ── Row 5: Returns hist  |  Open vs Close scatter ────────
    r5a, r5b = st.columns(2)

    with r5a:
        ret_data = dv["DailyReturn"].replace([np.inf, -np.inf], np.nan).dropna()
        if len(ret_data) >= 5:
            fig9 = go.Figure(go.Histogram(x=ret_data, nbinsx=60,
                                          marker_color=ACCENT, opacity=0.8,
                                          name="Daily Return %"))
            fig9.update_layout(**base_layout(340, "Daily Return Distribution (%)"),
                               bargap=0.02,
                               xaxis=dict(gridcolor=GRID_COL, ticksuffix="%"),
                               yaxis=dict(gridcolor=GRID_COL))
            st.plotly_chart(fig9, use_container_width=True)
        else:
            empty_state("📉", "Not enough return data in this range.")

    with r5b:
        oc_data = dv[["Open", "Close", "Volume"]].dropna()
        if len(oc_data) >= 5:
            fig10 = go.Figure(go.Scatter(
                x=oc_data["Open"], y=oc_data["Close"],
                mode="markers",
                marker=dict(color=oc_data["Volume"], colorscale="Viridis",
                            size=4, opacity=0.6,
                            colorbar=dict(title="Volume", tickfont=dict(size=9))),
                name="Open vs Close",
                hovertemplate="Open: $%{x:.2f}<br>Close: $%{y:.2f}<extra></extra>",
            ))
            fig10.update_layout(
                **base_layout(340, "Open vs Close (coloured by Volume)"),
                xaxis=dict(gridcolor=GRID_COL, tickprefix="$", title="Open"),
                yaxis=dict(gridcolor=GRID_COL, tickprefix="$", title="Close"),
            )
            st.plotly_chart(fig10, use_container_width=True)
        else:
            empty_state("📉", "Not enough data for scatter plot.")

    # ── Row 6: Correlation heatmap ────────────────────────────
    st.markdown('<p class="section-header">Feature Correlation Heatmap</p>',
                unsafe_allow_html=True)

    corr_cols    = [c for c in ["Open","High","Low","Close","Adj Close","Volume","Spread"]
                    if c in dv.columns]
    corr_data    = dv[corr_cols].dropna()

    if len(corr_data) >= 5 and len(corr_cols) >= 2:
        corr_mat = corr_data.corr().round(3)
        fig11 = go.Figure(go.Heatmap(
            z=corr_mat.values, x=corr_cols, y=corr_cols,
            colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            text=corr_mat.values, texttemplate="%{text:.2f}",
            showscale=True,
        ))
        fig11.update_layout(
            paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
            font_color=FONT_COL, height=380,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig11, use_container_width=True)
    else:
        empty_state("📊", "Not enough columns or rows to compute a correlation matrix.")
