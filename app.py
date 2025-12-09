import math
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, render_template, request

app = Flask(__name__)

# --------- INDICATORS USING PAST DATA --------- #

def ema(series: pd.Series, window: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=window, adjust=False).mean()

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """RSI momentum indicator using past candles."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

# --------- DATA FETCH + FEATURE BUILD --------- #

def get_latest_features():
    """
    Download recent NatGas (NG=F) candles from Yahoo Finance
    and calculate indicators that look at the past.
    """
    try:
        # 60 days of hourly candles (1h) gives a nice history
        df = yf.download("NG=F", period="60d", interval="1h")

        if df is None or df.empty:
            return None, "No NatGas data received from Yahoo Finance."

        df = df.dropna()
        if df.empty:
            return None, "NatGas data empty after cleaning."

        close = df["Close"]

        # Technical indicators that depend on past prices
        df["ema_fast"] = ema(close, 10)
        df["ema_slow"] = ema(close, 30)
        df["ema_diff"] = df["ema_fast"] - df["ema_slow"]
        df["rsi"] = rsi(close, 14)

        # 1h returns + rolling volatility as risk measure
        df["ret_1h"] = close.pct_change(1)
        df["volatility_24h"] = df["ret_1h"].rolling(24).std()

        df = df.dropna()
        if df.empty:
            return None, "Not enough candles to calculate indicators."

        latest = df.iloc[-1]
        ts = df.index[-1]
        try:
            ts_str = ts.tz_convert("UTC").strftime("%Y-%m-%d %H:%M")
        except Exception:
            ts_str = str(ts)

        feats = {
            "last_price": float(latest["Close"]),
            "ema_fast": float(latest["ema_fast"]),
            "ema_slow": float(latest["ema_slow"]),
            "ema_diff": float(latest["ema_diff"]),
            "rsi": float(latest["rsi"]),
            "vol_24h": float(latest["volatility_24h"]),
            "timestamp": ts_str,
        }
        return feats, None

    except Exception as e:
        return None, f"Data error: {e}"

# --------- SIGNAL LOGIC --------- #

def make_signal(features):
    """
    Rule-based signal using:
    - EMA 10 vs EMA 30 (trend)
    - RSI (momentum)
    - Volatility for stop size
    """
    last_price = features["last_price"]
    ema_fast_val = features["ema_fast"]
    ema_slow_val = features["ema_slow"]
    rsi_val = features["rsi"]
    vol_24h = features["vol_24h"]

    # Trend + RSI rules
    if ema_fast_val > ema_slow_val and 45 <= rsi_val <= 65:
        direction = "UP"
        base_conf = 0.65
    elif ema_fast_val < ema_slow_val and 35 <= rsi_val <= 55:
        direction = "DOWN"
        base_conf = 0.65
    else:
        direction = "FLAT"
        base_conf = 0.5

    # Volatility-based stop (0.5% to 3% of price)
    if vol_24h is None or math.isnan(vol_24h) or vol_24h == 0:
        stop_pct = 0.01
    else:
        stop_pct = min(max(vol_24h * 2.0, 0.005), 0.03)

    tp_pct = stop_pct * 3  # TP at 3x stop distance

    # Confidence nudged by EMA spread (how strong the trend is)
    ema_spread = abs(ema_fast_val - ema_slow_val) / (last_price + 1e-9)
    confidence = float(min(max(base_conf + ema_spread * 0.5, 0.4), 0.9))

    # Convert stop/TP % to price levels
    if direction == "UP":
        stop_loss = last_price * (1 - stop_pct)
        take_profit = last_price * (1 + tp_pct)
    elif direction == "DOWN":
        stop_loss = last_price * (1 + stop_pct)
        take_profit = last_price * (1 - tp_pct)
    else:
        stop_loss = last_price * (1 - stop_pct)
        take_profit = last_price * (1 + tp_pct)

    return {
        "direction": direction,
        "confidence": confidence,
        "stop_pct": stop_pct,
        "tp_pct": tp_pct,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
    }

# --------- FLASK WEB APP --------- #

@app.route("/", methods=["GET", "POST"])
def index():
    account_balance = None
    risk_pct = 1.0
    position_size = None
    signal = None
    last_price = None
    timestamp = None
    error_msg = None

    # 1) Fetch live NatGas data + build features
    feats, data_error = get_latest_features()
    if data_error:
        error_msg = data_error
    else:
        timestamp = feats["timestamp"]
        last_price = feats["last_price"]
        signal = make_signal(feats)

    # 2) Position sizing from user input
    if request.method == "POST":
        try:
            account_balance = float(request.form.get("account_balance", "0"))
        except ValueError:
            account_balance = 0.0

        try:
            risk_pct = float(request.form.get("risk_pct", "1.0"))
        except ValueError:
            risk_pct = 1.0

        if signal and account_balance and last_price:
            risk_amount = account_balance * (risk_pct / 100.0)
            stop_distance = abs(last_price - signal["stop_loss"])
            if stop_distance > 0:
                position_size = risk_amount / stop_distance

    return render_template(
        "index.html",
        signal=signal,
        last_price=last_price,
        timestamp=timestamp,
        account_balance=account_balance,
        risk_pct=risk_pct,
        position_size=position_size,
        error_msg=error_msg,
    )

if __name__ == "__main__":
    # Local run (Render will use gunicorn)
    app.run(host="0.0.0.0", port=8000, debug=True)
