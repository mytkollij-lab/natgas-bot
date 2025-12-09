import math
import json
from urllib.request import urlopen

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, render_template, request

app = Flask(__name__)

# --------- WEATHER SETTINGS (US + EUROPE) --------- #

WEATHER_LOCATIONS = [
    # US
    {"name": "US Northeast (New York)", "lat": 40.71, "lon": -74.00},
    {"name": "US Midwest (Chicago)", "lat": 41.88, "lon": -87.63},
    {"name": "US Texas (Houston)", "lat": 29.76, "lon": -95.37},
    {"name": "US Southeast (Atlanta)", "lat": 33.75, "lon": -84.39},
    {"name": "US West Coast (Los Angeles)", "lat": 34.05, "lon": -118.24},
    # Europe
    {"name": "UK (London)", "lat": 51.50, "lon": -0.12},
    {"name": "Germany (Berlin)", "lat": 52.52, "lon": 13.40},
    {"name": "Netherlands (Amsterdam)", "lat": 52.37, "lon": 4.90},
    {"name": "France (Paris)", "lat": 48.86, "lon": 2.35},
    {"name": "Italy (Milan)", "lat": 45.46, "lon": 9.19},
    {"name": "Spain (Madrid)", "lat": 40.42, "lon": -3.70},
]


def fetch_weather_for_location(lat: float, lon: float):
    """
    Use Open-Meteo free API to get next 7 days of hourly temperature.
    Return: (current_temp, HDD_7d, CDD_7d)
    HDD/CDD base = 18°C.
    """
    base_temp = 18.0
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m&forecast_days=7"
    )
    with urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    temps = data.get("hourly", {}).get("temperature_2m", [])
    if not temps:
        return None, 0.0, 0.0

    current_temp = temps[0]
    hdd = 0.0
    cdd = 0.0
    for t in temps:
        hdd += max(0.0, base_temp - t)
        cdd += max(0.0, t - base_temp)
    return current_temp, hdd, cdd


def get_weather_summary():
    """
    Fetch weather for all US + EU locations.
    Build a 'weather demand score' which we will use to adjust confidence.
    Returns:
      weather_info (dict) or None,
      weather_error (str or None),
      weather_score (float in [-0.25, +0.25])
    """
    locations_data = []
    total_hdd = 0.0
    total_cdd = 0.0
    count = 0

    for loc in WEATHER_LOCATIONS:
        try:
            temp, hdd, cdd = fetch_weather_for_location(loc["lat"], loc["lon"])
            locations_data.append(
                {
                    "name": loc["name"],
                    "temp": temp,
                    "hdd7": hdd,
                    "cdd7": cdd,
                }
            )
            total_hdd += hdd
            total_cdd += cdd
            count += 1
        except Exception:
            # Just skip this location if weather fails
            continue

    if count == 0:
        return None, "Weather API error (Open-Meteo).", 0.0

    avg_hdd = total_hdd / count
    avg_cdd = total_cdd / count

    # Rough scaling: bigger number => stronger demand
    heating_strength = avg_hdd / 100.0
    cooling_strength = avg_cdd / 100.0

    weather_score = 0.0

    # Cold = heating demand = bullish for NatGas
    if heating_strength > 1.5:
        weather_score += 0.20
    elif heating_strength > 0.8:
        weather_score += 0.10

    # Hot = cooling demand = bullish for NatGas
    if cooling_strength > 1.5:
        weather_score += 0.15
    elif cooling_strength > 0.8:
        weather_score += 0.07

    # Very mild = bearish
    if heating_strength < 0.4 and cooling_strength < 0.4:
        weather_score -= 0.15

    # Clamp to [-0.25, +0.25]
    weather_score = max(min(weather_score, 0.25), -0.25)

    # Text for UI
    if weather_score > 0.15:
        impact_text = "Weather strongly bullish for NatGas (high heating/cooling demand)."
    elif weather_score > 0.05:
        impact_text = "Weather slightly bullish for NatGas."
    elif weather_score < -0.05:
        impact_text = "Weather slightly bearish for NatGas (demand looks mild)."
    else:
        impact_text = "Weather roughly neutral for NatGas demand."

    weather_info = {
        "locations": locations_data,
        "avg_hdd": avg_hdd,
        "avg_cdd": avg_cdd,
        "impact_text": impact_text,
        "score": weather_score,
    }

    return weather_info, None, weather_score


# ---------- INDICATORS USING PAST MARKET DATA ---------- #

def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()


def bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return mid, upper, lower


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


# ---------- DATA FETCH + FEATURE BUILD (MARKET) ---------- #

def get_latest_features():
    """
    Download recent NatGas (NG=F) and Crude Oil (CL=F) candles from Yahoo Finance
    and calculate indicators that look at the past.
    """
    try:
        # 120 days of hourly candles
        ng = yf.download("NG=F", period="120d", interval="1h")
        cl = yf.download("CL=F", period="120d", interval="1h")

        if ng is None or ng.empty:
            return None, "No NatGas data received from Yahoo Finance."
        if cl is None or cl.empty:
            return None, "No Crude Oil data received from Yahoo Finance."

        df = pd.DataFrame(index=ng.index)
        df["ng_close"] = ng["Close"]
        df["ng_high"] = ng["High"]
        df["ng_low"] = ng["Low"]
        df["cl_close"] = cl["Close"]
        df = df.dropna()

        if df.empty:
            return None, "Not enough overlapping NG & CL data after cleaning."

        close = df["ng_close"]
        high = df["ng_high"]
        low = df["ng_low"]
        cl_close = df["cl_close"]

        # EMAs
        df["ema_fast"] = ema(close, 10)
        df["ema_slow"] = ema(close, 30)
        df["ema_long"] = ema(close, 50)

        # RSI
        df["rsi"] = rsi(close, 14)

        # Volatility (24h std of 1h returns)
        df["ret_1h"] = close.pct_change(1)
        df["volatility_24h"] = df["ret_1h"].rolling(24).std()

        # ATR
        df["atr_14"] = atr(high, low, close, 14)

        # Bollinger
        mid, upper, lower = bollinger_bands(close, 20, 2.0)
        df["bb_mid"] = mid
        df["bb_upper"] = upper
        df["bb_lower"] = lower
        df["bb_pos"] = (close - lower) / (upper - lower + 1e-9)

        # MACD
        macd_line, signal_line, hist = macd(close, 12, 26, 9)
        df["macd_line"] = macd_line
        df["macd_signal"] = signal_line
        df["macd_hist"] = hist

        # NG/CL ratio
        df["ng_cl_ratio"] = close / cl_close
        ratio = df["ng_cl_ratio"]
        ratio_ma = ratio.rolling(50).mean()
        ratio_std = ratio.rolling(50).std()
        df["ng_cl_ratio_z"] = (ratio - ratio_ma) / (ratio_std + 1e-9)

        df = df.dropna()
        if df.empty:
            return None, "Not enough candles to calculate extended indicators."

        latest = df.iloc[-1]
        ts = df.index[-1]
        try:
            ts_str = ts.tz_convert("UTC").strftime("%Y-%m-%d %H:%M")
        except Exception:
            ts_str = str(ts)

        feats = {
            "last_price": float(latest["ng_close"]),
            "ema_fast": float(latest["ema_fast"]),
            "ema_slow": float(latest["ema_slow"]),
            "ema_long": float(latest["ema_long"]),
            "rsi": float(latest["rsi"]),
            "vol_24h": float(latest["volatility_24h"]),
            "atr_14": float(latest["atr_14"]),
            "bb_pos": float(latest["bb_pos"]),
            "macd_line": float(latest["macd_line"]),
            "macd_hist": float(latest["macd_hist"]),
            "cl_price": float(latest["cl_close"]),
            "ng_cl_ratio_z": float(latest["ng_cl_ratio_z"]),
            "timestamp": ts_str,
        }
        return feats, None

    except Exception as e:
        return None, f"Data error: {e}"


# ---------- SIGNAL LOGIC (MARKET + WEATHER) ---------- #

def make_signal(features, weather_score: float = 0.0):
    """
    Uses:
      - Market data: EMAs, RSI, Bollinger, ATR, MACD, NG/CL ratio
      - Weather_score from US + Europe (HDD + CDD)
    to produce:
      - direction: UP / DOWN / FLAT
      - suggested BUY/SELL/NO TRADE
      - confidence adjusted by weather
    """
    last_price = features["last_price"]
    ema_fast_val = features["ema_fast"]
    ema_slow_val = features["ema_slow"]
    ema_long_val = features["ema_long"]
    rsi_val = features["rsi"]
    vol_24h = features["vol_24h"]
    atr_14 = features["atr_14"]
    bb_pos = features["bb_pos"]
    macd_line = features["macd_line"]
    macd_hist = features["macd_hist"]
    ratio_z = features["ng_cl_ratio_z"]

    # Trend conditions
    strong_up_trend = ema_fast_val > ema_slow_val > ema_long_val and macd_line > 0
    strong_down_trend = ema_fast_val < ema_slow_val < ema_long_val and macd_line < 0

    # Overbought / oversold
    overbought = (rsi_val > 70) or (bb_pos > 0.9) or (ratio_z > 1.0)
    oversold = (rsi_val < 30) or (bb_pos < 0.1) or (ratio_z < -1.0)

    # Base direction + base confidence
    if strong_up_trend and not overbought:
        direction = "UP"
        base_conf = 0.7
    elif strong_down_trend and not oversold:
        direction = "DOWN"
        base_conf = 0.7
    elif oversold and macd_hist > 0:
        direction = "UP"
        base_conf = 0.6
    elif overbought and macd_hist < 0:
        direction = "DOWN"
        base_conf = 0.6
    else:
        direction = "FLAT"
        base_conf = 0.5

    # Stop size
    if atr_14 and not math.isnan(atr_14) and atr_14 > 0:
        atr_pct = atr_14 / (last_price + 1e-9)
        stop_pct = min(max(atr_pct * 1.5, 0.0075), 0.04)
    elif vol_24h and not math.isnan(vol_24h) and vol_24h > 0:
        stop_pct = min(max(vol_24h * 2.0, 0.005), 0.03)
    else:
        stop_pct = 0.01

    tp_pct = stop_pct * 2.5

    # Trend strength adjustment
    trend_strength = abs(ema_fast_val - ema_long_val) / (last_price + 1e-9)
    conf_adj_trend = min(trend_strength * 0.8, 0.2)

    # Penalty for extreme NG/CL ratio
    ratio_penalty = min(abs(ratio_z) * 0.05, 0.15)

    confidence = base_conf + conf_adj_trend - ratio_penalty

    # Weather adjustment:
    # positive weather_score = bullish NatGas
    if weather_score != 0.0:
        if direction == "UP":
            confidence += weather_score  # bullish weather helps longs
        elif direction == "DOWN":
            confidence -= weather_score  # bullish weather hurts shorts
        else:
            confidence += 0.5 * weather_score  # FLAT gets nudged by weather

    # Clamp confidence
    confidence = float(min(max(confidence, 0.4), 0.98))

    # Price levels
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
        "weather_score": weather_score,
    }


# ---------- FLASK WEB APP ---------- #

@app.route("/", methods=["GET", "POST"])
def index():
    account_balance = None
    risk_pct = 1.0
    position_size = None
    signal = None
    last_price = None
    timestamp = None
    error_msg = None
    feats = None
    weather_info = None
    weather_error = None

    # 1) Market data
    feats, data_error = get_latest_features()
    if data_error:
        error_msg = data_error
    else:
        timestamp = feats["timestamp"]
        last_price = feats["last_price"]

    # 2) Weather data
    weather_score = 0.0
    weather_info, weather_error, ws = get_weather_summary()
    weather_score = ws

    # 3) Build signal if we have market features
    if feats is not None and error_msg is None:
        signal = make_signal(feats, weather_score)

    # 4) Position sizing from user input
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
        feats=feats,
        weather_info=weather_info,
        weather_error=weather_error,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
