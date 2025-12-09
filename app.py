import math
import json
import time
from urllib.request import urlopen

import pandas as pd
import yfinance as yf
from flask import Flask, render_template, request

app = Flask(__name__)

# ========= SIMPLE CACHE (to avoid being too slow) ========= #

CACHE_TTL_SECONDS = 300  # 5 minutes

market_cache = {"data": None, "error": None, "timestamp": 0.0}
weather_cache = {"info": None, "error": None, "score": 0.0, "timestamp": 0.0}

# ========= WEATHER REGIONS (kept small for speed) ========= #

WEATHER_LOCATIONS = [
    {"name": "US Northeast (New York)", "lat": 40.71, "lon": -74.00},
    {"name": "US Midwest (Chicago)", "lat": 41.88, "lon": -87.63},
    {"name": "US Texas (Houston)", "lat": 29.76, "lon": -95.37},
    {"name": "UK (London)", "lat": 51.50, "lon": -0.12},
    {"name": "Germany (Berlin)", "lat": 52.52, "lon": 13.40},
    {"name": "Italy (Milan)", "lat": 45.46, "lon": 9.19},
]


# ========= WEATHER HELPERS (for demand + confidence) ========= #

def fetch_weather_for_location(lat: float, lon: float):
    """
    Use Open-Meteo free API to get next 7 days of hourly temperature.
    Return: (current_temp, HDD_7d, CDD_7d) with base 18°C.
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


def compute_weather_summary():
    """
    Aggregate weather across key regions.
    Returns:
      weather_info (dict),
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
                {"name": loc["name"], "temp": temp, "hdd7": hdd, "cdd7": cdd}
            )
            total_hdd += hdd
            total_cdd += cdd
            count += 1
        except Exception:
            continue

    if count == 0:
        return None, "Weather API error (Open-Meteo).", 0.0

    avg_hdd = total_hdd / count
    avg_cdd = total_cdd / count

    # Make a rough demand score:
    heating_strength = avg_hdd / 100.0
    cooling_strength = avg_cdd / 100.0

    weather_score = 0.0

    # Colder/Hotter = bullish NatGas
    if heating_strength > 1.5:
        weather_score += 0.20
    elif heating_strength > 0.8:
        weather_score += 0.10

    if cooling_strength > 1.5:
        weather_score += 0.15
    elif cooling_strength > 0.8:
        weather_score += 0.07

    # Mild both sides = bearish
    if heating_strength < 0.4 and cooling_strength < 0.4:
        weather_score -= 0.15

    # Clamp
    weather_score = max(min(weather_score, 0.25), -0.25)

    if weather_score > 0.15:
        impact_text = "Weather: strongly supportive (high heating/cooling demand)."
    elif weather_score > 0.05:
        impact_text = "Weather: slightly supportive for NatGas."
    elif weather_score < -0.05:
        impact_text = "Weather: slightly against NatGas (mild temperatures)."
    else:
        impact_text = "Weather: roughly neutral impact on NatGas."

    weather_info = {
        "locations": locations_data,
        "avg_hdd": avg_hdd,
        "avg_cdd": avg_cdd,
        "impact_text": impact_text,
        "score": weather_score,
    }

    return weather_info, None, weather_score


def get_weather_summary_cached():
    now = time.time()
    age = now - weather_cache["timestamp"]
    if age < CACHE_TTL_SECONDS and weather_cache["info"] is not None:
        return (
            weather_cache["info"],
            weather_cache["error"],
            weather_cache["score"],
        )

    info, err, score = compute_weather_summary()
    weather_cache["info"] = info
    weather_cache["error"] = err
    weather_cache["score"] = score
    weather_cache["timestamp"] = now
    return info, err, score


# ========= INDICATORS USING MARKET DATA ========= #

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


# ========= MARKET DATA + FEATURES ========= #

def get_latest_features_fresh():
    """
    Download about 60 days of hourly NG + CL data and calculate indicators.
    """
    try:
        ng = yf.download(
            "NG=F", period="60d", interval="1h", progress=False, threads=False
        )
        cl = yf.download(
            "CL=F", period="60d", interval="1h", progress=False, threads=False
        )

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

        # Crude oil 3-day return (impact explanation)
        df["cl_ret_3d"] = cl_close.pct_change(72)  # 3 days ≈ 72 hours

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
            "cl_ret_3d": float(latest["cl_ret_3d"]),
            "timestamp": ts_str,
        }
        return feats, None

    except Exception as e:
        return None, f"Data error: {e}"


def get_latest_features_cached():
    now = time.time()
    age = now - market_cache["timestamp"]
    if age < CACHE_TTL_SECONDS and market_cache["data"] is not None:
        return market_cache["data"], market_cache["error"]

    feats, err = get_latest_features_fresh()
    market_cache["data"] = feats
    market_cache["error"] = err
    market_cache["timestamp"] = now
    return feats, err


# ========= CRUDE OIL IMPACT TEXT ========= #

def compute_crude_impact(features):
    ratio_z = features.get("ng_cl_ratio_z", 0.0)
    cl_ret_3d = features.get("cl_ret_3d", 0.0)
    cl_ret_pct = cl_ret_3d * 100.0

    if cl_ret_3d > 0.05:
        trend_label = "strong uptrend"
    elif cl_ret_3d > 0.01:
        trend_label = "mild uptrend"
    elif cl_ret_3d < -0.05:
        trend_label = "strong downtrend"
    elif cl_ret_3d < -0.01:
        trend_label = "mild downtrend"
    else:
        trend_label = "sideways / range-bound"

    if trend_label.startswith("strong up") and ratio_z < -0.5:
        impact_text = (
            "Crude is rising strongly and NatGas is cheap vs oil → supportive (bullish) backdrop."
        )
    elif trend_label.startswith("strong down") and ratio_z > 0.5:
        impact_text = (
            "Crude is falling strongly and NatGas is rich vs oil → headwind (bearish) backdrop."
        )
    elif "uptrend" in trend_label and ratio_z <= 0:
        impact_text = "Crude drifting higher; NatGas fairly priced/cheap vs oil → slightly bullish."
    elif "downtrend" in trend_label and ratio_z >= 0:
        impact_text = "Crude drifting lower; NatGas fairly priced/expensive vs oil → slightly bearish."
    else:
        impact_text = "Crude/NG spread looks mostly neutral right now."

    return {
        "cl_ret_3d_pct": cl_ret_pct,
        "trend_label": trend_label,
        "impact_text": impact_text,
    }


# ========= SIGNAL LOGIC ========= #

def make_signal(features, weather_score: float = 0.0):
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
    if weather_score != 0.0:
        if direction == "UP":
            confidence += weather_score
        elif direction == "DOWN":
            confidence -= weather_score
        else:
            confidence += 0.5 * weather_score

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


# ========= WEEKLY OUTLOOK (simple projection) ========= #

def make_weekly_forecast(signal, features):
    """
    Build a simple 5-day outlook based on:
      - current direction (UP / DOWN / FLAT)
      - confidence
      - volatility
      - trend strength (distance between EMAs)
    It does NOT magically know the future – it's a structured extrapolation.
    """
    if not signal or not features:
        return []

    direction = signal["direction"]
    base_conf = signal["confidence"]
    last_price = features["last_price"]
    ema_fast_val = features["ema_fast"]
    ema_long_val = features["ema_long"]
    vol_24h = features.get("vol_24h", 0.0)
    atr_14 = features.get("atr_14", 0.0)

    # Rough trend strength score
    trend_strength = abs(ema_fast_val - ema_long_val) / (last_price + 1e-9)
    trend_strength_score = min(trend_strength * 100, 30)  # cap

    # Volatility score
    vol_score = 0.0
    if vol_24h and not math.isnan(vol_24h):
        vol_score = min(vol_24h * 1000, 30)  # just to classify calm vs wild

    days = ["Today / next 24h", "Day 2", "Day 3", "Day 4", "Day 5"]
    outlook = []

    for i, label in enumerate(days):
        # Confidence decays a bit further into the week
        day_conf = max(min(base_conf - 0.03 * i, 0.95), 0.35)

        if direction == "FLAT":
            bias = "CHOPPY / RANGE"
        else:
            # If confidence drops a lot, we call it choppy
            if day_conf < 0.5:
                bias = "CHOPPY / RANGE"
            else:
                bias = direction

        # Text explanation
        if bias == "UP":
            note = "Bullish bias continues while current uptrend and demand factors stay intact."
        elif bias == "DOWN":
            note = "Bearish bias continues while current downtrend and demand factors stay intact."
        else:
            note = "Price likely to be more sideways / noisy; trend edge is weaker here."

        # Add a volatility tag
        if vol_score > 20:
            note += " Volatility: high – expect bigger swings."
        elif vol_score > 10:
            note += " Volatility: moderate."
        else:
            note += " Volatility: relatively calm (for NatGas)."

        outlook.append(
            {
                "label": label,
                "bias": bias,
                "confidence": day_conf,
                "trend_strength": trend_strength_score,
                "note": note,
            }
        )

    return outlook


# ========= FLASK ROUTE ========= #

@app.route("/", methods=["GET", "POST"])
def index():
    account_balance = None
    risk_pct = 1.0
    position_size = None

    signal = None
    weekly_outlook = []
    last_price = None
    timestamp = None
    error_msg = None
    feats = None
    weather_info = None
    weather_error = None
    cl_impact = None

    # 1) Market data (cached)
    feats, data_error = get_latest_features_cached()
    if data_error:
        error_msg = data_error
    else:
        timestamp = feats["timestamp"]
        last_price = feats["last_price"]

    # 2) Weather (cached)
    weather_score = 0.0
    weather_info, weather_error, ws = get_weather_summary_cached()
    weather_score = ws

    # 3) Signal + crude + weekly outlook
    if feats is not None and error_msg is None:
        cl_impact = compute_crude_impact(feats)
        signal = make_signal(feats, weather_score)
        weekly_outlook = make_weekly_forecast(signal, feats)

    # 4) Position sizing
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
        weekly_outlook=weekly_outlook,
        last_price=last_price,
        timestamp=timestamp,
        account_balance=account_balance,
        risk_pct=risk_pct,
        position_size=position_size,
        error_msg=error_msg,
        feats=feats,
        weather_info=weather_info,
        weather_error=weather_error,
        cl_impact=cl_impact,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
