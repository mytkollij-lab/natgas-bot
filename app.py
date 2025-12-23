import math
import time
import json
import requests

import pandas as pd
import yfinance as yf
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# ---------------------------------------------------------------------
# SIMPLE CACHE (faster)
# ---------------------------------------------------------------------
CACHE_TTL_SECONDS = 300  # 5 minutes

market_cache = {"data": None, "error": None, "timestamp": 0.0}
weather_cache = {"info": None, "error": None, "score": 0.0, "timestamp": 0.0}
chart_cache = {"data": None, "error": None, "timestamp": 0.0}

# ---------------------------------------------------------------------
# WEATHER REGIONS (US + Europe)
# ---------------------------------------------------------------------
WEATHER_LOCATIONS = [
    {"name": "US Northeast (New York)", "lat": 40.71, "lon": -74.00},
    {"name": "US Midwest (Chicago)", "lat": 41.88, "lon": -87.63},
    {"name": "US Texas (Houston)", "lat": 29.76, "lon": -95.37},
    {"name": "UK (London)", "lat": 51.50, "lon": -0.12},
    {"name": "Germany (Berlin)", "lat": 52.52, "lon": 13.40},
    {"name": "Italy (Milan)", "lat": 45.46, "lon": 9.19},
]

# ---------------------------------------------------------------------
# WEATHER HELPERS
# ---------------------------------------------------------------------
def fetch_weather_for_location(lat: float, lon: float):
    """
    Open-Meteo free API, next 7 days hourly temperature.
    Returns: (current_temp, HDD_7d, CDD_7d) base 18°C.
    """
    base_temp = 18.0
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m&forecast_days=7"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

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

    heating_strength = avg_hdd / 100.0
    cooling_strength = avg_cdd / 100.0
    weather_score = 0.0

    if heating_strength > 1.5:
        weather_score += 0.20
    elif heating_strength > 0.8:
        weather_score += 0.10

    if cooling_strength > 1.5:
        weather_score += 0.15
    elif cooling_strength > 0.8:
        weather_score += 0.07

    if heating_strength < 0.4 and cooling_strength < 0.4:
        weather_score -= 0.15

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
        "score": weather_score,  # [-0.25, +0.25]
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


# ---------------------------------------------------------------------
# INDICATORS
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# MARKET DATA + FEATURES
# ---------------------------------------------------------------------
def get_latest_features_fresh():
    """
    60d of 1h NG + CL, indicators for signal engine.
    """
    try:
        ng = yf.download("NG=F", period="60d", interval="1h", progress=False, threads=False)
        cl = yf.download("CL=F", period="60d", interval="1h", progress=False, threads=False)

        if ng is None or ng.empty:
            return None, "No NatGas data received from Yahoo Finance (NG=F)."
        if cl is None or cl.empty:
            return None, "No Crude Oil data received from Yahoo Finance (CL=F)."

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

        # NG/CL ratio z-score
        df["ng_cl_ratio"] = close / cl_close
        ratio = df["ng_cl_ratio"]
        ratio_ma = ratio.rolling(50).mean()
        ratio_std = ratio.rolling(50).std()
        df["ng_cl_ratio_z"] = (ratio - ratio_ma) / (ratio_std + 1e-9)

        # crude 3-day return
        df["cl_ret_3d"] = cl_close.pct_change(72)  # 72 hours ≈ 3 days

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


# ---------------------------------------------------------------------
# CHART DATA (graphs)
# ---------------------------------------------------------------------
def get_chart_data_fresh():
    try:
        ng = yf.download("NG=F", period="10d", interval="1h", progress=False, threads=False)
        cl = yf.download("CL=F", period="10d", interval="1h", progress=False, threads=False)

        if ng is None or ng.empty:
            return None, "No NatGas chart data received from Yahoo Finance (NG=F)."
        if cl is None or cl.empty:
            return None, "No Crude chart data received from Yahoo Finance (CL=F)."

        df = pd.DataFrame(index=ng.index)
        df["ng_close"] = ng["Close"]
        df["cl_close"] = cl["Close"]
        df = df.dropna()
        if df.empty:
            return None, "Chart data empty after cleaning."

        df["ema10"] = ema(df["ng_close"], 10)
        df["ema30"] = ema(df["ng_close"], 30)
        df["ema50"] = ema(df["ng_close"], 50)
        df["rsi14"] = rsi(df["ng_close"], 14)
        _, _, macd_hist = macd(df["ng_close"])
        df["macd_hist"] = macd_hist

        df["ng_cl_ratio"] = df["ng_close"] / df["cl_close"]
        ratio = df["ng_cl_ratio"]
        ratio_ma = ratio.rolling(50).mean()
        ratio_std = ratio.rolling(50).std()
        df["ratio_z"] = (ratio - ratio_ma) / (ratio_std + 1e-9)

        df = df.dropna()
        if df.empty:
            return None, "Not enough chart candles after indicator warm-up."

        df = df.tail(180)  # ~7 days hourly

        labels = []
        for ts in df.index:
            try:
                labels.append(ts.tz_convert("UTC").strftime("%Y-%m-%d %H:%M"))
            except Exception:
                labels.append(str(ts))

        out = {
            "labels": labels,
            "ng_close": [float(x) for x in df["ng_close"].values],
            "ema10": [float(x) for x in df["ema10"].values],
            "ema30": [float(x) for x in df["ema30"].values],
            "ema50": [float(x) for x in df["ema50"].values],
            "rsi14": [float(x) for x in df["rsi14"].values],
            "macd_hist": [float(x) for x in df["macd_hist"].values],
            "ratio_z": [float(x) for x in df["ratio_z"].values],
        }
        return out, None
    except Exception as e:
        return None, f"Chart data error: {e}"


def get_chart_data_cached():
    now = time.time()
    age = now - chart_cache["timestamp"]
    if age < CACHE_TTL_SECONDS and chart_cache["data"] is not None:
        return chart_cache["data"], chart_cache["error"]

    data, err = get_chart_data_fresh()
    chart_cache["data"] = data
    chart_cache["error"] = err
    chart_cache["timestamp"] = now
    return data, err


# ---------------------------------------------------------------------
# CRUDE OIL IMPACT + SCORE
# ---------------------------------------------------------------------
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
        impact_text = "Crude rising strongly and NatGas cheap vs oil → supportive (bullish)."
        crude_score = +0.18
    elif trend_label.startswith("strong down") and ratio_z > 0.5:
        impact_text = "Crude falling strongly and NatGas rich vs oil → headwind (bearish)."
        crude_score = -0.18
    elif "uptrend" in trend_label and ratio_z <= 0:
        impact_text = "Crude drifting higher; NatGas cheap/fair vs oil → slightly bullish."
        crude_score = +0.08
    elif "downtrend" in trend_label and ratio_z >= 0:
        impact_text = "Crude drifting lower; NatGas expensive/fair vs oil → slightly bearish."
        crude_score = -0.08
    else:
        impact_text = "Crude/NG spread looks mostly neutral."
        crude_score = 0.0

    return {
        "cl_ret_3d_pct": cl_ret_pct,
        "trend_label": trend_label,
        "impact_text": impact_text,
        "score": crude_score,  # approx [-0.25,+0.25]
    }


# ---------------------------------------------------------------------
# DECISION ENGINE: Regime / Bias / Move Quality / Action / Plan
# ---------------------------------------------------------------------
def _clamp(x, lo, hi):
    return float(max(lo, min(x, hi)))


def _eta_hours(distance: float, atr_per_hour: float):
    if atr_per_hour is None or atr_per_hour <= 0 or distance <= 0:
        return None
    hours = distance / atr_per_hour
    return float(max(0.5, min(hours, 240)))


def _market_regime(feats):
    ef, es, el = feats["ema_fast"], feats["ema_slow"], feats["ema_long"]
    macd_line = feats["macd_line"]
    bb_pos = feats["bb_pos"]
    vol = feats["vol_24h"]

    trending_up = (ef > es > el) and (macd_line > 0)
    trending_down = (ef < es < el) and (macd_line < 0)

    # choppy heuristic: EMAs not ordered OR near-mid band OR low vol
    ema_ordered = (ef > es > el) or (ef < es < el)
    low_vol = (vol is not None and not math.isnan(vol) and vol < 0.01)

    if trending_up:
        regime = "TRENDING UP"
        why = [
            "EMA10 > EMA30 > EMA50 (bullish structure)",
            "MACD line above 0 (positive momentum)",
        ]
    elif trending_down:
        regime = "TRENDING DOWN"
        why = [
            "EMA10 < EMA30 < EMA50 (bearish structure)",
            "MACD line below 0 (negative momentum)",
        ]
    else:
        regime = "RANGE / CHOP"
        why = []
        if not ema_ordered:
            why.append("EMAs not cleanly stacked (structure mixed)")
        else:
            why.append("Trend is weak despite EMA ordering")
        if 0.35 <= bb_pos <= 0.65:
            why.append("Price sitting mid-Bollinger range (mean-revert zone)")
        if low_vol:
            why.append("Volatility low (breakouts less reliable)")

        if not why:
            why = ["Mixed signals → no clean trend regime"]

    return {"label": regime, "why": why}


def _daily_bias(feats, weather_score, crude_score):
    """
    Bias = directional lean even if we don't trade immediately.
    """
    ef, es, el = feats["ema_fast"], feats["ema_slow"], feats["ema_long"]
    macd_line = feats["macd_line"]
    rsi_val = feats["rsi"]
    ratio_z = feats["ng_cl_ratio_z"]

    score = 0.0
    why = []

    # Trend contribution
    if ef > es > el and macd_line > 0:
        score += 0.22
        why.append("Trend/momentum aligned bullish (EMA stack + MACD>0)")
    elif ef < es < el and macd_line < 0:
        score -= 0.22
        why.append("Trend/momentum aligned bearish (EMA stack + MACD<0)")
    else:
        why.append("Trend alignment mixed (no strong directional structure)")

    # Stretch contribution (avoid chasing)
    if rsi_val > 72:
        score -= 0.08
        why.append("RSI elevated (risk of pullback / chase)")
    elif rsi_val < 28:
        score += 0.08
        why.append("RSI depressed (bounce risk / mean reversion)")

    # NG vs crude relative valuation (z)
    if ratio_z > 1.0:
        score -= 0.06
        why.append("NatGas expensive vs crude (ratio z-score high) → headwind")
    elif ratio_z < -1.0:
        score += 0.06
        why.append("NatGas cheap vs crude (ratio z-score low) → supportive")

    # Weather + crude (already bounded)
    score += _clamp(weather_score, -0.25, 0.25) * 0.8
    if abs(weather_score) > 0.02:
        why.append(f"Weather factor applied ({weather_score:+.2f})")

    score += _clamp(crude_score, -0.25, 0.25) * 0.8
    if abs(crude_score) > 0.02:
        why.append(f"Crude factor applied ({crude_score:+.2f})")

    # Decide label
    if score >= 0.10:
        label = "BULLISH"
    elif score <= -0.10:
        label = "BEARISH"
    else:
        label = "NEUTRAL"

    return {"label": label, "score": _clamp(score, -0.35, 0.35), "why": why}


def _move_quality(feats):
    """
    Quality = how likely the current push has follow-through.
    """
    rsi_val = feats["rsi"]
    macd_hist = feats["macd_hist"]
    bb_pos = feats["bb_pos"]
    vol = feats["vol_24h"]

    why = []
    score = 0.0

    # MACD histogram
    if macd_hist > 0:
        score += 0.10
        why.append("MACD histogram above 0 (bullish momentum)")
    else:
        score -= 0.10
        why.append("MACD histogram below 0 (bearish momentum)")

    # Volatility confirmation (if too low, breakouts fail)
    if vol is not None and not math.isnan(vol):
        if vol > 0.02:
            score += 0.06
            why.append("Volatility high (moves can travel)")
        elif vol < 0.01:
            score -= 0.06
            why.append("Volatility low (moves often stall)")

    # Bollinger position (stretch)
    if bb_pos > 0.92:
        score -= 0.06
        why.append("Near upper Bollinger edge (risk of snapback)")
    elif bb_pos < 0.08:
        score += 0.06
        why.append("Near lower Bollinger edge (bounce risk)")

    # RSI sanity
    if 45 <= rsi_val <= 65:
        score += 0.04
        why.append("RSI in healthy trend zone (not too stretched)")
    elif rsi_val > 75:
        score -= 0.05
        why.append("RSI very high (overextended)")
    elif rsi_val < 25:
        score += 0.05
        why.append("RSI very low (mean reversion possible)")

    # Label
    if score >= 0.10:
        label = "REAL MOVE"
    elif score <= -0.10:
        label = "FAKE MOVE / FRAGILE"
    else:
        label = "UNCLEAR"

    return {"label": label, "score": _clamp(score, -0.25, 0.25), "why": why}


def _best_setup_and_invalidation(feats, bias_label):
    """
    (Your request: BOTH)
    - Best setup to wait for
    - Invalidation scenario (what flips bias)
    """
    lp = feats["last_price"]
    ef, es, el = feats["ema_fast"], feats["ema_slow"], feats["ema_long"]
    macd_line = feats["macd_line"]
    rsi_val = feats["rsi"]

    setup = []
    invalid = []

    if bias_label == "BULLISH":
        setup = [
            f"Wait for pullback toward EMA30 (~{es:.4f}) and hold above it",
            "Or wait for break above recent highs with MACD histogram rising",
            "Prefer entries when RSI is 45–65 (not stretched)",
        ]
        invalid = [
            "EMA10 drops below EMA30 (structure weakening)",
            "MACD line flips below 0 (momentum reversal)",
            "Price loses EMA50 (trend breaks down)",
        ]
    elif bias_label == "BEARISH":
        setup = [
            f"Wait for bounce toward EMA30 (~{es:.4f}) and rejection below it",
            "Or breakdown below recent lows with MACD histogram falling",
            "Prefer entries when RSI is 35–55 (not deeply oversold)",
        ]
        invalid = [
            "EMA10 crosses above EMA30 (structure improving)",
            "MACD line flips above 0 (momentum reversal)",
            "Price reclaims EMA50 (trend breaks up)",
        ]
    else:
        setup = [
            "Wait for a clean EMA stack (EMA10>EMA30>EMA50 or the opposite)",
            "Or wait for volatility expansion + momentum confirmation (MACD hist strengthens)",
            "Avoid trading mid-Bollinger range (chop zone)",
        ]
        invalid = [
            "Bias flips only once EMAs stack cleanly AND MACD confirms (above/below 0)",
            "Until then, treat as range/chop",
        ]

    # Make it more human
    setup.insert(0, f"Current price: {lp:.4f}. Don’t chase — wait for the cleaner entry.")
    return {"setup": setup, "invalidation": invalid}


def compute_action_and_plan(feats, weather_score, crude_impact):
    """
    Returns:
      market_regime, daily_bias, move_quality, action_now, plan (if trade), wait_guidance (if wait)
      plus signal levels, confidence, ETA, meters
    """
    lp = feats["last_price"]
    ef, es, el = feats["ema_fast"], feats["ema_slow"], feats["ema_long"]
    rsi_val = feats["rsi"]
    atr_14 = feats["atr_14"]
    vol_24h = feats["vol_24h"]
    bb_pos = feats["bb_pos"]
    macd_line = feats["macd_line"]
    macd_hist = feats["macd_hist"]
    ratio_z = feats["ng_cl_ratio_z"]

    market_regime = _market_regime(feats)
    daily_bias = _daily_bias(feats, weather_score, crude_impact["score"])
    move_quality = _move_quality(feats)

    # Trend strength meter (0–100)
    trend_strength = abs(ef - el) / (lp + 1e-9)
    trend_strength_meter = _clamp(trend_strength * 100.0, 0, 100)

    # Overstretch flags
    overbought = (rsi_val > 70) or (bb_pos > 0.9) or (ratio_z > 1.0)
    oversold = (rsi_val < 30) or (bb_pos < 0.1) or (ratio_z < -1.0)

    # Base confidence from bias + move quality + regime
    conf = 0.50
    conf += abs(daily_bias["score"]) * 0.9
    conf += move_quality["score"] * 0.6

    if market_regime["label"] == "TRENDING UP" and daily_bias["label"] == "BULLISH":
        conf += 0.08
    if market_regime["label"] == "TRENDING DOWN" and daily_bias["label"] == "BEARISH":
        conf += 0.08
    if market_regime["label"] == "RANGE / CHOP":
        conf -= 0.08

    # Penalties for chasing
    if daily_bias["label"] == "BULLISH" and overbought:
        conf -= 0.10
    if daily_bias["label"] == "BEARISH" and oversold:
        conf -= 0.10

    conf = _clamp(conf, 0.40, 0.98)

    # Decide action
    # Rule: if confidence low OR regime chop OR move quality unclear -> WAIT
    if conf < 0.58 or market_regime["label"] == "RANGE / CHOP" or move_quality["label"] == "UNCLEAR":
        action = "WAIT (NO TRADE)"
    else:
        if daily_bias["label"] == "BULLISH":
            action = "BUY"
        elif daily_bias["label"] == "BEARISH":
            action = "SELL"
        else:
            action = "WAIT (NO TRADE)"

    # Stop/TP based on ATR (preferred), else vol
    if atr_14 and not math.isnan(atr_14) and atr_14 > 0:
        atr_pct = atr_14 / (lp + 1e-9)
        stop_pct = min(max(atr_pct * 1.5, 0.0075), 0.04)
    elif vol_24h and not math.isnan(vol_24h) and vol_24h > 0:
        stop_pct = min(max(vol_24h * 2.0, 0.005), 0.03)
    else:
        stop_pct = 0.01

    tp_pct = stop_pct * 2.5

    if action == "BUY":
        stop_loss = lp * (1 - stop_pct)
        take_profit = lp * (1 + tp_pct)
    elif action == "SELL":
        stop_loss = lp * (1 + stop_pct)
        take_profit = lp * (1 - tp_pct)
    else:
        stop_loss = lp * (1 - stop_pct)
        take_profit = lp * (1 + tp_pct)

    # ETA (hours) using ATR as “typical 1h move”
    dist_tp = abs(take_profit - lp)
    dist_sl = abs(stop_loss - lp)
    eta_tp_h = _eta_hours(dist_tp, atr_14 if (atr_14 and not math.isnan(atr_14)) else None)
    eta_sl_h = _eta_hours(dist_sl, atr_14 if (atr_14 and not math.isnan(atr_14)) else None)

    # Execution plan (only if trade)
    plan = None
    if action in ("BUY", "SELL"):
        entry_type = "trend continuation"
        if overbought and action == "BUY":
            entry_type = "pullback entry (avoid chasing overbought)"
        if oversold and action == "SELL":
            entry_type = "pullback entry (avoid chasing oversold)"

        invalid = _best_setup_and_invalidation(feats, daily_bias["label"])["invalidation"]

        plan = {
            "entry_type": entry_type,
            "rr": 2.5,
            "invalidation": invalid,
            "notes": [
                "If price action is choppy/whipsaw → reduce size or skip.",
                "NatGas reacts hard to news (EIA, weather shifts).",
            ],
        }

    # WAIT guidance (BOTH items)
    wait_guidance = None
    if action == "WAIT (NO TRADE)":
        wait_guidance = _best_setup_and_invalidation(feats, daily_bias["label"])

    # Score contributions (human readable)
    contrib = {
        "trend": 0.22 if (ef > es > el and macd_line > 0) else (-0.22 if (ef < es < el and macd_line < 0) else 0.0),
        "momentum": 0.10 if macd_hist > 0 else -0.10,
        "weather": _clamp(weather_score, -0.25, 0.25),
        "crude": _clamp(crude_impact["score"], -0.25, 0.25),
        "stretch": (-0.10 if overbought else (0.10 if oversold else 0.0)),
        "volatility": (0.06 if (vol_24h and not math.isnan(vol_24h) and vol_24h > 0.02) else (-0.06 if (vol_24h and not math.isnan(vol_24h) and vol_24h < 0.01) else 0.0)),
    }

    return {
        "market_regime": market_regime,
        "daily_bias": daily_bias,
        "move_quality": move_quality,
        "action": action,
        "confidence": conf,
        "trend_strength": trend_strength_meter,
        "stop_pct": stop_pct,
        "tp_pct": tp_pct,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "eta_tp_hours": eta_tp_h,
        "eta_sl_hours": eta_sl_h,
        "plan": plan,
        "wait_guidance": wait_guidance,
        "contrib": contrib,
    }


# ---------------------------------------------------------------------
# WEEKLY OUTLOOK
# ---------------------------------------------------------------------
def make_weekly_forecast(action_pack):
    if not action_pack:
        return []
    base_conf = action_pack["confidence"]
    direction = "CHOPPY"
    if action_pack["action"] == "BUY":
        direction = "UP"
    elif action_pack["action"] == "SELL":
        direction = "DOWN"

    days = ["Today / next 24h", "Day 2", "Day 3", "Day 4", "Day 5"]
    out = []
    for i, label in enumerate(days):
        day_conf = max(min(base_conf - 0.03 * i, 0.95), 0.35)

        if direction == "CHOPPY":
            bias = "CHOPPY"
        else:
            bias = "CHOPPY" if day_conf < 0.55 else direction

        if bias == "UP":
            note = "Bullish bias if structure + momentum stay intact."
        elif bias == "DOWN":
            note = "Bearish bias if structure + momentum stay intact."
        else:
            note = "More sideways/noisy likely; edge weaker."

        out.append(
            {"label": label, "bias": bias, "confidence": day_conf, "note": note}
        )
    return out


# ---------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------
@app.route("/refresh", methods=["POST", "GET"])
def refresh():
    # hard reset caches
    market_cache.update({"data": None, "error": None, "timestamp": 0.0})
    weather_cache.update({"info": None, "error": None, "score": 0.0, "timestamp": 0.0})
    chart_cache.update({"data": None, "error": None, "timestamp": 0.0})
    return redirect(url_for("index"))


@app.route("/", methods=["GET", "POST"])
def index():
    account_balance = None
    risk_pct = 1.0
    position_size = None

    last_price = None
    timestamp = None
    error_msg = None
    feats = None

    weather_info = None
    weather_error = None
    weather_score = 0.0

    chart_data = None
    chart_error = None

    crude_impact = None
    decision = None
    weekly_outlook = []

    # Market data
    feats, data_error = get_latest_features_cached()
    if data_error:
        error_msg = data_error
    else:
        timestamp = feats["timestamp"]
        last_price = feats["last_price"]

    # Weather
    weather_info, weather_error, weather_score = get_weather_summary_cached()

    # Crude impact + decision
    if feats is not None and error_msg is None:
        crude_impact = compute_crude_impact(feats)
        decision = compute_action_and_plan(feats, weather_score, crude_impact)
        weekly_outlook = make_weekly_forecast(decision)

    # Charts
    chart_data, chart_error = get_chart_data_cached()

    # Position sizing
    if request.method == "POST":
        try:
            account_balance = float(request.form.get("account_balance", "0"))
        except ValueError:
            account_balance = 0.0

        try:
            risk_pct = float(request.form.get("risk_pct", "1.0"))
        except ValueError:
            risk_pct = 1.0

        if decision and account_balance and last_price:
            risk_amount = account_balance * (risk_pct / 100.0)
            stop_distance = abs(last_price - decision["stop_loss"])
            if stop_distance > 0:
                position_size = risk_amount / stop_distance

    return render_template(
        "index.html",
        last_price=last_price,
        timestamp=timestamp,
        error_msg=error_msg,
        feats=feats,
        weather_info=weather_info,
        weather_error=weather_error,
        crude_impact=crude_impact,
        decision=decision,
        weekly_outlook=weekly_outlook,
        account_balance=account_balance,
        risk_pct=risk_pct,
        position_size=position_size,
        chart_data_json=json.dumps(chart_data) if chart_data else None,
        chart_error=chart_error,
        data_note="Data source: Yahoo Finance futures (NG=F). This can differ from Trading212 CFD pricing.",
    )


if __name__ == "__main__":
    # Local dev; on Render use gunicorn.
    app.run(host="0.0.0.0", port=8000, debug=True)
