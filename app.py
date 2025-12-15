import math
import time
import requests
import pandas as pd
import yfinance as yf
from flask import Flask, render_template, request

app = Flask(__name__)

# =====================================================
# CACHE
# =====================================================
CACHE_TTL_SECONDS = 300

market_cache = {"data": None, "error": None, "timestamp": 0.0}
weather_cache = {"info": None, "error": None, "score": 0.0, "timestamp": 0.0}
lng_cache = {"info": None, "error": None, "score": 0.0, "timestamp": 0.0}

# =====================================================
# WEATHER (Open-Meteo)
# =====================================================
WEATHER_LOCATIONS = [
    {"name": "US Northeast (New York)", "lat": 40.71, "lon": -74.00},
    {"name": "US Midwest (Chicago)", "lat": 41.88, "lon": -87.63},
    {"name": "Texas (Houston)", "lat": 29.76, "lon": -95.37},
    {"name": "UK (London)", "lat": 51.50, "lon": -0.12},
    {"name": "Germany (Berlin)", "lat": 52.52, "lon": 13.40},
]

def fetch_weather(lat, lon):
    base = 18.0
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m&forecast_days=7"
    )
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    temps = r.json()["hourly"]["temperature_2m"]
    hdd = sum(max(0, base - t) for t in temps)
    cdd = sum(max(0, t - base) for t in temps)
    return temps[0], hdd, cdd

def get_weather():
    now = time.time()
    if now - weather_cache["timestamp"] < CACHE_TTL_SECONDS:
        return weather_cache["info"], weather_cache["error"], weather_cache["score"]

    try:
        rows, hdd, cdd = [], 0, 0
        for l in WEATHER_LOCATIONS:
            t, h, c = fetch_weather(l["lat"], l["lon"])
            rows.append({"name": l["name"], "temp": t, "hdd7": h, "cdd7": c})
            hdd += h
            cdd += c

        avg_hdd = hdd / len(rows)
        avg_cdd = cdd / len(rows)

        score = 0.0
        if avg_hdd > 120: score += 0.15
        if avg_cdd > 120: score += 0.10
        if avg_hdd < 40 and avg_cdd < 40: score -= 0.15
        score = max(min(score, 0.25), -0.25)

        info = {
            "locations": rows,
            "avg_hdd": avg_hdd,
            "avg_cdd": avg_cdd,
            "impact_text": "Weather demand supports NatGas." if score > 0 else "Weather mostly neutral.",
            "score": score,
        }
        weather_cache.update({"info": info, "error": None, "score": score, "timestamp": now})
        return info, None, score
    except Exception as e:
        return None, str(e), 0.0

# =====================================================
# LNG (US EIA – Weekly)
# =====================================================
def get_lng_data():
    """
    EIA LNG exports (Bcf/week)
    """
    now = time.time()
    if now - lng_cache["timestamp"] < 3600:
        return lng_cache["info"], lng_cache["error"], lng_cache["score"]

    try:
        url = (
            "https://api.eia.gov/v2/natural-gas/exports/data/"
            "?frequency=weekly"
            "&data[0]=value"
            "&sort[0][column]=period"
            "&sort[0][direction]=desc"
            "&offset=0"
            "&length=10"
            "&api_key=DEMO_KEY"
        )
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()["response"]["data"]

        df = pd.DataFrame(data)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna()

        latest = df.iloc[0]["value"]
        prev = df.iloc[1]["value"]
        avg = df["value"].mean()

        delta = latest - prev
        score = 0.0

        if latest > avg * 1.05: score += 0.15
        if delta > 0.2: score += 0.10
        if latest < avg * 0.95: score -= 0.15

        score = max(min(score, 0.25), -0.25)

        info = {
            "latest": latest,
            "previous": prev,
            "average": avg,
            "delta": delta,
            "impact_text": (
                "LNG exports increased → bullish NatGas demand."
                if score > 0 else
                "LNG exports stable / lower → neutral to bearish."
            )
        }

        lng_cache.update({"info": info, "error": None, "score": score, "timestamp": now})
        return info, None, score

    except Exception as e:
        return None, str(e), 0.0

# =====================================================
# MARKET DATA (NG + CL)
# =====================================================
def get_market():
    try:
        ng = yf.download("NG=F", period="60d", interval="1h", progress=False)
        cl = yf.download("CL=F", period="60d", interval="1h", progress=False)

        df = pd.DataFrame(index=ng.index)
        df["ng"] = ng["Close"]
        df["cl"] = cl["Close"]
        df = df.dropna()

        close = df["ng"]
        df["ema_fast"] = close.ewm(span=10).mean()
        df["ema_slow"] = close.ewm(span=30).mean()
        df["rsi"] = 100 - (100 / (1 + close.diff().clip(lower=0).rolling(14).mean() /
                                  (-close.diff().clip(upper=0).rolling(14).mean() + 1e-9)))

        df = df.dropna()
        last = df.iloc[-1]

        return {
            "price": float(last["ng"]),
            "ema_fast": float(last["ema_fast"]),
            "ema_slow": float(last["ema_slow"]),
            "rsi": float(last["rsi"]),
        }, None

    except Exception as e:
        return None, str(e)

# =====================================================
# SIGNAL
# =====================================================
def make_signal(market, weather_score, lng_score):
    direction = "FLAT"
    confidence = 0.5

    if market["ema_fast"] > market["ema_slow"] and market["rsi"] < 70:
        direction = "UP"
        confidence = 0.65
    elif market["ema_fast"] < market["ema_slow"] and market["rsi"] > 30:
        direction = "DOWN"
        confidence = 0.65

    confidence += weather_score + lng_score
    confidence = max(min(confidence, 0.95), 0.4)

    return {
        "direction": direction,
        "confidence": confidence,
    }

# =====================================================
# ROUTE
# =====================================================
@app.route("/", methods=["GET", "POST"])
def index():
    market, err = get_market()
    weather, _, weather_score = get_weather()
    lng, _, lng_score = get_lng_data()

    signal = None
    if market:
        signal = make_signal(market, weather_score, lng_score)

    return render_template(
        "index.html",
        market=market,
        weather=weather,
        lng=lng,
        signal=signal,
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
