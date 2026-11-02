# DriveDeck: your own in-car dashboard

DriveDeck is a head-unit style screen for the car. It runs in the browser at **`/drive`** and has no build step and no API keys.

| Screen | What it does |
| --- | --- |
| **Home** | Big speedometer (GPS), heading, trip meter, quick destinations, now playing, local weather |
| **Maps** | OpenStreetMap with follow mode, place search, routing, **spoken turn-by-turn directions with automatic rerouting**, and hand-off to Google Maps, Waze or Apple Maps |
| **Audio** | Plays music files from the device and internet radio streams. Bluetooth and steering-wheel buttons work through the Media Session API, and music is lowered while directions are spoken |
| **Phone** | Favorites and a dial pad. Calls and texts are handed to the phone through `tel:` and `sms:` links |
| **Setup** | Units, Day/Night/Auto theme (Auto follows actual daylight), voice options, keep-screen-on, passenger mode, and a **simulated drive** for testing at your desk |

**Voice commands** use the mic button: *navigate to …*, *take me home*, *how long until we arrive*, *stop navigation*, *play*, *pause*, *next*, *play &lt;station&gt;*, *call &lt;name&gt;*, *text &lt;name&gt;*, *what's the weather*, *open maps*, *night mode*.

**Driving lock:** above about 10 km/h (6 mph), typing is disabled and the screen asks you to use your voice. Passenger mode turns this off.

## Running it in the car

1. **Phone or tablet (easiest).** Deploy the app as usual (Render, `gunicorn app:app`), open `https://<your-app>/drive`, and choose *Add to Home Screen*. It launches full screen in landscape. Pair the phone to the car over Bluetooth so audio and directions play through the speakers. GPS and the microphone need **HTTPS**, which Render already provides.
2. **Raspberry Pi + touchscreen (a dedicated head unit).** Use a Pi 4 or 5 with a 7" touchscreen and a USB microphone, and run Chromium in kiosk mode:
   ```bash
   chromium-browser --kiosk --noerrdialogs --autoplay-policy=no-user-gesture-required \
     https://<your-app>/drive
   ```
   **Known gap:** Chromium on Linux can't read a USB GPS receiver directly, so on a Pi the browser usually gets no position. Speed and turn-by-turn won't work there until a small gpsd-to-browser bridge is added (not built yet). For now, a phone or tablet is the setup that works fully.

## Services used (free, no keys)

- Map tiles: OpenStreetMap (`tile.openstreetmap.org`)
- Place search: Nominatim
- Routing: OSRM public demo server (`router.project-osrm.org`)
- Weather: Open-Meteo

These public servers are fine for personal use, but they have fair-use limits and no uptime guarantee. For anything beyond one car, self-host OSRM and a tile server, or switch to a paid provider. The URLs are all in `static/drive/drive.js`.

## Files

```
app.py                         serves the /drive page
templates/drive.html           screen layout
static/drive/drive.css         styling (night + day themes, landscape + portrait)
static/drive/drive.js          all logic: GPS, nav, voice, audio, phone, safety lock
static/drive/manifest.json     installable app (start_url /drive, landscape, fullscreen)
static/drive/vendor/leaflet/   Leaflet 1.9.4 map library (BSD-2-Clause), bundled locally
```

## Why this isn't Apple CarPlay

CarPlay itself is a closed protocol (iAP2 plus streamed H.264 video). The car side must hold an **Apple MFi authentication chip**, and the iPhone refuses to connect without it. Cloning that protocol and getting past the authentication isn't a practical or legal DIY project. DriveDeck is a separate, original dashboard instead, and you control everything on it.

If you want real CarPlay on a DIY screen as well, the legitimate route is a licensed **USB CarPlay adapter** (for example a Carlinkit dongle, which contains the MFi chip) together with an open-source host app such as *react-carplay* / *node-carplay* on the Pi. That runs Apple's own CarPlay UI, so you can't customise it the way you can DriveDeck.

> Safety: set up destinations, favorites and stations before you drive. Use voice while moving.
