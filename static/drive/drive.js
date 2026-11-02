/* DriveDeck — an in-car head unit that runs in the browser.
 *
 * Everything here is plain browser JavaScript (no build step):
 *   gps      – Geolocation, with derived speed/heading and a trip meter
 *   sim      – fake GPS that drives along the active route (desk testing)
 *   nav      – place search (Nominatim), routing (OSRM), turn-by-turn + rerouting
 *   mapView  – Leaflet map with follow mode
 *   media    – local files + internet radio, wired to Media Session (steering-wheel keys)
 *   phone    – favorites, dial pad, tel:/sms: hand-off to the phone
 *   markets  – the NatGas bot signal from this Flask app
 *   voice    – speech recognition + a small command parser, spoken replies
 *   safety   – locks typing while the car is moving
 */
(() => {
  'use strict';

  const CFG = window.DRIVEDECK || {};
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
  const uid = () => Math.random().toString(36).slice(2, 10);
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const lowerFirst = (s) => (s ? s.charAt(0).toLowerCase() + s.slice(1) : s);
  const icon = (name) => `<svg class="ic"><use href="#i-${name}"/></svg>`;

  // ------------------------------------------------------------------ storage
  const STORE_KEY = 'drivedeck:v1';
  const guessUnits = () => (/-(US|GB|LR|MM)$/i.test(navigator.language || '') ? 'imperial' : 'metric');
  const DEFAULTS = {
    settings: { units: guessUnits(), theme: 'auto', voiceGuidance: true, wakeLock: true, passenger: false, simulate: false, speechRate: 1 },
    favorites: [],
    stations: [],
    places: { home: null, work: null },
    recents: [],
  };

  function load() {
    const s = JSON.parse(JSON.stringify(DEFAULTS));
    let raw = null;
    try { raw = JSON.parse(localStorage.getItem(STORE_KEY) || 'null'); } catch (e) { raw = null; }
    if (raw && typeof raw === 'object') {
      Object.assign(s.settings, raw.settings || {});
      for (const k of ['favorites', 'stations', 'recents']) if (Array.isArray(raw[k])) s[k] = raw[k];
      if (raw.places && typeof raw.places === 'object') Object.assign(s.places, raw.places);
    }
    s.settings.simulate = false; // never resume a simulated drive on reload
    return s;
  }
  const state = load();
  function save() {
    try { localStorage.setItem(STORE_KEY, JSON.stringify(state)); } catch (e) { /* private mode etc. */ }
  }

  // ------------------------------------------------------------------ events
  const bus = {
    h: {},
    on(e, f) { (this.h[e] = this.h[e] || []).push(f); },
    emit(e, d) { (this.h[e] || []).forEach((f) => { try { f(d); } catch (err) { console.error(err); } }); },
  };

  // ------------------------------------------------------------------ geometry
  const R = 6371008.8;
  const toRad = (d) => (d * Math.PI) / 180;
  const toDeg = (r) => (r * 180) / Math.PI;
  function haversine(a, b) {
    const dLat = toRad(b[0] - a[0]);
    const dLon = toRad(b[1] - a[1]);
    const h = Math.sin(dLat / 2) ** 2 + Math.cos(toRad(a[0])) * Math.cos(toRad(b[0])) * Math.sin(dLon / 2) ** 2;
    return 2 * R * Math.asin(Math.min(1, Math.sqrt(h)));
  }
  function bearing(a, b) {
    const p1 = toRad(a[0]);
    const p2 = toRad(b[0]);
    const dl = toRad(b[1] - a[1]);
    const y = Math.sin(dl) * Math.cos(p2);
    const x = Math.cos(p1) * Math.sin(p2) - Math.sin(p1) * Math.cos(p2) * Math.cos(dl);
    return (toDeg(Math.atan2(y, x)) + 360) % 360;
  }
  // Project point p onto segment a–b (local flat-earth approximation, fine at street scale).
  function projectOnSegment(p, a, b) {
    const kx = Math.cos(toRad(p[0])) * R * (Math.PI / 180);
    const ky = R * (Math.PI / 180);
    const ax = (a[1] - p[1]) * kx; const ay = (a[0] - p[0]) * ky;
    const bx = (b[1] - p[1]) * kx; const by = (b[0] - p[0]) * ky;
    const dx = bx - ax; const dy = by - ay;
    const len2 = dx * dx + dy * dy;
    const t = len2 > 0 ? clamp(-(ax * dx + ay * dy) / len2, 0, 1) : 0;
    return { t, d: Math.hypot(ax + t * dx, ay + t * dy) };
  }
  const COMPASS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
  const compass = (deg) => COMPASS[Math.round((((deg % 360) + 360) % 360) / 45) % 8];
  const COMPASS_WORDS = { N: 'north', NE: 'northeast', E: 'east', SE: 'southeast', S: 'south', SW: 'southwest', W: 'west', NW: 'northwest' };

  // ------------------------------------------------------------------ formatting
  const imperial = () => state.settings.units === 'imperial';
  const fmt = {
    speed(ms) {
      if (ms == null || !Number.isFinite(ms)) return '0';
      return String(Math.round(Math.max(0, ms) * (imperial() ? 2.23694 : 3.6)));
    },
    speedUnit: () => (imperial() ? 'mph' : 'km/h'),
    dist(m, spoken = false) {
      if (m == null || !Number.isFinite(m)) return '–';
      m = Math.max(0, m);
      if (imperial()) {
        const mi = m / 1609.344;
        if (mi < 0.1) {
          const ft = Math.max(50, Math.round((m * 3.28084) / 50) * 50);
          return spoken ? `${ft} feet` : `${ft} ft`;
        }
        const v = mi < 10 ? Math.round(mi * 10) / 10 : Math.round(mi);
        return spoken ? `${v} ${v === 1 ? 'mile' : 'miles'}` : `${v} mi`;
      }
      if (m < 950) {
        const r = m < 300 ? Math.max(10, Math.round(m / 10) * 10) : Math.round(m / 50) * 50;
        return spoken ? `${r} meters` : `${r} m`;
      }
      const km = m / 1000;
      const v = km < 10 ? Math.round(km * 10) / 10 : Math.round(km);
      return spoken ? `${v} ${v === 1 ? 'kilometer' : 'kilometers'}` : `${v} km`;
    },
    time: (d) => d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }),
    duration(sec) {
      const m = Math.max(1, Math.round(sec / 60));
      if (m < 60) return `${m} min`;
      return `${Math.floor(m / 60)} h ${m % 60} min`;
    },
    durationSpoken(sec) {
      const m = Math.max(1, Math.round(sec / 60));
      if (m < 60) return `${m} ${m === 1 ? 'minute' : 'minutes'}`;
      const h = Math.floor(m / 60); const r = m % 60;
      return `${h} ${h === 1 ? 'hour' : 'hours'}${r ? ` ${r} minutes` : ''}`;
    },
    accuracy(m) {
      if (m == null || !Number.isFinite(m)) return '–';
      return imperial() ? `${Math.max(1, Math.round(m * 3.28084))} ft` : `${Math.max(1, Math.round(m))} m`;
    },
    clock: (sec) => {
      if (!Number.isFinite(sec)) return '0:00';
      const s = Math.max(0, Math.floor(sec));
      return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
    },
  };

  // ------------------------------------------------------------------ UI basics
  const ui = {
    view: 'home',
    show(name) {
      if (!$(`#view-${name}`)) return;
      this.view = name;
      $$('.view').forEach((v) => v.classList.toggle('is-active', v.id === `view-${name}`));
      $$('.rail-btn[data-view]').forEach((b) => {
        const on = b.dataset.view === name;
        b.classList.toggle('is-active', on);
        if (on) b.setAttribute('aria-current', 'page'); else b.removeAttribute('aria-current');
      });
      bus.emit('view', name);
    },
    toast(msg, ms = 3000) {
      const t = $('#toast');
      t.textContent = msg;
      t.classList.add('is-on');
      clearTimeout(this.toastTimer);
      this.toastTimer = setTimeout(() => t.classList.remove('is-on'), ms);
    },
    applyTheme() {
      let theme = state.settings.theme;
      if (theme === 'auto') {
        const hour = new Date().getHours();
        const isDay = weather.isDay != null ? weather.isDay : hour >= 7 && hour < 19;
        theme = isDay ? 'day' : 'night';
      }
      document.documentElement.dataset.theme = theme;
      const bg = getComputedStyle(document.documentElement).getPropertyValue('--bg').trim();
      const meta = $('meta[name="theme-color"]');
      if (meta && bg) meta.content = bg;
    },
  };

  function openLink(href) {
    const a = document.createElement('a');
    a.href = href;
    a.rel = 'noopener';
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  // ------------------------------------------------------------------ speech output
  const speech = {
    supported: 'speechSynthesis' in window && typeof SpeechSynthesisUtterance !== 'undefined',
    say(text, { interrupt = false } = {}) {
      if (!this.supported || !text) return;
      if (interrupt) speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.rate = Number(state.settings.speechRate) || 1;
      u.lang = navigator.language || 'en-US';
      u.onstart = () => media.duck(true);
      u.onend = u.onerror = () => setTimeout(() => {
        if (!speechSynthesis.speaking && !speechSynthesis.pending) media.duck(false);
      }, 80);
      speechSynthesis.speak(u);
    },
  };

  // ------------------------------------------------------------------ GPS
  const gps = {
    pos: null,
    status: 'idle',
    watchId: null,
    trip: 0,
    start() {
      if (!('geolocation' in navigator)) { this.setStatus('unsupported'); return; }
      if (!window.isSecureContext) ui.toast('Location needs HTTPS. Open DriveDeck over https:// to use GPS.', 6000);
      this.setStatus('searching');
      this.watchId = navigator.geolocation.watchPosition(
        (p) => this.onFix(p),
        (err) => this.onErr(err),
        { enableHighAccuracy: true, maximumAge: 1000, timeout: 20000 },
      );
    },
    onFix(p) {
      if (sim.active) return;
      const c = p.coords;
      this.ingest({
        lat: c.latitude, lon: c.longitude, speed: c.speed, heading: c.heading,
        accuracy: c.accuracy, ts: p.timestamp || Date.now(), simulated: false,
      });
    },
    ingest(fix) {
      const prev = this.pos;
      const bad = (v) => v == null || !Number.isFinite(v);
      if (prev) {
        const d = haversine([prev.lat, prev.lon], [fix.lat, fix.lon]);
        const dt = (fix.ts - prev.ts) / 1000;
        if (bad(fix.speed) && dt > 0 && dt < 10) fix.speed = d / dt;
        if (bad(fix.heading) || (fix.speed ?? 0) < 1) {
          fix.heading = d > 4 && (fix.speed ?? 0) >= 1 ? bearing([prev.lat, prev.lon], [fix.lat, fix.lon]) : prev.heading;
        }
        if (d > 2 && d < 500 && (fix.accuracy ?? 99) < 50 && (fix.speed ?? 0) > 0.8) this.trip += d;
      }
      if (bad(fix.speed)) fix.speed = 0;
      if (bad(fix.heading)) fix.heading = null;
      this.pos = fix;
      this.setStatus(fix.simulated ? 'sim' : 'ok');
      bus.emit('position', fix);
    },
    onErr(err) {
      if (err.code === 1) {
        this.setStatus('denied');
        ui.toast('Location is blocked. Allow it in your browser to see speed and maps.', 5000);
      } else if (this.status !== 'ok') {
        this.setStatus('searching');
      }
    },
    setStatus(s) {
      if (s === this.status) return;
      this.status = s;
      bus.emit('gps-status', s);
    },
  };

  // ------------------------------------------------------------------ simulated drive
  const sim = {
    active: false,
    timer: null,
    along: 0,
    base: null,
    start() {
      if (this.active) return;
      this.active = true;
      this.along = 0;
      this.base = gps.pos ? [gps.pos.lat, gps.pos.lon] : [51.5074, -0.1278];
      gps.setStatus('sim');
      this.timer = setInterval(() => this.tick(), 1000);
      this.tick();
      ui.toast('Simulated drive on. Pick a destination and press Start.');
    },
    stop() {
      if (!this.active) return;
      this.active = false;
      clearInterval(this.timer);
      this.timer = null;
      if (gps.pos) gps.pos = { ...gps.pos, speed: 0, simulated: false };
      gps.setStatus(gps.watchId != null ? 'searching' : 'idle');
      bus.emit('position', gps.pos || { speed: 0 });
    },
    pointAt(route, along) {
      const { coords, cum } = route;
      if (coords.length < 2) return { pt: coords[0], heading: 0 };
      let lo = 0; let hi = cum.length - 1;
      while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (cum[mid] <= along) lo = mid; else hi = mid; }
      const segLen = cum[hi] - cum[lo] || 1;
      const t = clamp((along - cum[lo]) / segLen, 0, 1);
      const a = coords[lo]; const b = coords[hi];
      return { pt: [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t], heading: bearing(a, b) };
    },
    tick() {
      const r = nav.route;
      const now = Date.now();
      if (!r || !nav.active) {
        const p = gps.pos ? [gps.pos.lat, gps.pos.lon] : this.base;
        gps.ingest({ lat: p[0], lon: p[1], speed: 0, heading: gps.pos?.heading ?? 0, accuracy: 5, ts: now, simulated: true });
        return;
      }
      const next = nav.nextStep();
      const toTurn = next ? next.at - this.along : Infinity;
      const speed = toTurn < 70 ? 8 : 15; // ease off for turns, ~54 km/h otherwise
      this.along = Math.min(r.total, this.along + speed);
      const { pt, heading } = this.pointAt(r, this.along);
      const atEnd = this.along >= r.total;
      gps.ingest({ lat: pt[0], lon: pt[1], speed: atEnd ? 0 : speed, heading, accuracy: 5, ts: now, simulated: true });
    },
  };

  // ------------------------------------------------------------------ weather
  const WMO = {
    0: 'Clear', 1: 'Mostly clear', 2: 'Partly cloudy', 3: 'Overcast', 45: 'Fog', 48: 'Freezing fog',
    51: 'Light drizzle', 53: 'Drizzle', 55: 'Heavy drizzle', 56: 'Freezing drizzle', 57: 'Freezing drizzle',
    61: 'Light rain', 63: 'Rain', 65: 'Heavy rain', 66: 'Freezing rain', 67: 'Freezing rain',
    71: 'Light snow', 73: 'Snow', 75: 'Heavy snow', 77: 'Snow grains',
    80: 'Showers', 81: 'Showers', 82: 'Heavy showers', 85: 'Snow showers', 86: 'Snow showers',
    95: 'Thunderstorm', 96: 'Thunderstorm, hail', 99: 'Thunderstorm, hail',
  };
  const weather = {
    data: null,
    isDay: null,
    lastAt: 0,
    lastPos: null,
    loading: false,
    maybeRefresh(p) {
      if (!p || p.lat == null) return;
      const moved = this.lastPos ? haversine(this.lastPos, [p.lat, p.lon]) : Infinity;
      if (this.loading || (Date.now() - this.lastAt < 10 * 60e3 && moved < 10000)) return;
      this.fetch(p.lat, p.lon);
    },
    async fetch(lat, lon) {
      this.loading = true;
      const imp = imperial();
      const url = 'https://api.open-meteo.com/v1/forecast'
        + `?latitude=${lat.toFixed(3)}&longitude=${lon.toFixed(3)}`
        + '&current=temperature_2m,apparent_temperature,weather_code,wind_speed_10m,is_day'
        + '&timezone=auto&forecast_days=1'
        + (imp ? '&temperature_unit=fahrenheit&wind_speed_unit=mph' : '');
      try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const c = (await res.json()).current || {};
        this.data = {
          temp: c.temperature_2m, feels: c.apparent_temperature, code: c.weather_code, wind: c.wind_speed_10m,
          tempUnit: imp ? '°F' : '°C', windUnit: imp ? 'mph' : 'km/h',
        };
        this.isDay = c.is_day == null ? null : c.is_day === 1;
        this.lastAt = Date.now();
        this.lastPos = [lat, lon];
        bus.emit('weather');
        ui.applyTheme();
      } catch (e) {
        this.lastAt = Date.now() - 8 * 60e3; // retry in ~2 minutes
      } finally {
        this.loading = false;
      }
    },
    describe: (code) => WMO[code] || 'Weather',
    sentence() {
      const d = this.data;
      if (!d) return gps.pos ? "I don't have the weather yet." : 'I need your location to check the weather.';
      return `It's ${Math.round(d.temp)} degrees and ${this.describe(d.code).toLowerCase()}, feels like ${Math.round(d.feels)}.`;
    },
  };

  // ------------------------------------------------------------------ turn instructions
  const ORD = ['', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth'];
  function modWord(mod) {
    switch (mod) {
      case 'slight left': return 'slightly left';
      case 'slight right': return 'slightly right';
      case 'sharp left': return 'sharp left';
      case 'sharp right': return 'sharp right';
      case 'uturn': return 'around';
      default: return mod || 'straight';
    }
  }
  function instruction(s) {
    const m = s.maneuver || {};
    const type = m.type;
    const mod = m.modifier;
    const road = s.name || s.ref || '';
    const onto = road ? ` onto ${road}` : '';
    const side = mod && mod.includes('left') ? 'left' : mod && mod.includes('right') ? 'right' : '';
    switch (type) {
      case 'depart': {
        const dir = m.bearing_after != null ? COMPASS_WORDS[compass(m.bearing_after)] : '';
        return `Head ${dir || 'out'}${road ? ` on ${road}` : ''}`;
      }
      case 'arrive':
        return side ? `Arrive at your destination on the ${side}` : 'Arrive at your destination';
      case 'roundabout':
      case 'rotary':
        return m.exit ? `At the roundabout, take the ${ORD[m.exit] || `number ${m.exit}`} exit${onto}` : `Enter the roundabout${onto}`;
      case 'exit roundabout':
      case 'exit rotary':
        return `Exit the roundabout${onto}`;
      case 'merge':
        return `Merge${side ? ` ${side}` : ''}${onto}`;
      case 'on ramp':
        return `Take the ramp${side ? ` on the ${side}` : ''}${onto}`;
      case 'off ramp':
        return `Take the exit${side ? ` on the ${side}` : ''}${onto}`;
      case 'fork':
        return `Keep ${side || 'straight'} at the fork${onto}`;
      case 'end of road':
        return `At the end of the road, turn ${side || modWord(mod)}${onto}`;
      case 'new name':
      case 'notification':
        return `Continue${onto}`;
      case 'continue':
        if (!mod || mod === 'straight') return `Continue straight${onto}`;
        if (mod === 'uturn') return `Make a U-turn${onto}`;
        return `Keep ${modWord(mod)}${onto}`;
      default: // 'turn' and anything new
        if (mod === 'uturn') return `Make a U-turn${onto}`;
        if (!mod || mod === 'straight') return `Continue straight${onto}`;
        if (mod.startsWith('slight')) return `Bear ${side}${onto}`;
        return `Turn ${modWord(mod)}${onto}`;
    }
  }
  const MOD_ANGLE = { straight: 0, 'slight right': 45, right: 90, 'sharp right': 135, uturn: 180, 'sharp left': -135, left: -90, 'slight left': -45 };
  function stepIcon(step) {
    if (!step) return { name: 'arrow', rot: 0 };
    if (step.type === 'arrive') return { name: 'flag', rot: 0 };
    if (step.type === 'roundabout' || step.type === 'rotary') return { name: 'round', rot: 0 };
    if (step.modifier === 'uturn') return { name: 'uturn', rot: 0 };
    return { name: 'arrow', rot: MOD_ANGLE[step.modifier] ?? 0 };
  }

  function buildRoute(r) {
    const coords = r.geometry.coordinates.map(([lon, lat]) => [lat, lon]);
    const cum = [0];
    for (let i = 1; i < coords.length; i++) cum.push(cum[i - 1] + haversine(coords[i - 1], coords[i]));
    const total = cum[cum.length - 1];
    const steps = [];
    let seg = 0;
    for (const leg of r.legs || []) {
      for (const s of leg.steps || []) {
        const loc = [s.maneuver.location[1], s.maneuver.location[0]];
        let best = { d: Infinity, at: cum[seg] || 0, seg };
        for (let i = seg; i < coords.length - 1; i++) {
          const pr = projectOnSegment(loc, coords[i], coords[i + 1]);
          if (pr.d < best.d) best = { d: pr.d, at: cum[i] + pr.t * (cum[i + 1] - cum[i]), seg: i };
          if (pr.d < 0.5 || (best.d < 5 && cum[i] - best.at > 300)) break;
        }
        seg = best.seg;
        steps.push({
          at: best.at, loc, type: s.maneuver.type, modifier: s.maneuver.modifier,
          name: s.name || s.ref || '', text: instruction(s),
        });
      }
    }
    return { coords, cum, total, duration: r.duration, distance: r.distance, steps };
  }

  // ------------------------------------------------------------------ navigation
  const nav = {
    dest: null,
    route: null,
    active: false,
    routing: false,
    progress: 0,
    seg: 0,
    offCount: 0,
    lastRouteAt: 0,
    announced: new Set(),
    lastUpdate: null,

    async search(q) {
      q = (q || '').trim();
      if (!q) return [];
      const ll = q.match(/^\s*(-?\d{1,2}(?:\.\d+)?)\s*,\s*(-?\d{1,3}(?:\.\d+)?)\s*$/);
      if (ll) return [{ name: `${ll[1]}, ${ll[2]}`, detail: 'Coordinates', lat: +ll[1], lon: +ll[2] }];
      const params = new URLSearchParams({ q, format: 'jsonv2', limit: '6', 'accept-language': navigator.language || 'en' });
      if (gps.pos) {
        const { lat, lon } = gps.pos;
        params.set('viewbox', [lon - 0.6, lat + 0.4, lon + 0.6, lat - 0.4].map((n) => n.toFixed(4)).join(','));
      }
      const res = await fetch(`https://nominatim.openstreetmap.org/search?${params}`);
      if (!res.ok) throw new Error(`search failed (${res.status})`);
      const list = await res.json();
      return list.map((x) => {
        const parts = String(x.display_name || '').split(', ');
        const name = x.name || parts[0];
        const rest = parts[0] === name ? parts.slice(1) : parts;
        return { name, detail: rest.slice(0, 3).join(', '), lat: +x.lat, lon: +x.lon };
      });
    },

    async plan(dest, { start = false, reroute = false } = {}) {
      if (!gps.pos || gps.pos.lat == null) {
        ui.toast('Waiting for your location. Allow location access or turn on Simulate a drive.', 4500);
        return false;
      }
      if (this.routing) return false;
      this.routing = true;
      if (!reroute) bus.emit('routing', dest);
      try {
        const from = [gps.pos.lat, gps.pos.lon];
        const url = 'https://router.project-osrm.org/route/v1/driving/'
          + `${from[1].toFixed(6)},${from[0].toFixed(6)};${dest.lon.toFixed(6)},${dest.lat.toFixed(6)}`
          + '?overview=full&geometries=geojson&steps=true';
        const res = await fetch(url);
        const j = await res.json().catch(() => ({}));
        if (!res.ok || j.code !== 'Ok' || !j.routes || !j.routes.length) throw new Error(j.message || 'no route found');
        this.dest = { name: dest.name, detail: dest.detail || '', lat: dest.lat, lon: dest.lon };
        this.route = buildRoute(j.routes[0]);
        this.progress = 0;
        this.seg = 0;
        this.offCount = 0;
        this.lastRouteAt = Date.now();
        this.announced = new Set();
        sim.along = 0;
        if (!reroute) addRecent(this.dest);
        bus.emit('route', this.route);
        if (start || reroute) this.start({ quiet: reroute });
        return true;
      } catch (e) {
        ui.toast(`Couldn't get directions: ${e.message}`, 4500);
        if (!reroute) bus.emit('route', this.route);
        return false;
      } finally {
        this.routing = false;
      }
    },

    start({ quiet = false } = {}) {
      if (!this.route) return;
      this.active = true;
      if (!quiet && state.settings.voiceGuidance) {
        const r = this.route;
        speech.say(`Starting route to ${this.dest.name}. ${fmt.durationSpoken(r.duration)}, ${fmt.dist(r.total, true)}. ${r.steps[0] ? `${r.steps[0].text}.` : ''}`);
      }
      wake.sync();
      bus.emit('nav');
      if (gps.pos) this.onPosition(gps.pos);
    },

    stop({ arrived = false } = {}) {
      const wasActive = this.active;
      this.active = false;
      this.route = null;
      this.dest = null;
      this.lastUpdate = null;
      if (!arrived && wasActive && speech.supported) window.speechSynthesis.cancel();
      bus.emit('route', null);
      bus.emit('nav');
    },

    nextStep() {
      if (!this.route) return null;
      return this.route.steps.find((s) => s.at > this.progress + 8) || null;
    },

    nearest(pt, lo, hi) {
      const { coords, cum } = this.route;
      let best = { d: Infinity, at: 0, seg: lo };
      for (let i = lo; i <= hi && i < coords.length - 1; i++) {
        const pr = projectOnSegment(pt, coords[i], coords[i + 1]);
        if (pr.d < best.d) best = { d: pr.d, at: cum[i] + pr.t * (cum[i + 1] - cum[i]), seg: i };
      }
      return best;
    },

    onPosition(p) {
      const r = this.route;
      if (!r || p.lat == null) return;
      const pt = [p.lat, p.lon];
      const last = r.coords.length - 2;
      let best = this.nearest(pt, Math.max(0, this.seg - 3), Math.min(last, this.seg + 300));
      if (best.d > 40) {
        const g = this.nearest(pt, 0, last);
        if (g.d < best.d - 10) best = g;
      }
      const tol = clamp((p.accuracy || 0) * 1.5, 45, 150);
      if (best.d > tol) {
        this.offCount++;
      } else {
        this.offCount = 0;
        this.seg = best.seg;
        this.progress = best.at;
      }

      if (this.active && this.offCount >= 3 && !this.routing && Date.now() - this.lastRouteAt > 12000) {
        ui.toast('Rerouting…');
        if (state.settings.voiceGuidance) speech.say('Rerouting.', { interrupt: true });
        this.plan(this.dest, { reroute: true });
        return;
      }

      const remaining = Math.max(0, r.total - this.progress);
      const idx = r.steps.findIndex((s) => s.at > this.progress + 8);
      const step = idx >= 0 ? r.steps[idx] : null;
      const dist = step ? step.at - this.progress : remaining;
      const remainingSec = r.total > 0 ? r.duration * (remaining / r.total) : 0;
      this.lastUpdate = {
        step, then: idx >= 0 ? r.steps[idx + 1] || null : null, dist, remaining, remainingSec,
        arrival: new Date(Date.now() + remainingSec * 1000),
      };

      if (this.active && (remaining < 25 || (!step && remaining < 60))) {
        const name = this.dest?.name;
        if (state.settings.voiceGuidance) speech.say(`You have arrived${name ? ` at ${name}` : ''}.`, { interrupt: true });
        ui.toast(`Arrived${name ? ` at ${name}` : ''}`);
        this.stop({ arrived: true });
        return;
      }

      if (this.active && step && state.settings.voiceGuidance) this.prompt(idx, step, dist, p.speed || 0);
      bus.emit('nav-update', this.lastUpdate);
    },

    prompt(idx, step, dist, speed) {
      const v = Math.max(speed, 6);
      const nowD = clamp(v * 6, 30, 150);
      const nearD = clamp(v * 20, 200, 800);
      const farD = clamp(v * 60, 800, 2500);
      let stage = null;
      if (dist <= nowD) stage = 'now';
      else if (dist <= nearD) stage = 'near';
      else if (dist <= farD && dist > nearD + 150) stage = 'far';
      if (!stage) return;
      const key = (s) => `${idx}:${s}`;
      if (this.announced.has(key(stage))) return;
      for (const s of ['far', 'near', 'now']) { this.announced.add(key(s)); if (s === stage) break; }

      let text = stage === 'now' ? step.text : `In ${fmt.dist(dist, true)}, ${lowerFirst(step.text)}`;
      const then = this.route.steps[idx + 1];
      if (stage === 'now' && then && then.at - step.at < 150) text += `, then ${lowerFirst(then.text)}`;
      speech.say(`${text}.`, { interrupt: stage === 'now' });
    },

    etaSentence() {
      const u = this.lastUpdate;
      if (!this.active || !u) return "You're not navigating right now.";
      return `You'll arrive at ${fmt.time(u.arrival)}, in ${fmt.durationSpoken(u.remainingSec)}. ${fmt.dist(u.remaining, true)} to go.`;
    },
  };

  function addRecent(dest) {
    const key = (d) => `${d.lat.toFixed(4)},${d.lon.toFixed(4)}`;
    state.recents = [dest, ...state.recents.filter((d) => key(d) !== key(dest))].slice(0, 6);
    save();
    bus.emit('places');
  }

  // ------------------------------------------------------------------ map
  const mapView = {
    map: null,
    car: null,
    line: null,
    casing: null,
    destMarker: null,
    follow: true,
    init() {
      if (this.map) return true;
      if (!window.L) { $('#mapFallback').hidden = false; return false; }
      const p = gps.pos && gps.pos.lat != null ? gps.pos : null;
      this.map = L.map('map', {
        zoomControl: false,
        center: p ? [p.lat, p.lon] : [39.5, -98.35],
        zoom: p ? 16 : 4,
      });
      L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a>',
      }).addTo(this.map);
      this.map.on('dragstart', () => this.setFollow(false));
      if (p) this.updateCar(p);
      if (nav.route) this.drawRoute(nav.route, nav.dest);
      return true;
    },
    onShow() {
      if (!this.init()) return;
      this.map.invalidateSize();
      if (this.follow && gps.pos && gps.pos.lat != null) this.map.setView([gps.pos.lat, gps.pos.lon], Math.max(this.map.getZoom(), 15), { animate: false });
    },
    setFollow(on) {
      this.follow = on;
      $('#recenterBtn').classList.toggle('is-on', !on);
      if (on && this.map && gps.pos && gps.pos.lat != null) this.map.setView([gps.pos.lat, gps.pos.lon], Math.max(this.map.getZoom(), 16));
    },
    updateCar(p) {
      if (!this.map || p.lat == null) return;
      const ll = [p.lat, p.lon];
      if (!this.car) {
        this.car = L.marker(ll, {
          icon: L.divIcon({
            className: 'car-icon',
            html: '<div class="car"><svg viewBox="0 0 24 24"><path d="M12 2.5l7.5 18.5-7.5-4.2-7.5 4.2z"/></svg></div>',
            iconSize: [44, 44],
            iconAnchor: [22, 22],
          }),
          interactive: false,
          keyboard: false,
          zIndexOffset: 1000,
        }).addTo(this.map);
        if (this.follow && ui.view === 'map') this.map.setView(ll, 16);
      } else {
        this.car.setLatLng(ll);
      }
      const el = this.car.getElement();
      const car = el && el.querySelector('.car');
      if (car) car.style.transform = `rotate(${p.heading || 0}deg)`;
      if (this.follow && ui.view === 'map') this.map.panTo(ll, { animate: true, duration: 0.8 });
    },
    drawRoute(route, dest) {
      if (!this.map) return;
      [this.line, this.casing, this.destMarker].forEach((l) => l && l.remove());
      this.line = this.casing = this.destMarker = null;
      if (!route) return;
      const css = getComputedStyle(document.documentElement);
      this.casing = L.polyline(route.coords, { color: '#0b1220', weight: 12, opacity: 0.55, interactive: false }).addTo(this.map);
      this.line = L.polyline(route.coords, { color: css.getPropertyValue('--accent').trim() || '#ffb020', weight: 7, opacity: 0.95, interactive: false }).addTo(this.map);
      if (dest) {
        this.destMarker = L.circleMarker([dest.lat, dest.lon], {
          radius: 10, color: '#ffffff', weight: 3, fillColor: css.getPropertyValue('--stop').trim() || '#f2555a', fillOpacity: 1,
        }).addTo(this.map);
      }
      if (ui.view === 'map' && !nav.active) {
        this.setFollow(false);
        this.map.fitBounds(this.line.getBounds(), { padding: [60, 60], paddingBottomRight: [60, 220] });
      }
    },
  };

  // ------------------------------------------------------------------ media
  const media = {
    audio: new Audio(),
    library: [],
    list: 'library',
    shownList: 'library',
    index: -1,
    current: null,
    ducked: false,
    init() {
      const a = this.audio;
      a.preload = 'metadata';
      ['play', 'pause', 'loadedmetadata'].forEach((ev) => a.addEventListener(ev, () => bus.emit('media')));
      a.addEventListener('timeupdate', () => bus.emit('media-time'));
      a.addEventListener('ended', () => { if (this.current?.kind === 'file') this.next(); });
      a.addEventListener('error', () => { if (this.current) ui.toast(`Couldn't play “${this.current.title}”.`); });
      if ('mediaSession' in navigator) {
        const set = (action, fn) => { try { navigator.mediaSession.setActionHandler(action, fn); } catch (e) { /* unsupported */ } };
        set('play', () => this.play());
        set('pause', () => this.pause());
        set('stop', () => this.pause());
        set('previoustrack', () => this.prev());
        set('nexttrack', () => this.next());
      }
    },
    items(list = this.list) {
      if (list === 'radio') return state.stations.map((s) => ({ id: s.id, title: s.name, artist: 'Radio', src: s.url, kind: 'radio' }));
      return this.library;
    },
    playItem(list, i) {
      const items = this.items(list);
      const it = items[i];
      if (!it) return;
      this.list = list;
      this.index = i;
      this.current = it;
      this.audio.src = it.src;
      this.audio.play().catch((e) => { if (e.name !== 'AbortError') ui.toast('Tap play to start audio.'); });
      if ('mediaSession' in navigator && 'MediaMetadata' in window) {
        navigator.mediaSession.metadata = new MediaMetadata({ title: it.title, artist: it.artist, album: 'DriveDeck' });
      }
      bus.emit('media');
    },
    play() {
      if (!this.current) {
        for (const list of [this.shownList, this.shownList === 'radio' ? 'library' : 'radio']) {
          if (this.items(list).length) { this.playItem(list, 0); return; }
        }
        ui.toast('Add music files or a radio station first.');
        return;
      }
      this.audio.play().catch(() => ui.toast('Tap play to start audio.'));
    },
    pause() { this.audio.pause(); },
    toggle() { if (this.audio.paused) this.play(); else this.pause(); },
    next() {
      const items = this.items();
      if (!items.length) return this.play();
      this.playItem(this.list, (this.index + 1) % items.length);
    },
    prev() {
      if (this.current?.kind === 'file' && this.audio.currentTime > 4) { this.audio.currentTime = 0; return; }
      const items = this.items();
      if (!items.length) return this.play();
      this.playItem(this.list, (this.index - 1 + items.length) % items.length);
    },
    addFiles(files) {
      let added = 0;
      for (const f of files) {
        if (!(f.type || '').startsWith('audio/') && !/\.(mp3|m4a|aac|wav|ogg|opus|flac)$/i.test(f.name)) continue;
        this.library.push({ id: uid(), title: f.name.replace(/\.[^.]+$/, ''), artist: 'On this device', src: URL.createObjectURL(f), kind: 'file' });
        added++;
      }
      ui.toast(added ? `Added ${added} track${added === 1 ? '' : 's'}.` : 'No audio files found in that selection.');
      bus.emit('media');
    },
    removeItem(list, id) {
      if (list === 'radio') {
        state.stations = state.stations.filter((s) => s.id !== id);
        save();
      } else {
        const it = this.library.find((x) => x.id === id);
        if (it) URL.revokeObjectURL(it.src);
        this.library = this.library.filter((x) => x.id !== id);
      }
      if (this.current && this.current.id === id) { this.audio.pause(); this.audio.removeAttribute('src'); this.current = null; this.index = -1; }
      else if (this.current && this.list === list) this.index = this.items(list).findIndex((x) => x.id === this.current.id);
      bus.emit('media');
    },
    playByVoice(q, reply) {
      const needle = q.toLowerCase().replace(/^(?:the|some|my)\s+/, '').replace(/\s+(?:station|radio)$/, '');
      for (const list of ['radio', 'library']) {
        const i = this.items(list).findIndex((x) => x.title.toLowerCase().includes(needle));
        if (i >= 0) { this.playItem(list, i); return; }
      }
      reply(`I couldn't find ${q} in your music or radio stations.`);
    },
    duck(on) {
      if (on === this.ducked) return;
      this.ducked = on;
      this.audio.volume = on ? 0.25 : 1;
    },
  };

  // ------------------------------------------------------------------ phone
  const phone = {
    dial: '',
    editing: false,
    call(number, name) {
      const n = String(number).replace(/[^\d+*#]/g, '');
      if (!n) return;
      ui.toast(`Calling ${name || n}…`);
      openLink(`tel:${n}`);
    },
    text(number) {
      const n = String(number).replace(/[^\d+]/g, '');
      if (n) openLink(`sms:${n}`);
    },
    find(q) {
      const needle = q.toLowerCase().replace(/^(?:my|the)\s+/, '').trim();
      return state.favorites.find((f) => f.name.toLowerCase() === needle)
        || state.favorites.find((f) => f.name.toLowerCase().includes(needle) || needle.includes(f.name.toLowerCase()));
    },
    callByVoice(q, reply) {
      const digits = q.replace(/[^\d+]/g, '');
      if (digits.length >= 3) { reply(`Calling ${digits.split('').join(' ')}.`); this.call(digits); return; }
      const f = this.find(q);
      if (!f) { reply(`I couldn't find ${q} in your favorites.`); return; }
      reply(`Calling ${f.name}.`);
      setTimeout(() => this.call(f.number, f.name), 900);
    },
    textByVoice(q, reply) {
      const f = this.find(q);
      if (!f) { reply(`I couldn't find ${q} in your favorites.`); return; }
      reply(`Opening a message to ${f.name}.`);
      this.text(f.number);
    },
  };

  // ------------------------------------------------------------------ NatGas signal
  const markets = {
    data: null,
    error: null,
    at: 0,
    loading: false,
    async refresh() {
      if (!CFG.signalUrl || this.loading) return;
      this.loading = true;
      bus.emit('markets');
      try {
        const res = await fetch(CFG.signalUrl, { headers: { Accept: 'application/json' } });
        const j = await res.json().catch(() => ({ ok: false, error: `Server returned ${res.status}` }));
        if (!j.ok) throw new Error(j.error || 'Signal unavailable');
        this.data = j;
        this.error = null;
      } catch (e) {
        this.error = e.message || String(e);
      } finally {
        this.at = Date.now();
        this.loading = false;
        bus.emit('markets');
      }
    },
    async refreshIfStale() {
      if (Date.now() - this.at > 5 * 60e3) await this.refresh();
    },
    shortAction() {
      const a = this.data?.action || '';
      if (a.startsWith('BUY')) return 'BUY';
      if (a.startsWith('SELL')) return 'SELL';
      return a ? 'WAIT' : '';
    },
    summary() {
      if (!this.data) return this.error ? `I couldn't load the NatGas signal: ${this.error}` : 'The NatGas signal is still loading.';
      const d = this.data;
      const act = this.shortAction();
      const conf = d.confidence != null ? `${Math.round(d.confidence * 100)} percent` : 'unknown';
      const price = d.price != null ? `Natural gas is at ${d.price.toFixed(3)}. ` : '';
      const call = act === 'WAIT' ? 'The bot says wait, no trade right now.' : `The bot says ${act.toLowerCase()} with ${conf} confidence.`;
      return `${price}${call} Daily bias is ${String(d.bias || '').toLowerCase()}, and the market is ${String(d.regime || '').toLowerCase().replace('/', 'or')}.`;
    },
  };

  // ------------------------------------------------------------------ driving lock
  const safety = {
    moving: false,
    since: 0,
    update(p) {
      const kmh = (p.speed || 0) * 3.6;
      const now = Date.now();
      if (kmh > 10) {
        this.since = now;
        if (!this.moving) { this.moving = true; this.apply(); }
      } else if (this.moving && kmh < 3 && now - this.since > 5000) {
        this.moving = false;
        this.apply();
      }
    },
    locked() { return this.moving && !state.settings.passenger; },
    apply() {
      const l = this.locked();
      document.body.classList.toggle('is-locked', l);
      $$('[data-lock]').forEach((el) => { el.disabled = l; });
      if (l && document.activeElement && document.activeElement.matches('input, textarea')) document.activeElement.blur();
    },
  };

  // ------------------------------------------------------------------ screen wake lock
  const wake = {
    sentinel: null,
    async sync() {
      if (!('wakeLock' in navigator)) return;
      const want = state.settings.wakeLock && document.visibilityState === 'visible';
      try {
        if (want && !this.sentinel) {
          this.sentinel = await navigator.wakeLock.request('screen');
          this.sentinel.addEventListener('release', () => { this.sentinel = null; });
        } else if (!want && this.sentinel) {
          await this.sentinel.release();
          this.sentinel = null;
        }
      } catch (e) { /* needs a user gesture on some browsers; retried on next tap */ }
    },
  };

  // ------------------------------------------------------------------ voice commands
  const VIEW_ALIASES = { home: 'home', map: 'map', maps: 'map', navigation: 'map', music: 'music', audio: 'music', radio: 'music', phone: 'phone', market: 'markets', markets: 'markets', natgas: 'markets', settings: 'settings', setup: 'settings' };

  const commands = {
    reply(msg) {
      ui.toast(msg, 4500);
      speech.say(msg, { interrupt: true });
    },
    async run(raw) {
      const reply = (m) => this.reply(m);
      const text = raw.toLowerCase().replace(/[.!?,]+$/g, '').replace(/\s+/g, ' ').trim();
      let m;
      if (!text) return;

      if ((m = text.match(/^(?:open|show|go to|switch to)(?: the)? (home|maps?|navigation|music|audio|radio|phone|markets?|natgas|settings|setup)(?: (?:screen|app|page))?$/))) {
        ui.show(VIEW_ALIASES[m[1]]);
        return;
      }
      if (/^(?:stop|cancel|end|exit|quit) (?:the )?(?:navigation|navigating|route|directions|guidance)$/.test(text)) {
        if (nav.route) { nav.stop(); reply('Navigation ended.'); } else reply("You're not navigating right now.");
        return;
      }
      if (/(how far|how long|\beta\b|when will (?:i|we) (?:arrive|get there)|arrival time)/.test(text)) { reply(nav.etaSentence()); return; }
      if ((m = text.match(/^(?:navigate|directions|drive|take me|go|get me|route|head)(?: me)?(?: (?:to|towards))? (.+)$/))) {
        await this.navigateTo(m[1]);
        return;
      }
      if ((m = text.match(/^(?:call|phone|dial|ring) (.+)$/))) { phone.callByVoice(m[1], reply); return; }
      if ((m = text.match(/^(?:text|message|send a (?:message|text) to) (.+)$/))) { phone.textByVoice(m[1], reply); return; }
      if (/^(?:pause|stop|mute)(?: (?:the )?(?:music|audio|radio|song|playing|playback))?$/.test(text)) { media.pause(); return; }
      if (/^(?:next|skip)(?: (?:song|track|station))?$/.test(text)) { media.next(); return; }
      if (/^(?:previous|back|go back)(?: (?:song|track|station))?$/.test(text)) { media.prev(); return; }
      if (/^(?:play|resume|unpause)(?: (?:the )?(?:music|audio|radio|something))?$/.test(text)) { media.play(); return; }
      if ((m = text.match(/^play (.+)$/))) { media.playByVoice(m[1], reply); return; }
      if (/(nat ?gas|natural gas|gas (?:signal|price|market)|the market|signal|should i (?:buy|sell)|trade)/.test(text)) {
        await markets.refreshIfStale();
        reply(markets.summary());
        return;
      }
      if (/(weather|temperature|outside|raining|snowing|cold|hot)/.test(text)) { reply(weather.sentence()); return; }
      if (/(what time|the time|what's the time)/.test(text)) { reply(`It's ${fmt.time(new Date())}.`); return; }
      if (/(how fast|my speed|what speed)/.test(text)) { reply(`You're doing ${fmt.speed(gps.pos?.speed)} ${imperial() ? 'miles per hour' : 'kilometers per hour'}.`); return; }
      if (/(night|dark) (?:mode|theme)/.test(text)) { setSetting('theme', 'night'); reply('Night mode on.'); return; }
      if (/(day|light) (?:mode|theme)/.test(text)) { setSetting('theme', 'day'); reply('Day mode on.'); return; }
      reply(`Sorry, I didn't get “${raw}”. Try “navigate to…”, “play”, “call…”, or “gas signal”.`);
    },
    async navigateTo(target) {
      const t = target.replace(/^(?:the|a)\s+/, '').trim();
      const placeKey = /^(?:home|my house|my home|house)$/.test(t) ? 'home' : /^(?:work|the office|office|my work|my office)$/.test(t) ? 'work' : null;
      if (placeKey) {
        const place = state.places[placeKey];
        if (!place) { this.reply(`Your ${placeKey} address isn't set. Find it on the map and tap Save as ${placeKey === 'home' ? 'Home' : 'Work'}.`); return; }
        await nav.plan(place, { start: true });
        return;
      }
      const recent = state.recents.find((d) => d.name.toLowerCase() === t);
      if (recent) { await nav.plan(recent, { start: true }); return; }
      try {
        const results = await nav.search(t);
        if (!results.length) { this.reply(`I couldn't find ${target}.`); return; }
        await nav.plan(results[0], { start: true });
      } catch (e) {
        this.reply(`Search isn't working right now: ${e.message}.`);
      }
    },
  };

  const voice = {
    SR: window.SpeechRecognition || window.webkitSpeechRecognition,
    rec: null,
    listening: false,
    open() {
      $('#voice').hidden = false;
      $('#voiceInput').value = '';
      if (this.SR) this.listen();
      else this.setText(speech.supported ? 'Voice input isn’t supported here. Type a command below.' : 'Type a command below.');
    },
    close() {
      const rec = this.rec;
      this.rec = null; // detach first so a late onend can't run a half-heard command
      if (rec) { try { rec.abort(); } catch (e) { /* noop */ } }
      this.listening = false;
      this.setListening(false);
      $('#voice').hidden = true;
    },
    setText(t) { $('#voiceText').textContent = t; },
    setListening(on) {
      $('#voiceOrb').classList.toggle('is-listening', on);
      $('#micBtn').classList.toggle('is-listening', on);
    },
    listen() {
      if (speech.supported) speechSynthesis.cancel();
      const rec = new this.SR();
      this.rec = rec;
      rec.lang = navigator.language || 'en-US';
      rec.interimResults = true;
      rec.continuous = false;
      rec.maxAlternatives = 1;
      let finalText = '';
      rec.onresult = (e) => {
        let interim = '';
        for (let i = e.resultIndex; i < e.results.length; i++) {
          const r = e.results[i];
          if (r.isFinal) finalText += r[0].transcript; else interim += r[0].transcript;
        }
        this.setText(finalText || interim || 'Listening…');
      };
      rec.onerror = (e) => {
        if (e.error === 'not-allowed' || e.error === 'service-not-allowed') this.setText('Microphone access is blocked. Type a command instead.');
        else if (e.error === 'no-speech') this.setText("I didn't hear anything. Tap the mic to try again.");
        else if (e.error !== 'aborted') this.setText(`Voice error: ${e.error}`);
      };
      rec.onend = () => {
        if (this.rec !== rec) return;
        this.listening = false;
        this.setListening(false);
        const said = finalText.trim();
        if (said) {
          $('#voice').hidden = true;
          commands.run(said);
        }
      };
      try {
        rec.start();
        this.listening = true;
        this.setListening(true);
        this.setText('Listening…');
      } catch (e) {
        this.setText('Couldn’t start the microphone. Type a command below.');
      }
    },
  };

  // ------------------------------------------------------------------ settings
  function setSetting(key, value) {
    state.settings[key] = value;
    save();
    switch (key) {
      case 'units':
        weather.lastAt = 0;
        if (gps.pos) weather.maybeRefresh(gps.pos);
        break;
      case 'theme':
        ui.applyTheme();
        if (mapView.line && nav.route) mapView.drawRoute(nav.route, nav.dest);
        break;
      case 'wakeLock':
        wake.sync();
        break;
      case 'passenger':
        safety.apply();
        break;
      case 'simulate':
        if (value) sim.start(); else sim.stop();
        break;
      default:
        break;
    }
    bus.emit('settings');
  }

  // ================================================================== RENDERING
  const clockFmt = new Intl.DateTimeFormat([], { hour: 'numeric', minute: '2-digit' });
  function renderClock() {
    const parts = clockFmt.formatToParts(new Date());
    const period = parts.filter((x) => x.type === 'dayPeriod').map((x) => x.value).join('');
    const main = parts.filter((x) => x.type !== 'dayPeriod').map((x) => x.value).join('').trim();
    $('#railClock').innerHTML = `${esc(main)}${period ? `<small>${esc(period)}</small>` : ''}`;
    $('#statusClock').textContent = `${main}${period ? ` ${period}` : ''}`;
  }

  const GPS_LABEL = { idle: 'GPS off', searching: 'Finding GPS', ok: 'GPS', denied: 'GPS blocked', unsupported: 'No GPS', sim: 'Simulated' };
  function renderGpsStatus() {
    const chip = $('#gpsChip');
    chip.dataset.state = gps.status;
    let label = GPS_LABEL[gps.status] || 'GPS';
    if (gps.status === 'ok' && gps.pos?.accuracy != null) label = `GPS ±${fmt.accuracy(gps.pos.accuracy)}`;
    chip.querySelector('span').textContent = label;
  }

  function renderNet() {
    const on = navigator.onLine !== false;
    const chip = $('#netChip');
    chip.dataset.state = on ? 'ok' : 'offline';
    chip.querySelector('span').textContent = on ? 'Online' : 'Offline';
  }

  function renderSpeed() {
    const p = gps.pos;
    $('#speedVal').textContent = fmt.speed(p?.speed);
    $('#speedUnit').textContent = fmt.speedUnit();
    $('#headingVal').textContent = p?.heading != null ? `${compass(p.heading)} ${Math.round(p.heading)}°` : '–';
    let acc = 'Waiting for GPS…';
    if (gps.status === 'denied') acc = 'Location blocked';
    else if (gps.status === 'unsupported') acc = 'No GPS on this device';
    else if (gps.status === 'sim') acc = 'Simulated position';
    else if (p?.accuracy != null) acc = `Accuracy ±${fmt.accuracy(p.accuracy)}`;
    $('#gpsAccuracy').textContent = acc;
    const trip = imperial() ? gps.trip / 1609.344 : gps.trip / 1000;
    $('#tripVal').textContent = `${trip.toFixed(1)} ${imperial() ? 'mi' : 'km'}`;
  }

  function destButton(kind, place, iconName, title) {
    const sub = place ? (place.detail || place.name) : 'Not set, tap to search';
    return `<button class="dest" data-dest="${kind}">${icon(iconName)}<span class="dest-text"><span class="dest-name">${esc(title)}</span><span class="dest-sub">${esc(sub)}</span></span></button>`;
  }

  function renderHomeNav() {
    const box = $('#homeNav');
    if (nav.active && nav.route) {
      const u = nav.lastUpdate;
      const ic = stepIcon(u?.step);
      box.innerHTML = `
        <div class="label">Navigating to ${esc(nav.dest?.name || '')}</div>
        <div class="nav-live">
          <div class="nav-live-top">
            <svg class="ic" style="transform:rotate(${ic.rot}deg)"><use href="#i-${ic.name}"/></svg>
            <div>
              <div class="nav-live-dist">${u ? fmt.dist(u.dist) : '–'}</div>
              <div class="nav-live-text">${esc(u?.step?.text || 'Starting…')}</div>
            </div>
          </div>
          <div class="nav-live-meta">${u ? `Arrive ${fmt.time(u.arrival)} · ${fmt.dist(u.remaining)} · ${fmt.duration(u.remainingSec)}` : ''}</div>
        </div>`;
      return;
    }
    const parts = [
      destButton('home', state.places.home, 'home', 'Home'),
      destButton('work', state.places.work, 'work', 'Work'),
    ];
    state.recents.slice(0, 3).forEach((d, i) => {
      parts.push(`<button class="dest" data-dest="recent:${i}">${icon('clock')}<span class="dest-text"><span class="dest-name">${esc(d.name)}</span><span class="dest-sub">${esc(d.detail || '')}</span></span></button>`);
    });
    parts.push(`<button class="dest dest-search" data-dest="search">${icon('search')}<span class="dest-text"><span class="dest-name">Search</span><span class="dest-sub">Find a place</span></span></button>`);
    box.innerHTML = `<div class="label">Where to?</div><div class="dest-grid" id="destGrid">${parts.join('')}</div>`;
  }

  function renderGuide() {
    const g = $('#guide');
    const u = nav.lastUpdate;
    if (!nav.active || !nav.route) { g.hidden = true; return; }
    g.hidden = false;
    const ic = stepIcon(u?.step);
    const use = $('#guideIcon use');
    use.setAttribute('href', `#i-${ic.name}`);
    $('#guideIcon').style.transform = `rotate(${ic.rot}deg)`;
    $('#guideDist').textContent = u ? fmt.dist(u.dist) : '–';
    $('#guideText').textContent = u?.step?.text || 'Starting route…';
    const then = $('#guideThen');
    if (u?.then && u.step && u.then.at - u.step.at < 400) {
      then.hidden = false;
      then.textContent = `Then ${lowerFirst(u.then.text)}`;
    } else {
      then.hidden = true;
    }
    $('#guideArrive').textContent = u ? fmt.time(u.arrival) : '--:--';
    $('#guideLeft').textContent = u ? `${fmt.dist(u.remaining)} · ${fmt.duration(u.remainingSec)}` : '';
  }

  function renderRouteCard() {
    const card = $('#routeCard');
    const r = nav.route;
    // While guiding, the banner carries ETA + End, so keep the map clear.
    if (!r || !nav.dest || nav.active) { card.hidden = true; return; }
    card.hidden = false;
    $('#routeName').textContent = nav.dest.name;
    const arrive = new Date(Date.now() + (nav.lastUpdate?.remainingSec ?? r.duration) * 1000);
    $('#routeMeta').textContent = `${fmt.duration(nav.lastUpdate?.remainingSec ?? r.duration)} · ${fmt.dist(nav.lastUpdate?.remaining ?? r.total)} · arrive ${fmt.time(arrive)}`;
    const start = $('#routeStart');
    start.textContent = nav.active ? 'End route' : 'Start';
    start.classList.toggle('btn-go', !nav.active);
    start.classList.toggle('btn-danger', nav.active);
    const { lat, lon } = nav.dest;
    $('#handoffGoogle').href = `https://www.google.com/maps/dir/?api=1&destination=${lat},${lon}&travelmode=driving`;
    $('#handoffWaze').href = `https://waze.com/ul?ll=${lat},${lon}&navigate=yes`;
    $('#handoffApple').href = `https://maps.apple.com/?daddr=${lat},${lon}&dirflg=d`;
  }

  function renderResults(list, message) {
    const ul = $('#results');
    if (!list && !message) { ul.hidden = true; ul.innerHTML = ''; return; }
    ul.hidden = false;
    if (message) { ul.innerHTML = `<li class="r-empty">${esc(message)}</li>`; return; }
    if (!list.length) { ul.innerHTML = '<li class="r-empty">No places found. Try adding a town or postcode.</li>'; return; }
    ul.innerHTML = list.map((r, i) => {
      const d = gps.pos?.lat != null ? fmt.dist(haversine([gps.pos.lat, gps.pos.lon], [r.lat, r.lon])) : '';
      return `<li><button data-result="${i}">${icon('pin')}<span class="r-text"><span class="r-name">${esc(r.name)}</span><span class="r-sub">${esc(r.detail)}</span></span><span class="r-dist">${d}</span></button></li>`;
    }).join('');
  }

  function artFor(item) {
    if (!item) return icon('music');
    return esc((item.title.trim()[0] || '♪').toUpperCase());
  }
  function artColor(item) {
    if (!item) return '';
    let h = 0;
    for (const c of item.title) h = (h * 31 + c.charCodeAt(0)) % 360;
    return `linear-gradient(135deg, hsl(${h} 65% 45%), hsl(${(h + 50) % 360} 70% 30%))`;
  }

  function renderMedia() {
    const it = media.current;
    const playing = !media.audio.paused && !!it;
    $$('[data-media="toggle"] use').forEach((u) => u.setAttribute('href', playing ? '#i-pause' : '#i-play'));
    for (const [art, title, sub] of [['#homeArt', '#homeTitle', '#homeSub'], ['#npArt', '#npTitle', '#npSub']]) {
      $(art).innerHTML = artFor(it);
      $(art).style.background = artColor(it);
      $(title).textContent = it ? it.title : 'Nothing playing';
      $(sub).textContent = it ? (it.kind === 'radio' ? 'Live radio' : it.artist) : (art === '#homeArt' ? 'Add music or a station in Audio' : 'Pick a track or station');
    }
    $('#homeMediaSource').textContent = it ? (it.kind === 'radio' ? 'Radio' : 'On this device') : 'Audio';
    const chip = $('#nowPlayingChip');
    chip.hidden = !playing;
    chip.textContent = playing ? `♪ ${it.title}` : '';
    $('#seekWrap').hidden = !(it && it.kind === 'file' && Number.isFinite(media.audio.duration));

    // list
    const list = media.shownList;
    $$('.tab').forEach((t) => {
      const on = t.dataset.list === list;
      t.classList.toggle('is-active', on);
      t.setAttribute('aria-selected', String(on));
    });
    $('#libraryFoot').hidden = list !== 'library';
    $('#stationForm').hidden = list !== 'radio';
    const items = media.items(list);
    const ul = $('#mediaList');
    if (!items.length) {
      ul.innerHTML = `<li class="empty">${list === 'radio'
        ? 'No stations yet. Add a stream URL from your favorite station’s “listen live” page.'
        : 'No music yet. Add audio files from this device.'}</li>`;
    } else {
      ul.innerHTML = items.map((x, i) => `
        <li>
          <button class="row-main${it && it.id === x.id ? ' is-current' : ''}" data-play="${i}">
            ${icon(list === 'radio' ? 'radio' : 'music')}
            <span class="row-text"><span class="row-name">${esc(x.title)}</span><span class="row-sub">${esc(list === 'radio' ? x.src : x.artist)}</span></span>
          </button>
          <button class="icon-btn row-del" style="display:grid" data-remove="${esc(x.id)}" aria-label="Remove ${esc(x.title)}">${icon('x')}</button>
        </li>`).join('');
    }
    renderMediaTime();
  }

  function renderMediaTime() {
    const a = media.audio;
    if ($('#seekWrap').hidden) return;
    const seek = $('#seek');
    if (!seek.matches(':active')) {
      seek.max = String(Math.max(1, Math.floor(a.duration || 1)));
      seek.value = String(Math.floor(a.currentTime || 0));
    }
    $('#seekCur').textContent = fmt.clock(a.currentTime);
    $('#seekDur').textContent = fmt.clock(a.duration);
  }

  function renderPhone() {
    const ul = $('#favList');
    ul.classList.toggle('is-editing', phone.editing);
    $('#favEdit').textContent = phone.editing ? 'Done' : 'Edit';
    if (!state.favorites.length) {
      ul.innerHTML = '<li class="empty">No favorites yet. Add the people you call most, then say “call …” while driving.</li>';
    } else {
      ul.innerHTML = state.favorites.map((f) => `
        <li>
          <button class="row-main" data-call="${esc(f.id)}">${icon('phone')}<span class="row-text"><span class="row-name">${esc(f.name)}</span><span class="row-sub">${esc(f.number)}</span></span></button>
          <button class="icon-btn" data-sms="${esc(f.id)}" aria-label="Message ${esc(f.name)}">${icon('msg')}</button>
          <button class="icon-btn row-del" data-unfav="${esc(f.id)}" aria-label="Remove ${esc(f.name)}">${icon('x')}</button>
        </li>`).join('');
    }
    $('#dialOut').textContent = phone.dial;
  }

  function renderMarkets() {
    const d = markets.data;
    const act = markets.shortAction();
    const ga = $('#gasAction');
    ga.dataset.action = act || 'none';
    ga.textContent = act || '–';
    $('#gasSub').textContent = d
      ? `${d.price != null ? d.price.toFixed(3) : '–'} · ${d.confidence != null ? Math.round(d.confidence * 100) : '–'}% conf.`
      : (markets.loading ? 'Loading signal…' : (markets.error ? 'Signal unavailable' : '–'));

    $('#gasHero').dataset.action = act || 'none';
    $('#gasHero').textContent = act || (markets.loading ? '…' : '–');
    $('#gasPrice').textContent = d?.price != null ? d.price.toFixed(3) : '–';
    const conf = d?.confidence != null ? Math.round(d.confidence * 100) : null;
    $('#gasConfBar').style.width = `${conf ?? 0}%`;
    $('#gasConf').textContent = conf != null ? `Confidence ${conf}%${act === 'WAIT' ? ' · no trade' : ''}` : 'Confidence –';
    $('#gasBias').textContent = d?.bias || '–';
    $('#gasRegime').textContent = d?.regime || '–';
    $('#gasQuality').textContent = d?.move_quality || '–';
    $('#gasRsi').textContent = d?.rsi != null ? d.rsi.toFixed(0) : '–';
    $('#gasStop').textContent = d?.stop_loss != null && act !== 'WAIT' ? d.stop_loss.toFixed(3) : '–';
    $('#gasTp').textContent = d?.take_profit != null && act !== 'WAIT' ? d.take_profit.toFixed(3) : '–';
    $('#gasUpdated').textContent = d?.timestamp ? `Candle ${d.timestamp} UTC` : '';
    const err = $('#gasError');
    err.hidden = !markets.error;
    err.textContent = markets.error ? `Couldn't refresh: ${markets.error}` : '';
    $('#gasRefresh').textContent = markets.loading ? 'Refreshing…' : 'Refresh';
  }

  function renderWeather() {
    const w = weather.data;
    const chip = $('#tempChip');
    if (!w) {
      $('#wxTemp').textContent = '–';
      $('#wxDesc').textContent = gps.pos ? 'Loading weather…' : 'Needs your location';
      chip.hidden = true;
      return;
    }
    const t = `${Math.round(w.temp)}${w.tempUnit}`;
    $('#wxTemp').textContent = t;
    $('#wxDesc').textContent = `${weather.describe(w.code)} · wind ${Math.round(w.wind)} ${w.windUnit}`;
    chip.hidden = false;
    chip.textContent = t;
  }

  function renderSettings() {
    const s = state.settings;
    $$('#view-settings input[type="radio"]').forEach((r) => { r.checked = String(s[r.name]) === r.value; });
    $$('#view-settings input[type="checkbox"]').forEach((c) => { c.checked = !!s[c.name]; });
    $('#homePlace').textContent = state.places.home ? state.places.home.name : 'Not set. Search a place and tap Save as Home.';
    $('#workPlace').textContent = state.places.work ? state.places.work.name : 'Not set. Search a place and tap Save as Work.';
    $$('[data-clear-place]').forEach((b) => { b.disabled = !state.places[b.dataset.clearPlace]; });
  }

  // ================================================================== WIRING
  function wire() {
    // App rail
    $$('.rail-btn[data-view]').forEach((b) => b.addEventListener('click', () => ui.show(b.dataset.view)));
    $$('[data-goto]').forEach((b) => b.addEventListener('click', () => ui.show(b.dataset.goto)));
    $('#weatherGlance').addEventListener('click', () => commands.reply(weather.sentence()));
    $('#micBtn').addEventListener('click', () => (voice.listening ? voice.close() : voice.open()));

    // Media transport (home + audio screen)
    document.addEventListener('click', (e) => {
      const b = e.target.closest('[data-media]');
      if (b) media[b.dataset.media]();
    });

    // Home: destinations
    $('#homeNav').addEventListener('click', async (e) => {
      const b = e.target.closest('[data-dest]');
      if (!b) return;
      const key = b.dataset.dest;
      if (key === 'search') { ui.show('map'); if (!safety.locked()) setTimeout(() => $('#searchInput').focus(), 50); return; }
      let place = null;
      if (key === 'home' || key === 'work') {
        place = state.places[key];
        if (!place) {
          ui.show('map');
          ui.toast(`Search for your ${key} address, then tap Save as ${key === 'home' ? 'Home' : 'Work'}.`, 4500);
          return;
        }
      } else if (key.startsWith('recent:')) {
        place = state.recents[Number(key.split(':')[1])];
      }
      if (place) {
        ui.show('map');
        await nav.plan(place, { start: true });
      }
    });

    $('#tripBtn').addEventListener('click', () => { gps.trip = 0; renderSpeed(); ui.toast('Trip reset.'); });

    // Map search
    let lastResults = [];
    $('#searchForm').addEventListener('submit', async (e) => {
      e.preventDefault();
      const q = $('#searchInput').value;
      if (!q.trim()) return;
      $('#searchInput').blur();
      renderResults(null, 'Searching…');
      try {
        lastResults = await nav.search(q);
        renderResults(lastResults);
      } catch (err) {
        renderResults(null, `Search failed: ${err.message}`);
      }
    });
    $('#searchInput').addEventListener('input', () => {
      $('#searchClear').hidden = !$('#searchInput').value;
      if (!$('#searchInput').value) renderResults(null);
    });
    $('#searchClear').addEventListener('click', () => {
      $('#searchInput').value = '';
      $('#searchClear').hidden = true;
      renderResults(null);
    });
    $('#results').addEventListener('click', async (e) => {
      const b = e.target.closest('[data-result]');
      if (!b) return;
      const r = lastResults[Number(b.dataset.result)];
      renderResults(null);
      if (r) await nav.plan(r);
    });

    $('#routeStart').addEventListener('click', () => {
      if (nav.active) nav.stop();
      else { nav.start(); mapView.setFollow(true); }
    });
    $('#routeClose').addEventListener('click', () => nav.stop());
    $('#guideEnd').addEventListener('click', () => nav.stop());
    const savePlace = (kind) => {
      if (!nav.dest) return;
      state.places[kind] = { ...nav.dest };
      save();
      bus.emit('places');
      ui.toast(`Saved ${nav.dest.name} as ${kind === 'home' ? 'Home' : 'Work'}.`);
    };
    $('#routeSaveHome').addEventListener('click', () => savePlace('home'));
    $('#routeSaveWork').addEventListener('click', () => savePlace('work'));

    $('#zoomIn').addEventListener('click', () => mapView.map && mapView.map.zoomIn());
    $('#zoomOut').addEventListener('click', () => mapView.map && mapView.map.zoomOut());
    $('#recenterBtn').addEventListener('click', () => mapView.setFollow(true));

    // Audio
    $$('.tab').forEach((t) => t.addEventListener('click', () => { media.shownList = t.dataset.list; renderMedia(); }));
    $('#mediaList').addEventListener('click', (e) => {
      const play = e.target.closest('[data-play]');
      const rm = e.target.closest('[data-remove]');
      if (play) media.playItem(media.shownList, Number(play.dataset.play));
      else if (rm) media.removeItem(media.shownList, rm.dataset.remove);
    });
    $('#fileInput').addEventListener('change', (e) => { media.addFiles(Array.from(e.target.files || [])); e.target.value = ''; });
    $('#stationForm').addEventListener('submit', (e) => {
      e.preventDefault();
      const name = $('#stationName').value.trim();
      const url = $('#stationUrl').value.trim();
      if (!name || !/^https?:\/\//i.test(url)) { ui.toast('Enter a name and a stream URL starting with https://'); return; }
      if (location.protocol === 'https:' && url.startsWith('http:')) ui.toast('Heads up: browsers block http:// streams on https pages. Use an https:// stream if it won’t play.', 6000);
      state.stations.push({ id: uid(), name, url });
      save();
      $('#stationName').value = '';
      $('#stationUrl').value = '';
      renderMedia();
    });
    $('#seek').addEventListener('input', (e) => { media.audio.currentTime = Number(e.target.value); });

    // Phone
    $('#favForm').addEventListener('submit', (e) => {
      e.preventDefault();
      const name = $('#favName').value.trim();
      const number = $('#favNumber').value.trim();
      if (!name || !/[\d]{3,}/.test(number.replace(/\D/g, ''))) { ui.toast('Enter a name and a phone number.'); return; }
      state.favorites.push({ id: uid(), name, number });
      save();
      $('#favName').value = '';
      $('#favNumber').value = '';
      renderPhone();
    });
    $('#favEdit').addEventListener('click', () => { phone.editing = !phone.editing; renderPhone(); });
    $('#favList').addEventListener('click', (e) => {
      const byId = (id) => state.favorites.find((f) => f.id === id);
      const call = e.target.closest('[data-call]');
      const sms = e.target.closest('[data-sms]');
      const del = e.target.closest('[data-unfav]');
      if (call) { const f = byId(call.dataset.call); if (f) phone.call(f.number, f.name); }
      if (sms) { const f = byId(sms.dataset.sms); if (f) phone.text(f.number); }
      if (del) { state.favorites = state.favorites.filter((f) => f.id !== del.dataset.unfav); save(); renderPhone(); }
    });
    $('#pad').addEventListener('click', (e) => {
      const k = e.target.closest('[data-key]');
      if (!k || phone.dial.length >= 24) return;
      phone.dial += k.dataset.key;
      renderPhone();
    });
    $('#dialBack').addEventListener('click', () => { phone.dial = phone.dial.slice(0, -1); renderPhone(); });
    $('#dialCall').addEventListener('click', () => {
      if (!phone.dial) { ui.toast('Enter a number first.'); return; }
      phone.call(phone.dial);
    });

    // NatGas
    $('#gasSpeak').addEventListener('click', () => commands.reply(markets.summary()));
    $('#gasRefresh').addEventListener('click', () => markets.refresh());

    // Settings
    $('#view-settings').addEventListener('change', (e) => {
      const el = e.target;
      if (!el.name) return;
      if (el.type === 'checkbox') setSetting(el.name, el.checked);
      else if (el.type === 'radio') setSetting(el.name, el.name === 'speechRate' ? Number(el.value) : el.value);
    });
    $$('[data-clear-place]').forEach((b) => b.addEventListener('click', () => {
      state.places[b.dataset.clearPlace] = null;
      save();
      bus.emit('places');
    }));
    $('#clearRecents').addEventListener('click', () => { state.recents = []; save(); bus.emit('places'); ui.toast('Recent places cleared.'); });
    $('#resetAll').addEventListener('click', () => {
      if (!window.confirm('Reset DriveDeck? This removes your favorites, stations, saved places and settings.')) return;
      try { localStorage.removeItem(STORE_KEY); } catch (e) { /* noop */ }
      location.reload();
    });
    $('#fullscreenBtn').addEventListener('click', () => {
      const el = document.documentElement;
      if (document.fullscreenElement) document.exitFullscreen?.();
      else if (el.requestFullscreen) el.requestFullscreen().catch(() => ui.toast('Full screen isn’t available here.'));
      else ui.toast('Full screen isn’t available here. Use “Add to Home Screen” instead.');
    });

    // Voice overlay
    $('#voiceCancel').addEventListener('click', () => voice.close());
    $('#voice').addEventListener('click', (e) => { if (e.target.id === 'voice') voice.close(); });
    $('#voiceForm').addEventListener('submit', (e) => {
      e.preventDefault();
      const v = $('#voiceInput').value.trim();
      if (!v) return;
      voice.close();
      commands.run(v);
    });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && !$('#voice').hidden) voice.close(); });

    // System
    window.addEventListener('online', renderNet);
    window.addEventListener('offline', renderNet);
    document.addEventListener('visibilitychange', () => {
      wake.sync();
      if (document.visibilityState === 'visible') markets.refreshIfStale();
    });
    document.addEventListener('pointerdown', () => wake.sync(), { once: true });
  }

  function subscribe() {
    bus.on('position', (p) => {
      renderSpeed();
      renderGpsStatus();
      safety.update(p);
      if (p.lat != null) {
        mapView.updateCar(p);
        nav.onPosition(p);
        weather.maybeRefresh(p);
      }
    });
    bus.on('gps-status', () => { renderGpsStatus(); renderSpeed(); });
    bus.on('view', (v) => { if (v === 'map') mapView.onShow(); if (v === 'markets') markets.refreshIfStale(); });
    bus.on('routing', (dest) => {
      $('#routeCard').hidden = false;
      $('#routeName').textContent = dest.name;
      $('#routeMeta').textContent = 'Finding the best route…';
    });
    bus.on('route', (r) => {
      mapView.drawRoute(r, nav.dest);
      renderRouteCard();
      renderHomeNav();
      renderGuide();
    });
    bus.on('nav', () => {
      if (nav.active) mapView.setFollow(true);
      renderRouteCard();
      renderHomeNav();
      renderGuide();
    });
    bus.on('nav-update', () => { renderGuide(); renderHomeNav(); renderRouteCard(); });
    bus.on('places', () => { renderHomeNav(); renderSettings(); });
    bus.on('media', renderMedia);
    bus.on('media-time', renderMediaTime);
    bus.on('markets', renderMarkets);
    bus.on('weather', renderWeather);
    bus.on('settings', () => {
      renderSettings();
      renderSpeed();
      renderGpsStatus();
      renderHomeNav();
      renderGuide();
      renderRouteCard();
    });
  }

  function boot() {
    ui.applyTheme();
    subscribe();
    wire();
    media.init();
    renderClock();
    renderNet();
    renderGpsStatus();
    renderSpeed();
    renderHomeNav();
    renderMedia();
    renderPhone();
    renderMarkets();
    renderWeather();
    renderSettings();
    safety.apply();

    setInterval(renderClock, 1000);
    setInterval(() => { if (state.settings.theme === 'auto') ui.applyTheme(); }, 60e3);
    setInterval(() => { if (document.visibilityState === 'visible') markets.refresh(); }, 5 * 60e3);

    gps.start();
    markets.refresh();
    wake.sync();

    // Debug/test hook (harmless in production).
    window.DriveDeck = { state, gps, sim, nav, media, markets, commands, ui, weather, setSetting };
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
