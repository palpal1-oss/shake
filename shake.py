
# author's Zenodo  https://zenodo.org/records/17634111

# follow the link for the full article or you can read a short summary in PDF next to it


#!/usr/bin/env python3

import os
import sys
import json
import time
import math
from io import StringIO
from datetime import datetime, timedelta
from urllib.parse import urlencode

import requests
import numpy as np
import pandas as pd
from scipy import linalg
import plotly.graph_objects as go
import reverse_geocoder as rg


TIME_MAG_THRESHOLD = 5.5
MICRO_RADIUS_KM = 120
MICRO_MAX_AGE_HOURS = 48
MICRO_MIN_MAG = 0.5
MICRO_MAX_MAG = 2.5
FETCH_DAYS = 60          # how many days back to fetch events (default 7)
MIN_MAG_FETCH = 1.0      # min magnitude to download
L_KM = 100.0             # horizontal kernel lengthscale (km)
L_DEPTH = 80.0           # depth attenuation lengthscale (km) for event energy
NX = 80                  # grid resolution lon (cells)
NY = 80                  # grid resolution lat (cells)
LAM = 1e10               # Tikhonov regularization strength
M_MIN_SHOW = 4.0         # show predicted cells with M_pred >= this
CACHE_FILE = "usgs_cache_recent.json"
OUTPUT_HTML = "quake_predictor_globe_depth_plasticity.html"
GRID_MARGIN_DEG = 1.0    # degrees margin around events bounding box
MAX_DAYS_AHEAD = 30      # slider range 0..30
TAU_DAYS = 5.0           # temporal decay timescale in days
E_MIN_MAG = 2.0          # baseline mag for halo radius calc
MAX_HALO_RADIUS_KM = 1000.0  # cap halo radius to avoid massive circles
BVALUE_RADIUS_KM = 150   # radius around point to compute local b-value
MIN_EVENTS_FOR_B = 8     # minimum events to compute b-value reliably
VERBOSE = True


ICON_SCALE_DIRECT = 1.0    # multiplier for direct candidate marker sizes
ICON_SCALE_INV = 1.0       # multiplier for inversion candidate marker sizes
ICON_SCALE_USGS = 2.8      # base multiplier for USGS event marker sizes
ICON_SCALE_NEW = 1.2       # multiplier for "new predicted" symbols


def log(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

def utcnow_str():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

# Haversine distance
def haversine_km_vec(lat1, lon1, lat_arr, lon_arr, R=6371.0):
    lat1r = math.radians(lat1); lon1r = math.radians(lon1)
    lat2r = np.radians(lat_arr); lon2r = np.radians(lon_arr)
    dlat = lat2r - lat1r; dlon = lon2r - lon1r
    a = np.sin(dlat/2.0)**2 + np.cos(lat1r)*np.cos(lat2r)*np.sin(dlon/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

# single-pair haversine (km)
def haversine_km(lat1, lon1, lat2, lon2, R=6371.0):
    lat1r = math.radians(lat1); lon1r = math.radians(lon1)
    lat2r = math.radians(lat2); lon2r = math.radians(lon2)
    dlat = lat2r - lat1r; dlon = lon2r - lon1r
    a = math.sin(dlat/2.0)**2 + math.cos(lat1r)*math.cos(lat2r)*math.sin(dlon/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def energy_from_mag(M):
    return 10.0**(1.5 * M + 4.8)

def mag_from_energy(E):
    Epos = np.maximum(E, 1e-30)
    return (np.log10(Epos) - 4.8) / 1.5

def depth_attenuation_factor(depth_km, Ldepth=L_DEPTH):
    return np.exp(-depth_km / Ldepth)

def time_decay(E, days, tau=TAU_DAYS):
    return E * np.exp(-days / tau)

def decay_radius_km_from_energy(E, E_min=None, L=L_KM):
    if E_min is None:
        E_min = energy_from_mag(E_MIN_MAG)
    Epos = float(max(E, float(E_min) * 1.00001))
    R = L * math.log(Epos / E_min)
    if R < 0: return 0.0
    if R > MAX_HALO_RADIUS_KM: return MAX_HALO_RADIUS_KM
    return R

def circle_polygon(lat0, lon0, R_km, npoints=72):
    if R_km <= 0: return [lat0], [lon0]
    angles = np.linspace(0, 2*math.pi, npoints)
    lat_circle = lat0 + (R_km / 111.0) * np.cos(angles)
    coslat = math.cos(math.radians(lat0))
    if abs(coslat) < 1e-6: coslat = 1e-6
    lon_circle = lon0 + (R_km / (111.0 * coslat)) * np.sin(angles)
    return lat_circle.tolist(), lon_circle.tolist()


def fetch_usgs_day_csv(start, end, minmagnitude=1.0, timeout=90, attempt_pause=0.2):
    params = {
        "format": "csv",
        "starttime": start,
        "endtime": end,
        "minmagnitude": minmagnitude,
        "orderby": "time",
        "limit": 20000
    }
    base = "https://earthquake.usgs.gov/fdsnws/event/1/query?"
    url = base + urlencode(params)
    for attempt in range(1, 6):
        try:
            time.sleep(attempt_pause)
            r = requests.get(url, timeout=timeout)
            if r.status_code != 200:
                raise ValueError(f"HTTP {r.status_code}")
            text = r.text
            if not text or len(text) < 50:
                raise ValueError("Empty or short CSV response")
            df = pd.read_csv(StringIO(text))
            col_map = {}
            for c in df.columns:
                lc = c.lower()
                if lc == "latitude": col_map[c] = "lat"
                if lc == "longitude": col_map[c] = "lon"
                if lc == "depth": col_map[c] = "depth_km"
                if lc == "mag": col_map[c] = "mag"
                if lc == "time": col_map[c] = "time"
            df = df.rename(columns=col_map)
            if "lat" not in df.columns or "lon" not in df.columns or "mag" not in df.columns:
                raise ValueError("Required columns missing in USGS CSV")
            if "depth_km" not in df.columns:
                df["depth_km"] = 0.0
            df["depth_km"] = df["depth_km"].fillna(0.0).astype(float)
            if "time" in df.columns:
                try:
                    df["time"] = pd.to_datetime(df["time"], errors="coerce")
                except Exception:
                    pass
            else:
                df["time"] = pd.NaT
            df = df[["lat", "lon", "depth_km", "mag", "time"]]
            return df
        except Exception as e:
            log(f"[CSV-fetch] {start}->{end} attempt {attempt} failed: {e}")
            time.sleep(0.8 + attempt * 0.2)
    log(f"[CSV-fetch] FAILED for {start}->{end}")
    return pd.DataFrame(columns=["lat", "lon", "depth_km", "mag", "time"])

def fetch_usgs_multi(days_back, minmagnitude=1.0, start_date=None):
    if start_date is None:
        now = datetime.utcnow()
    else:
        # accept YYY-M-DD or YY-MM-DDTHH:M
        try:
            now = datetime.strptime(start_date, "%Y-%m-%d")
        except Exception:
            try:
                now = datetime.strptime(start_date, "%Y-%m-%dT%H:%M")
            except Exception:
                now = datetime.utcnow()
    dfs = []
    for k in range(days_back):
        start = (now - timedelta(days=k+1)).strftime("%Y-%m-%d")
        end = (now - timedelta(days=k)).strftime("%Y-%m-%d")
        log(f"Fetching {start} -> {end}")
        df_day = fetch_usgs_day_csv(start, end, minmagnitude=minmagnitude)
        if not df_day.empty:
            dfs.append(df_day)
    if not dfs:
        if os.path.exists(CACHE_FILE):
            try:
                df_cache = pd.read_json(CACHE_FILE, orient="records")
                if not df_cache.empty:
                    log("Using cached USGS file.")
                    return df_cache
            except Exception:
                pass
        return pd.DataFrame(columns=["lat", "lon", "depth_km", "mag", "time"])
    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["lat","lon","mag","time"])
    df_all = df_all.sort_values("mag", ascending=False).reset_index(drop=True)
    try:
        df_all.to_json(CACHE_FILE, orient="records", date_format="iso")
    except Exception:
        pass
    return df_all


def compute_b_value(magnitudes, bin_correction=0.2):
    mags = np.array(magnitudes, dtype=float)
    if mags.size < MIN_EVENTS_FOR_B:
        return None
    Mmin = mags.min()
    Mmean = mags.mean()
    denom = (Mmean - (Mmin + bin_correction))
    if denom <= 0:
        return None
    b = (math.log10(math.e)) / denom
    if b <= 0: b = 0.01
    if b > 5: b = 5.0
    return float(b)

def b_to_plasticity(b):
    if b is None:
        return 0.5
    plast = (b - 0.8) / 0.7
    return float(max(0.0, min(1.0, plast)))

def region_b_value(df_events, lat, lon, radius_km=BVALUE_RADIUS_KM):
    if df_events is None or df_events.empty:
        return None, 0.5
    dists = haversine_km_vec(lat, lon, df_events["lat"].values, df_events["lon"].values)
    mask = dists <= radius_km
    local_mags = df_events.loc[mask, "mag"].values
    if local_mags.size < MIN_EVENTS_FOR_B:
        for r in [radius_km*2, radius_km*4, 500]:
            mask = dists <= r
            local_mags = df_events.loc[mask, "mag"].values
            if local_mags.size >= MIN_EVENTS_FOR_B:
                break
    if local_mags.size < 5:
        return None, 0.5
    b = compute_b_value(local_mags)
    plast = b_to_plasticity(b)
    return b, plast


def get_land_or_sea(lat, lon):
    try:
        res = rg.search((lat, lon), mode=1)[0]
        country = res.get("cc", "??")
        city = res.get("name", "")
        admin = res.get("admin1", "")
        if country == "??" or str(admin).strip() == "":
            return f"OCEAN ({lat:.2f}, {lon:.2f})"
        low = city.lower() if isinstance(city, str) else ""
        ocean_markers = ("sea","ocean","bay","gulf","strait","channel")
        if any(w in low for w in ocean_markers):
            return f"OCEAN ({lat:.2f}, {lon:.2f})"
        return f"{city}, {admin}, {country}"
    except Exception:
        return f"OCEAN ({lat:.2f}, {lon:.2f})"



def compute_micro_trigger(df, lat, lon,
                          r_km=MICRO_RADIUS_KM,
                          max_age_hours=MICRO_MAX_AGE_HOURS,
                          min_mag=MICRO_MIN_MAG,
                          max_mag=MICRO_MAX_MAG):

    if df.empty:
        return 0.0, "no data"

    now = df["time"].max()
    if pd.isna(now):
        return 0.0, "no timestamps"

    # фильтруем время + маленькие магнитуды
    df2 = df[
        (df["mag"] >= min_mag) &
        (df["mag"] <= max_mag) &
        (df["time"] >= now - pd.Timedelta(hours=max_age_hours))
    ]

    if df2.empty:
        return 0.0, "no micro eqs"

    dists = haversine_km_vec(lat, lon,
                             df2["lat"].values,
                             df2["lon"].values)
    df_local = df2[dists <= r_km]

    if df_local.empty:
        return 0.0, "no micro nearby"

    # интервалы между микро-событиями
    times = np.sort(df_local["time"].values.astype("datetime64[s]"))
    if len(times) < 2:
        return 0.0, "1 micro-event"

    deltas = np.diff(times) / np.timedelta64(1, "s")
    mean_dt = np.mean(deltas) if len(deltas) else 99999

    count = len(times)
    mean_mag = df_local["mag"].mean()

    # 🔥 Micro Trigger Index
    mti = count * mean_mag * (1.0 / (mean_dt + 1e-6))

    if mti < 1e-5:
        msg = "weak micro activity"
    elif mti < 1e-4:
        msg = "moderate micro clustering"
    else:
        msg = "STRONG micro-trigger"

    return float(mti), msg

def estimate_time_window(mnow, m0, days_passed, mti):

    if mnow < TIME_MAG_THRESHOLD:
        return ""

    if days_passed <= 0:
        return ""

    if m0 is None:
        return ""

    rate = (mnow - m0) / float(days_passed)
    if rate <= 0:
        return ""

    # оценка достижения M6, если меньше 6
    if mnow < 6.0:
        t_hit_days = (6.0 - mnow) / rate
        if t_hit_days < 0:
            t_hit_days = 0
    else:
        t_hit_days = 0

    # MTI → окно времени (центральная часть модели)
    if mti < 1e-5:
        window = "48–72 hours"
    elif mti < 1e-4:
        window = "24–48 hours"
    else:
        window = "0–24 hours"

    return f"<br><b>Time-window:</b> {window}"

def build_and_save_globe(start_date=None, fetch_days=FETCH_DAYS, min_fetch_mag=MIN_MAG_FETCH,
                         nx=NX, ny=NY, l_km=L_KM, l_depth=L_DEPTH,
                         lam=LAM, days_ahead_max=MAX_DAYS_AHEAD):
    log("=== Earthquake predictor run at", utcnow_str(), "===")
    df = fetch_usgs_multi(fetch_days, minmagnitude=min_fetch_mag, start_date=start_date)
    if df.empty:
        raise RuntimeError("No earthquake data available. Try increasing FETCH_DAYS or check network.")
    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)
    df["depth_km"] = df["depth_km"].fillna(0.0).astype(float)
    df["mag"] = df["mag"].astype(float)
    df["E_J"] = df["mag"].apply(energy_from_mag)
    df["depth_factor"] = df["depth_km"].apply(lambda z: depth_attenuation_factor(z, Ldepth=l_depth))
    df["E_eff_J"] = df["E_J"] * df["depth_factor"]
    min_lat = float(df["lat"].min()) - GRID_MARGIN_DEG
    max_lat = float(df["lat"].max()) + GRID_MARGIN_DEG
    min_lon = float(df["lon"].min()) - GRID_MARGIN_DEG
    max_lon = float(df["lon"].max()) + GRID_MARGIN_DEG
    lat_vals = np.linspace(min_lat, max_lat, ny)
    lon_vals = np.linspace(min_lon, max_lon, nx)
    Lon, Lat = np.meshgrid(lon_vals, lat_vals)
    grid_pts = np.column_stack((Lat.ravel(), Lon.ravel()))
    n_pts = grid_pts.shape[0]
    ev_lat = df["lat"].values; ev_lon = df["lon"].values
    n_ev = len(df)
    log("Building kernel matrix A for", n_ev, "events x", n_pts, "grid cells ...")
    A = np.empty((n_ev, n_pts), dtype=np.float64)
    for k in range(n_ev):
        d = haversine_km_vec(ev_lat[k], ev_lon[k], grid_pts[:,0], grid_pts[:,1])
        A[k, :] = np.exp(-d / l_km)
    AtA = A.T @ A
    frames = []
    direct_mag_day0 = None
    inv_mag_day0 = None
    # Precompute now for time extrapolation base
    base_now = datetime.utcnow()
    for days in range(0, days_ahead_max + 1):
        log("Computing frame for days ahead:", days)
        y_time = time_decay(df["E_eff_J"].values, days, tau=TAU_DAYS)
        direct_grid = (A.T @ y_time).reshape((ny, nx))
        rhs = A.T @ y_time
        Mmat = AtA + lam * np.eye(n_pts)
        s_hat = linalg.solve(Mmat, rhs)
        s_hat = np.maximum(s_hat, 0.0)
        s_grid = s_hat.reshape((ny, nx))
        direct_mag = mag_from_energy(direct_grid)
        inv_mag = mag_from_energy(s_grid)
        if days == 0:
            direct_mag_day0 = direct_mag.copy()
            inv_mag_day0 = inv_mag.copy()
        mask_direct = (direct_mag >= M_MIN_SHOW)
        mask_inv = (inv_mag >= M_MIN_SHOW)
        new_direct_mask = np.zeros_like(mask_direct, dtype=bool)
        new_inv_mask = np.zeros_like(mask_inv, dtype=bool)
        if days > 0 and direct_mag_day0 is not None:
            new_direct_mask = (direct_mag >= M_MIN_SHOW) & (direct_mag_day0 < M_MIN_SHOW)
            new_inv_mask = (inv_mag >= M_MIN_SHOW) & (inv_mag_day0 < M_MIN_SHOW)
        d_idx = np.argwhere(mask_direct)
        if len(d_idx):
            d_lats = lat_vals[d_idx[:,0]]
            d_lons = lon_vals[d_idx[:,1]]
            d_mags = direct_mag[mask_direct]
            d_E = 10**(1.5*d_mags + 4.8)
            d_norm = d_E / (d_E.max() if d_E.max() > 0 else 1.0)
            d_sizes = ((6 + 20 * d_norm) * ICON_SCALE_DIRECT).tolist()
        else:
            d_lats = np.array([]); d_lons = np.array([]); d_mags = np.array([]); d_sizes = []
        i_idx = np.argwhere(mask_inv)
        if len(i_idx):
            i_lats = lat_vals[i_idx[:,0]]
            i_lons = lon_vals[i_idx[:,1]]
            i_mags = inv_mag[mask_inv]
            i_E = 10**(1.5*i_mags + 4.8)
            i_norm = i_E / (i_E.max() if i_E.max() > 0 else 1.0)
            i_sizes = ((6 + 20 * i_norm) * ICON_SCALE_INV).tolist()
        else:
            i_lats = np.array([]); i_lons = np.array([]); i_mags = np.array([]); i_sizes = []
        nd_idx = np.argwhere(new_direct_mask)
        if len(nd_idx):
            nd_lats = lat_vals[nd_idx[:,0]]
            nd_lons = lon_vals[nd_idx[:,1]]
            nd_mags = direct_mag[new_direct_mask]
            nd_sizes = ((8 + nd_mags * 3) * ICON_SCALE_NEW).tolist()
        else:
            nd_lats = np.array([]); nd_lons = np.array([]); nd_mags = np.array([]); nd_sizes = []
        ni_idx = np.argwhere(new_inv_mask)
        if len(ni_idx):
            ni_lats = lat_vals[ni_idx[:,0]]
            ni_lons = lon_vals[ni_idx[:,1]]
            ni_mags = inv_mag[new_inv_mask]
            ni_sizes = ((8 + ni_mags * 3) * ICON_SCALE_NEW).tolist()
        else:
            ni_lats = np.array([]); ni_lons = np.array([]); ni_mags = np.array([]); ni_sizes = []
        dh_lats=[]; dh_lons=[]
        ih_lats=[]; ih_lons=[]
        E_min = energy_from_mag(E_MIN_MAG)
        if len(d_idx):
            for latc, lonc, magc in zip(d_lats, d_lons, d_mags):
                Ecell = 10**(1.5*magc + 4.8)
                R = decay_radius_km_from_energy(Ecell, E_min=E_min, L=l_km)
                if R <= 0: continue
                latc_list, lonc_list = circle_polygon(float(latc), float(lonc), R, npoints=48)
                dh_lats += latc_list + [None]; dh_lons += lonc_list + [None]
        if len(i_idx):
            for latc, lonc, magc in zip(i_lats, i_lons, i_mags):
                Ecell = 10**(1.5*magc + 4.8)
                R = decay_radius_km_from_energy(Ecell, E_min=E_min, L=l_km)
                if R <= 0: continue
                latc_list, lonc_list = circle_polygon(float(latc), float(lonc), R, npoints=48)
                ih_lats += latc_list + [None]; ih_lons += lonc_list + [None]
        big = df[df["mag"] >= 4.0]
        traces = []
        # Direct with plasticity & hover (also compute time estimate if mag >=6)
        direct_texts=[]; direct_colors=[]; direct_sizes2=[]
        for latv, lonv, magv in zip(d_lats, d_lons, d_mags):
            b, plast = region_b_value(df, float(latv), float(lonv), radius_km=BVALUE_RADIUS_KM)
            if plast < 0.4: color="red"
            elif plast < 0.8: color="orange"
            else: color="green"
            size = (6 + magv*3) * (1.15 + (1.0 - plast)) * ICON_SCALE_DIRECT
            direct_colors.append(color); direct_sizes2.append(size)
            loc = get_land_or_sea(float(latv), float(lonv))
            # tsunami flag (using conservative criteria)
            is_ocean = ("OCEAN" in loc.upper())
            depth_est = 20.0
            tsunami_msg = ""
            if is_ocean and magv >= 6.5 and depth_est <= 50:
                tsunami_msg = "<br><b>* Hey, this is dangerous — possible tsunami risk *</b>"
            # Time prediction for M>=6: compute linear growth rate from day0 to current day
            time_msg = ""
            try:
                # find grid indices
                i_idx_closest = (np.abs(lat_vals - float(latv))).argmin()
                j_idx_closest = (np.abs(lon_vals - float(lonv))).argmin()
                M0 = float(inv_mag_day0[i_idx_closest, j_idx_closest]) if inv_mag_day0 is not None else None
                Mnow = float(magv)
                if M0 is not None and days > 0:
                    rate = (Mnow - M0) / float(days)  # mag per day
                    if rate > 0:
                        if Mnow >= 6.0:
                            # already at/above 6 -> timestamp is now (approx)
                            hit_time = base_now + timedelta(days=days)
                            time_msg = f"<br><b>Predicted time (≈):</b> {hit_time.strftime('%Y-%m-%d %H:%M')}"
                        else:
                            t_hit_days = (6.0 - Mnow) / rate
                            if 0 <= t_hit_days <= 30:  # only show plausible near-term estimates
                                hit_time = base_now + timedelta(days=days + t_hit_days)
                                time_msg = f"<br><b>Predicted time:</b> {hit_time.strftime('%Y-%m-%d %H:%M')}"
                # else: no valid prediction
            except Exception:
                time_msg = ""
            mti, mti_info = compute_micro_trigger(df,
                                                  float(latv),
                                                  float(lonv),
                                                  r_km=MICRO_RADIUS_KM,
                                                  max_age_hours=MICRO_MAX_AGE_HOURS)

            # Time-window
            time_window_msg = estimate_time_window(
                mnow=float(magv),
                m0=float(inv_mag_day0[i_idx_closest, j_idx_closest]) if inv_mag_day0 is not None else None,
                days_passed=days,
                mti=mti
            )

            direct_texts.append(
                f"<b>Direct M_pred={magv:.2f}</b>"
                f"<br>Depth_est: {depth_est:.1f} km"
                f"<br>{loc}{tsunami_msg}{time_msg}"
                f"{time_window_msg}"
                f"<br>b={b if b else 'N/A'}, plast={plast:.2f}"
                f"<br>Micro-trigger: {mti:.3e} — {mti_info}"
            )
        traces.append(go.Scattergeo(
            lon = d_lons.tolist(),
            lat = d_lats.tolist(),
            mode = "markers",
            marker = dict(size = direct_sizes2, color = direct_colors, opacity=0.92, line=dict(width=0.6, color="black")),
            name = "Direct candidates (plasticity)",
            hoverinfo = "text",
            text = direct_texts
        ))
        # Inversion with plasticity & hover (with time prediction)
        inv_texts=[]; inv_colors=[]; inv_sizes2=[]
        for latv, lonv, magv in zip(i_lats, i_lons, i_mags):
            b, plast = region_b_value(df, float(latv), float(lonv), radius_km=BVALUE_RADIUS_KM)
            if plast < 0.4: color="red"
            elif plast < 0.8: color="orange"
            else: color="green"
            size = (6 + magv*3) * (1.15 + (1.0 - plast)) * ICON_SCALE_INV
            inv_colors.append(color); inv_sizes2.append(size)
            loc = get_land_or_sea(float(latv), float(lonv))
            is_ocean = ("OCEAN" in loc.upper())
            depth_est = 20.0
            tsunami_msg = ""
            if is_ocean and magv >= 6.5 and depth_est <= 50:
                tsunami_msg = "<br><b>* Hey, this is dangerous — possible tsunami risk *</b>"
            time_msg = ""
            try:
                i_idx_closest = (np.abs(lat_vals - float(latv))).argmin()
                j_idx_closest = (np.abs(lon_vals - float(lonv))).argmin()
                M0 = float(inv_mag_day0[i_idx_closest, j_idx_closest]) if inv_mag_day0 is not None else None
                Mnow = float(magv)
                if M0 is not None and days > 0:
                    rate = (Mnow - M0) / float(days)
                    if rate > 0:
                        if Mnow >= 6.0:
                            hit_time = base_now + timedelta(days=days)
                            time_msg = f"<br><b>Predicted time (≈):</b> {hit_time.strftime('%Y-%m-%d %H:%M')}"
                        else:
                            t_hit_days = (6.0 - Mnow) / rate
                            if 0 <= t_hit_days <= 30:
                                hit_time = base_now + timedelta(days=days + t_hit_days)
                                time_msg = f"<br><b>Predicted time:</b> {hit_time.strftime('%Y-%m-%d %H:%M')}"
            except Exception:
                time_msg = ""
            inv_texts.append(f"<b>Inversion M_pred={magv:.2f}</b><br>Depth_est: {depth_est:.1f} km<br>{loc}{tsunami_msg}{time_msg}<br>b={b if b else 'N/A'}, plast={plast:.2f}")
        traces.append(go.Scattergeo(
            lon = i_lons.tolist(),
            lat = i_lats.tolist(),
            mode = "markers",
            marker = dict(symbol="diamond", size = inv_sizes2, color = inv_colors, opacity=0.92, line=dict(width=0.6, color="black")),
            name = "Inversion candidates (plasticity)",
            hoverinfo = "text",
            text = inv_texts
        ))
        # New predicted (direct) with plasticity
        new_direct_texts=[]; new_direct_colors=[]; new_direct_sizes=[]
        for latv, lonv, magv in zip(nd_lats, nd_lons, nd_mags):
            b, plast = region_b_value(df, float(latv), float(lonv), radius_km=BVALUE_RADIUS_KM)
            if plast < 0.4: col="red"
            elif plast < 0.8: col="orange"
            else: col="green"
            size = (8 + magv*3) * (1.05 + (1.0 - plast)) * ICON_SCALE_NEW
            new_direct_colors.append(col); new_direct_sizes.append(size)
            time_msg = ""
            try:
                i_idx_closest = (np.abs(lat_vals - float(latv))).argmin()
                j_idx_closest = (np.abs(lon_vals - float(lonv))).argmin()
                M0 = float(inv_mag_day0[i_idx_closest, j_idx_closest]) if inv_mag_day0 is not None else None
                Mnow = float(magv)
                if M0 is not None and days > 0:
                    rate = (Mnow - M0) / float(days)
                    if rate > 0:
                        if Mnow >= 6.0:
                            hit_time = base_now + timedelta(days=days)
                            time_msg = f"<br><b>Pred time (≈):</b> {hit_time.strftime('%Y-%m-%d %H:%M')}"
                        else:
                            t_hit_days = (6.0 - Mnow) / rate
                            if 0 <= t_hit_days <= 30:
                                hit_time = base_now + timedelta(days=days + t_hit_days)
                                time_msg = f"<br><b>Pred time:</b> {hit_time.strftime('%Y-%m-%d %H:%M')}"
            except Exception:
                time_msg = ""
            new_direct_texts.append(f"<b>New Direct M_pred={magv:.2f}</b>{time_msg}<br>b={b if b else 'N/A'}, plast={plast:.2f}")
        traces.append(go.Scattergeo(
            lon = nd_lons.tolist(),
            lat = nd_lats.tolist(),
            mode = "markers",
            marker = dict(symbol="star", size = new_direct_sizes, color = new_direct_colors, opacity=0.95, line=dict(width=0.6, color="black")),
            name = "New predicted (direct)",
            hoverinfo = "text",
            text = new_direct_texts
        ))
        # New predicted (inv)
        new_inv_texts=[]; new_inv_colors=[]; new_inv_sizes=[]
        for latv, lonv, magv in zip(ni_lats, ni_lons, ni_mags):
            b, plast = region_b_value(df, float(latv), float(lonv), radius_km=BVALUE_RADIUS_KM)
            if plast < 0.4: col="red"
            elif plast < 0.8: col="orange"
            else: col="green"
            size = (8 + magv*3) * (1.05 + (1.0 - plast)) * ICON_SCALE_NEW
            new_inv_colors.append(col); new_inv_sizes.append(size)
            time_msg = ""
            try:
                i_idx_closest = (np.abs(lat_vals - float(latv))).argmin()
                j_idx_closest = (np.abs(lon_vals - float(lonv))).argmin()
                M0 = float(inv_mag_day0[i_idx_closest, j_idx_closest]) if inv_mag_day0 is not None else None
                Mnow = float(magv)
                if M0 is not None and days > 0:
                    rate = (Mnow - M0) / float(days)
                    if rate > 0:
                        if Mnow >= 6.0:
                            hit_time = base_now + timedelta(days=days)
                            time_msg = f"<br><b>Pred time (≈):</b> {hit_time.strftime('%Y-%m-%d %H:%M')}"
                        else:
                            t_hit_days = (6.0 - Mnow) / rate
                            if 0 <= t_hit_days <= 30:
                                hit_time = base_now + timedelta(days=days + t_hit_days)
                                time_msg = f"<br><b>Pred time:</b> {hit_time.strftime('%Y-%m-%d %H:%M')}"
            except Exception:
                time_msg = ""
            new_inv_texts.append(f"<b>New Inversion M_pred={magv:.2f}</b>{time_msg}<br>b={b if b else 'N/A'}, plast={plast:.2f}")
        traces.append(go.Scattergeo(
            lon = ni_lons.tolist(),
            lat = ni_lats.tolist(),
            mode = "markers",
            marker = dict(symbol="triangle-up", size = new_inv_sizes, color = new_inv_colors, opacity=0.95, line=dict(width=0.6, color="black")),
            name = "New predicted (inversion)",
            hoverinfo = "text",
            text = new_inv_texts
        ))
        # Real USGS events (M>=4) with depth color
        if not big.empty:
            event_texts = [f"M: {m:.1f}<br>Depth: {d:.1f} km<br>{get_land_or_sea(lat, lon)}" for m,d,lat,lon in zip(big["mag"].tolist(), big["depth_km"].tolist(), big["lat"].tolist(), big["lon"].tolist())]
            traces.append(go.Scattergeo(
                lon = big["lon"].tolist(),
                lat = big["lat"].tolist(),
                mode = "markers",
                marker = dict(symbol="x", size = ((6 + big["mag"]*ICON_SCALE_USGS).tolist() if hasattr(big["mag"], 'tolist') else [6]), color = big["depth_km"].tolist(), colorscale="Portland", cmin=0, cmax=700, opacity=0.95, colorbar=dict(title="Depth (km)")),
                name = f"USGS events (M≥4) [{fetch_days}d]",
                hoverinfo = "text",
                text = event_texts
            ))
        else:
            traces.append(go.Scattergeo(lon=[], lat=[], mode="markers", marker=dict(size=[]), name=f"USGS events (M≥4) [{fetch_days}d]", hoverinfo="text", text=[]))
        traces.append(go.Scattergeo(lon = dh_lons, lat = dh_lats, mode="lines", line=dict(width=1, color="rgba(255,80,80,0.25)"), name="Direct halos", hoverinfo="none", showlegend=False))
        traces.append(go.Scattergeo(lon = ih_lons, lat = ih_lats, mode="lines", line=dict(width=1, color="rgba(30,144,255,0.25)"), name="Inversion halos", hoverinfo="none", showlegend=False))
        frame = go.Frame(data=traces, name=str(days), layout=go.Layout(title_text=f"Days ahead: {days}"))
        frames.append(frame)
    # end frames loop

    fig = go.Figure(data=frames[0].data if frames else [], frames=frames)
    steps = []
    for i in range(0, days_ahead_max + 1):
        steps.append(dict(method="animate", args=[[str(i)], {"mode":"immediate", "frame":{"duration":0, "redraw":True}, "transition":{"duration":0}}], label=str(i)))
    sliders = [dict(active=0, currentvalue={"prefix":"Days ahead: "}, pad={"t":50}, steps=steps)]
    updatemenus = [
        dict(type="buttons", direction="left", x=0.02, y=0.12, showactive=True,
             buttons=[
                 dict(label="Both", method="update", args=[{"visible":[True, True, True, True, True, True, True]}, {"title":"Both"}]),
                 dict(label="Direct", method="update", args=[{"visible":[True, False, False, True, True, False, False]}, {"title":"Direct only"}]),
                 dict(label="Inversion", method="update", args=[{"visible":[False, True, False, False, False, True, True]}, {"title":"Inversion only"}]),
             ]),
        dict(type="buttons", direction="down", x=0.02, y=0.02, showactive=True,
             buttons=[
                 dict(label="Hide USGS", method="update", args=[{"visible":[None, None, False, None, None, None, None]}, {"title":"USGS hidden"}]),
                 dict(label="Show USGS", method="update", args=[{"visible":[None, None, True, None, None, None, None]}, {"title":"USGS shown"}]),
             ])
    ]
    fig.update_layout(title=f"Predicted hotspots M≥{M_MIN_SHOW} (depth-aware + plasticity) — last {fetch_days}d", geo=dict(projection_type="orthographic", showland=True, landcolor="rgb(230,230,230)", showocean=True, oceancolor="rgb(180,200,250)", showcountries=True), sliders=sliders, updatemenus=updatemenus, legend=dict(x=0.01, y=0.99))
    fig.update_layout(annotations=[dict(text="Use legend/buttons. Slider: days ahead (0..30).", showarrow=False, x=0.5, y=0.01, xref="paper", yref="paper", font=dict(size=12))])
    fig.write_html(OUTPUT_HTML, auto_open=True)
    log("Saved interactive globe to", OUTPUT_HTML)
    if frames:
        y0 = time_decay(df["E_eff_J"].values, 0, tau=TAU_DAYS)
        direct0 = (A.T @ y0).reshape((ny, nx))
        rhs0 = A.T @ y0
        s0 = linalg.solve(AtA + lam * np.eye(n_pts), rhs0).reshape((ny, nx))
        out = pd.DataFrame({"lat":grid_pts[:,0], "lon":grid_pts[:,1], "direct_E":direct0.ravel(), "direct_M":mag_from_energy(direct0.ravel()), "inv_E":s0.ravel(), "inv_M":mag_from_energy(s0.ravel())})
        out_csv = "quake_predictor_grid_day0_depth_plasticity.csv"
        out.to_csv(out_csv, index=False)
        log("Saved day0 grid CSV:", out_csv)
    log("=== run finished at", utcnow_str(), "===")


if __name__ == "__main__":
    start_date = None
    fetch_days = FETCH_DAYS
    if len(sys.argv) >= 2:
        start_date = sys.argv[1]
    if len(sys.argv) >= 3:
        try:
            fetch_days = int(sys.argv[2])
        except Exception:
            fetch_days = FETCH_DAYS
    try:
        build_and_save_globe(start_date=start_date, fetch_days=fetch_days, nx=NX, ny=NY, l_km=L_KM, l_depth=L_DEPTH, lam=LAM, days_ahead_max=MAX_DAYS_AHEAD)
    except Exception as e:
        print("Error in main:", e)
        raise



