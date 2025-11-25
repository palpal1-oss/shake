#!/usr/bin/env python3
"""
deep_shake_v2.py

"""

import os, sys, time, math, json, argparse
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go
import requests

# -------------------- Config / Defaults --------------------
LOCAL_CACHE_DEFAULT = "/mnt/data/usgs_week_cache.json"
OUTPUT_HTML_DEFAULT = "deepquake_v2_globe.html"
MAX_DAYS_WINDOW = 365 * 30  # clamp intervals to 30 years max
MIN_DAYS_WINDOW = 7         # minimum realistic window to report
DEFAULT_DAYS = 60

# -------------------- Helpers --------------------
def now_utc():
    return datetime.now(timezone.utc)

def energy_from_mag(M):
    return 10.0 ** (1.5 * M + 4.8)

def mag_from_energy(E):
    Epos = np.maximum(E, 1e-30)
    return (np.log10(Epos) - 4.8) / 1.5

def haversine_km_vec(lat1, lon1, lat_arr, lon_arr):
    # vectorized haversine distance
    R = 6371.0
    lat1r = math.radians(float(lat1)); lon1r = math.radians(float(lon1))
    lat2r = np.radians(lat_arr); lon2r = np.radians(lon_arr)
    dlat = lat2r - lat1r; dlon = lon2r - lon1r
    a = np.sin(dlat/2.0)**2 + np.cos(lat1r)*np.cos(lat2r)*np.sin(dlon/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def clamp_days(x, min_days=MIN_DAYS_WINDOW, max_days=MAX_DAYS_WINDOW):
    try:
        if not np.isfinite(x):
            return max_days
        xi = float(x)
    except Exception:
        return max_days
    if xi <= 0:
        return min_days
    if xi < min_days:
        return int(round(min_days))
    if xi > max_days:
        return int(round(max_days))
    return int(round(xi))

def days_to_date_str_utc(start_dt, days):
    # start_dt is timezone-aware datetime
    try:
        dt = start_dt + timedelta(days=int(days))
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return "N/A"

# -------------------- USGS fetch/load --------------------
def fetch_usgs_day_csv(start, end, minmagnitude=1.0, timeout=30):
    params = {"format":"csv","starttime":start,"endtime":end,"minmagnitude":minmagnitude,"orderby":"time","limit":20000}
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query?" + urlencode(params)
    r = requests.get(url, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"USGS fetch failed: {r.status_code}")
    text = r.text
    if len(text) < 50:
        raise RuntimeError("USGS returned empty")
    df = pd.read_csv(pd.io.common.StringIO(text))
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc == "latitude": col_map[c] = "lat"
        if lc == "longitude": col_map[c] = "lon"
        if lc == "depth": col_map[c] = "depth_km"
        if lc == "mag": col_map[c] = "mag"
        if lc == "time": col_map[c] = "time"
    df = df.rename(columns=col_map)
    if "depth_km" not in df.columns:
        df["depth_km"] = 0.0
    df["depth_km"] = df["depth_km"].fillna(0.0).astype(float)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df[["lat","lon","depth_km","mag","time"]]

def load_events(days_back=DEFAULT_DAYS, min_mag=2.5, start_date=None, use_local_if_present=True, local_path=LOCAL_CACHE_DEFAULT):
    # Prefer local cache if present (user-uploaded)
    if use_local_if_present and os.path.exists(local_path):
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "features" in data:
                rows = []
                for feat in data["features"]:
                    prop = feat.get("properties", {})
                    geom = feat.get("geometry", {})
                    coords = geom.get("coordinates", [None, None, None])
                    rows.append({
                        "time": prop.get("time"),
                        "lat": coords[1] if len(coords) > 1 else prop.get("latitude"),
                        "lon": coords[0] if len(coords) > 0 else prop.get("longitude"),
                        "depth_km": coords[2] if len(coords) > 2 else prop.get("depth"),
                        "mag": prop.get("mag") if prop.get("mag") is not None else prop.get("magnitude")
                    })
                df = pd.DataFrame(rows)
                if "time" in df.columns:
                    try:
                        df["time"] = pd.to_datetime(df["time"], unit="ms", errors="coerce").dt.tz_localize('UTC')
                    except Exception:
                        df["time"] = pd.to_datetime(df["time"], errors="coerce").dt.tz_localize('UTC')
                for c in ["lat","lon","mag","depth_km"]:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df.dropna(subset=["lat","lon","mag"]).reset_index(drop=True)
                return df
        except Exception as e:
            print("[load] failed to parse local cache:", e, file=sys.stderr)
    # Fetch day-by-day
    print("[load] fetching USGS day-by-day...")
    if start_date is None:
        now = datetime.utcnow()
    else:
        now = datetime.strptime(start_date, "%Y-%m-%d")
    dfs = []
    for k in range(days_back):
        s = (now - timedelta(days=k+1)).strftime("%Y-%m-%d")
        e = (now - timedelta(days=k)).strftime("%Y-%m-%d")
        try:
            df_day = fetch_usgs_day_csv(s, e, minmagnitude=min_mag)
            dfs.append(df_day)
            time.sleep(0.15)
        except Exception as exc:
            print(f"[fetch] {s}->{e} failed: {exc}", file=sys.stderr)
            time.sleep(0.5)
    if not dfs:
        raise RuntimeError("No data fetched from USGS")
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["lat","lon","mag","time"]).sort_values("time").reset_index(drop=True)
    # Ensure timezone-aware times in UTC when possible
    if "time" in df.columns:
        try:
            df["time"] = pd.to_datetime(df["time"], errors="coerce").dt.tz_localize('UTC')
        except Exception:
            pass
    # cache for convenience
    try:
        df.to_json(local_path, orient="records", date_format="iso")
        print(f"[load] cached to {local_path}")
    except Exception:
        pass
    return df

# -------------------- Grid & Kernel --------------------
def build_grid_around_events(df, nx=60, ny=40, pad_deg=1.0):
    min_lat = float(df['lat'].min()) - pad_deg
    max_lat = float(df['lat'].max()) + pad_deg
    min_lon = float(df['lon'].min()) - pad_deg
    max_lon = float(df['lon'].max()) + pad_deg
    lat_vals = np.linspace(min_lat, max_lat, ny)
    lon_vals = np.linspace(min_lon, max_lon, nx)
    Lon, Lat = np.meshgrid(lon_vals, lat_vals)
    pts = np.column_stack((Lat.ravel(), Lon.ravel()))
    return lat_vals, lon_vals, pts

def build_sparse_A(events, grid_pts, model=1, L_h=150.0, H_depth=1000.0, horiz_radius_km=1000.0):
    n_ev = len(events); n_pts = grid_pts.shape[0]
    rows = []; cols = []; vals = []
    for ei, ev in events.iterrows():
        try:
            elat = float(ev['lat']); elon = float(ev['lon']); ed = float(ev.get('depth_km', 0.0))
        except Exception:
            continue
        dists = haversine_km_vec(elat, elon, grid_pts[:,0], grid_pts[:,1])
        mask = dists <= horiz_radius_km
        if not np.any(mask):
            continue
        idxs = np.nonzero(mask)[0]
        horiz = np.exp(-dists[mask] / (L_h + 1e-12))
        if model == 1:
            depth_factor = math.exp(-ed / (H_depth + 1e-12)) if ed >= 0 else 1.0
            vals_local = (horiz * depth_factor).tolist()
        else:
            depth_factor = math.exp(-ed / (H_depth * 0.9 + 1e-12))
            vals_local = (horiz * depth_factor * (1.0 + 0.1 * np.exp(-((ed - 200.0)/200.0)**2))).tolist()
        rows.extend([ei] * len(idxs)); cols.extend(idxs.tolist()); vals.extend(vals_local)
    if len(vals) == 0:
        return sparse.csr_matrix((n_ev, n_pts))
    A = sparse.coo_matrix((vals, (rows, cols)), shape=(n_ev, n_pts)).tocsr()
    return A

# -------------------- Solver --------------------
def solve_tikhonov_cg(A, y, lam=1e10, rtol=1e-6, maxiter=1000):
    n = A.shape[1]; At = A.transpose()
    At_y = At.dot(y)
    def matvec(x):
        return At.dot(A.dot(x)) + lam * x
    linop = LinearOperator((n,n), matvec=matvec, dtype=np.float64)
    x0 = np.zeros(n, dtype=np.float64)
    sol, info = cg(linop, At_y, x0=x0, rtol=rtol, atol=0.0, maxiter=maxiter)
    if info != 0:
        print(f"[solve] CG info={info}", file=sys.stderr)
    sol = np.maximum(sol, 0.0)
    return sol

# -------------------- Candidates & Clustering --------------------
def find_candidates_from_grid(Mgrid, lat_vals, lon_vals, threshold=6.5):
    mask = np.isfinite(Mgrid) & (Mgrid >= threshold)
    idxs = np.argwhere(mask)
    if idxs.size == 0:
        return np.empty((0,)), np.empty((0,)), np.empty((0,))
    lats = lat_vals[idxs[:,0]]; lons = lon_vals[idxs[:,1]]; mags = Mgrid[mask]
    return lats, lons, mags

def cluster_exits(lats, lons, mags, eps_km=25.0, min_samples=1):
    if lats.size == 0:
        return np.empty((0,2)), np.empty((0,))
    deg_per_km = 1.0 / 111.0
    eps_deg = eps_km * deg_per_km
    coords = np.column_stack((lats, lons))
    db = DBSCAN(eps=eps_deg, min_samples=min_samples).fit(coords)
    labels = db.labels_
    clusters = {}
    for i, lab in enumerate(labels):
        clusters.setdefault(lab, []).append((lats[i], lons[i], mags[i]))
    centroids = []; rep_mags = []
    for lab, items in clusters.items():
        arr = np.array(items)
        mean_lat = float(arr[:,0].mean()); mean_lon = float(arr[:,1].mean())
        rep_mag = float(arr[:,2].max())
        centroids.append((mean_lat, mean_lon)); rep_mags.append(rep_mag)
    return np.array(centroids), np.array(rep_mags)

# -------------------- Linking & Interval estimator --------------------
def link_deep_to_surface(deep_peaks, centroids, max_link_km=800.0):
    links = {i:[] for i in range(len(deep_peaks))}
    if len(deep_peaks) == 0 or len(centroids) == 0:
        return links
    scoords = np.array(centroids)
    for di, dp in enumerate(deep_peaks):
        dlat, dlon, dmag, dE, dE_dt = dp
        dists = haversine_km_vec(dlat, dlon, scoords[:,0], scoords[:,1])
        linked = np.where(dists <= max_link_km)[0].tolist()
        links[di] = linked
    return links

def estimate_release_window_safe(E, dE_dt, safety_factor_min=0.6, safety_factor_max=1.8, min_days=MIN_DAYS_WINDOW, max_days=MAX_DAYS_WINDOW):
    # E: J, dE_dt: J/day
    if not np.isfinite(E) or not np.isfinite(dE_dt) or E <= 0 or dE_dt <= 0:
        return None
    # raw days
    raw_days = E / dE_dt
    t_min = safety_factor_min * raw_days
    t_max = safety_factor_max * raw_days
    # clamp to sensible bounds
    t_min_c = clamp_days(t_min, min_days, max_days)
    t_max_c = clamp_days(t_max, min_days, max_days)
    t_most = int(round(0.5 * (t_min_c + t_max_c)))
    return int(t_min_c), int(t_most), int(t_max_c)

# -------------------- Plotting --------------------
def safe_marker_size(arr, base=6.0, scale=6.0):
    raw = base + scale * (np.array(arr) - 6.0)
    raw = np.asarray(raw, dtype=float)
    raw = np.where(np.isfinite(raw), raw, base)
    raw = np.maximum(raw, 2.0)
    return raw.tolist()

def plot_results(surface_groups, deep_peaks, release_info, out_html=OUTPUT_HTML_DEFAULT):
    fig = go.Figure()
    palette = [
        "#e6194b","#3cb44b","#ffe119","#4363d8","#f58231","#911eb4","#46f0f0","#f032e6",
        "#bcf60c","#fabebe","#008080","#e6beff","#9a6324","#fffac8","#800000"
    ]
    # surface groups
    for gi, grp in enumerate(surface_groups):
        lat, lon = grp['cent']; mag = grp['mag']; color = grp.get('color', palette[gi % len(palette)]); members = grp.get('members', [])
        size = safe_marker_size([mag])[0]
        hover = f"Pred M≈{mag:.2f}<br>Members: {len(members)}<br>Confidence: {grp.get('conf', 0.0):.2f}"
        fig.add_trace(go.Scattergeo(
            lon=[lon], lat=[lat], mode="markers+text",
            marker=dict(size=size, color=color, opacity=0.9, line=dict(width=0.6,color='black')),
            text=[f"{mag:.2f}"], textposition="middle center", hoverinfo="text", hovertext=hover,
            name=f"Candidate M{mag:.2f}"
        ))
        if members:
            mem_lats = [m[0] for m in members]; mem_lons = [m[1] for m in members]
            fig.add_trace(go.Scattergeo(
                lon=mem_lons, lat=mem_lats, mode="markers",
                marker=dict(size=4, color=color, opacity=0.5),
                hoverinfo="none", showlegend=False
            ))
    # deep peaks markers
    if deep_peaks:
        dp_lats = [p[0] for p in deep_peaks]; dp_lons = [p[1] for p in deep_peaks]; dp_mags=[p[2] for p in deep_peaks]
        fig.add_trace(go.Scattergeo(
            lon=dp_lons, lat=dp_lats, mode="markers+text",
            marker=dict(size=safe_marker_size(dp_mags), color="black", opacity=0.6, symbol="x"),
            text=[f"{m:.2f}" for m in dp_mags], textposition="top center", name="Deep src (proj)",
            hovertext=[f"Deep source projection M≈{m:.2f}" for m in dp_mags]
        ))
    # annotations: add small text traces for windows (safe)
    start = now_utc()
    for info in release_info:
        di = info.get('deep_index')
        if info.get('window') is None:
            continue
        tmin, tmost, tmax = info['window']
        # guard against invalid windows
        if not all(np.isfinite([tmin, tmost, tmax])):
            continue
        deep = deep_peaks[di]
        ann_text = (f"Src#{di} M≈{deep[2]:.2f}<br>"
                    f"Window: {days_to_date_str_utc(start, tmin)} → {days_to_date_str_utc(start, tmax)}<br>"
                    f"Most likely: {days_to_date_str_utc(start, tmost)}")
        # add text trace near point
        fig.add_trace(go.Scattergeo(
            lon=[deep[1] + 0.1], lat=[deep[0] + 0.05], mode="text",
            text=[ann_text], showlegend=False, hoverinfo="none",
            textfont=dict(size=10, color="black")
        ))
    fig.update_layout(
        title="Deepquake v2: predicted super-event exits (honest intervals)",
        geo=dict(projection_type="orthographic", showland=True, landcolor="rgb(230,230,230)",
                 showocean=True, oceancolor="rgb(180,200,250)", showcountries=True),
        legend=dict(x=0.01, y=0.99)
    )
    fig.write_html(out_html, auto_open=True)
    print(f"[plot] saved to {out_html}")

# -------------------- Pipeline --------------------
def run_deepquake(args):
    print("[deepquake_v2] start:", now_utc().isoformat())
    df = load_events(days_back=args.days, min_mag=args.min_mag, start_date=args.date,
                     use_local_if_present=not args.force_fetch, local_path=args.local_cache)
    print(f"[data] loaded {len(df)} events: {df['time'].min()} -> {df['time'].max()}")
    df_surface = df[df['depth_km'] <= args.surface_depth_cut].reset_index(drop=True)
    if df_surface.empty:
        raise RuntimeError("No surface events after depth filter; increase surface_depth_cut or days window.")
    lat_vals, lon_vals, grid_pts = build_grid_around_events(df_surface, nx=args.nx, ny=args.ny, pad_deg=args.pad_deg)
    print(f"[grid] {args.nx}x{args.ny} => {grid_pts.shape[0]} cells")
    events_for_A = df if args.deep_model == 2 else df_surface
    A = build_sparse_A(events_for_A, grid_pts, model=args.deep_model, L_h=args.L_h, H_depth=args.H_depth, horiz_radius_km=args.horiz_radius)
    print(f"[A] built sparse A shape={A.shape}, nnz={A.nnz}")
    E = energy_from_mag(events_for_A['mag'].values)
    depth_f = np.exp(-events_for_A['depth_km'].values / (args.H_depth + 1e-12))
    y = E * depth_f
    print("[solve] Tikhonov inversion via CG...")
    s = solve_tikhonov_cg(A, y, lam=args.lam, rtol=args.cg_rtol, maxiter=args.cg_maxiter)
    grid_E = s.reshape((args.ny, args.nx))
    grid_M = mag_from_energy(grid_E)
    lats, lons, mags = find_candidates_from_grid(grid_M, lat_vals, lon_vals, threshold=args.show_threshold)
    print(f"[candidates] raw cells >= {args.show_threshold}: {len(mags)}")
    centroids, rep_mags = cluster_exits(lats, lons, mags, eps_km=args.cluster_eps_km, min_samples=1)
    print(f"[cluster] centroids found: {len(centroids)}")
    flat = grid_E.ravel()
    top_idx = np.argsort(flat)[-args.top_deep_peaks:][::-1]
    deep_peaks = []
    # compute coarse global proxy for dE/dt to fallback
    try:
        times = pd.to_datetime(events_for_A['time'])
        now = times.max()
        sum7 = energy_from_mag(events_for_A.loc[times >= (now - pd.Timedelta(days=7)), 'mag'].values).sum()
        sum14 = energy_from_mag(events_for_A.loc[times >= (now - pd.Timedelta(days=14)), 'mag'].values).sum()
        global_dE_dt = max(1e-12, (sum7 - sum14) / 7.0)
    except Exception:
        global_dE_dt = 1e-12
    for idx in top_idx:
        val = float(flat[idx])
        if val <= 0.0:
            continue
        i = int(idx // args.nx); j = int(idx % args.nx)
        mag = float(mag_from_energy(np.array([val]))[0])
        glat = float(lat_vals[i]); glon = float(lon_vals[j])
        # local proxy for dE/dt (safe guarded)
        try:
            ev_lats = events_for_A['lat'].values; ev_lons = events_for_A['lon'].values
            dists = haversine_km_vec(glat, glon, ev_lats, ev_lons)
            times = pd.to_datetime(events_for_A['time'])
            near_mask_7 = (dists <= args.local_growth_km) & (times >= (times.max() - pd.Timedelta(days=7)))
            near_mask_14 = (dists <= args.local_growth_km) & (times >= (times.max() - pd.Timedelta(days=14)))
            E7 = energy_from_mag(events_for_A.loc[near_mask_7, 'mag'].values).sum() if near_mask_7.any() else 0.0
            E14 = energy_from_mag(events_for_A.loc[near_mask_14, 'mag'].values).sum() if near_mask_14.any() else 0.0
            local_dE_dt = max(1e-12, (E7 - E14) / 7.0)
            # if local derivative is too small, fallback to global proxy
            if local_dE_dt < 1e-8:
                local_dE_dt = global_dE_dt
        except Exception:
            local_dE_dt = global_dE_dt
        deep_peaks.append((glat, glon, mag, float(val), local_dE_dt))
    print(f"[deep] deep peaks extracted: {len(deep_peaks)}")
    links = link_deep_to_surface(deep_peaks, centroids, max_link_km=args.link_max_km)
    palette = [
        "#e6194b","#3cb44b","#ffe119","#4363d8","#f58231","#911eb4","#46f0f0","#f032e6",
        "#bcf60c","#fabebe","#008080","#e6beff","#9a6324","#fffac8","#800000"
    ]
    surface_groups = []
    for si, cent in enumerate(centroids):
        assigned = False; color = palette[si % len(palette)]
        for di, linked in links.items():
            if si in linked:
                color = palette[di % len(palette)]; assigned = True; break
        members = []
        if lats.size > 0:
            dists = haversine_km_vec(cent[0], cent[1], lats, lons)
            mask = dists <= args.cluster_eps_km
            for a,b,c in zip(lats[mask], lons[mask], mags[mask]):
                members.append((float(a), float(b), float(c)))
        # confidence heuristic: fraction of linked deep peaks / magnitude scaling (simple)
        conf = 0.5
        for di, linked in links.items():
            if si in linked:
                conf = min(1.0, 0.4 + 0.15 * len(links[di]))
        surface_groups.append({'cent':(float(cent[0]), float(cent[1])), 'mag':float(rep_mags[si]), 'color':color, 'members':members, 'conf':conf})
    # estimate release windows
    release_info = []
    for di, dp in enumerate(deep_peaks):
        glat, glon, mag, Ecell, local_dE_dt = dp
        win = estimate_release_window_safe(Ecell, local_dE_dt, safety_factor_min=0.6, safety_factor_max=1.8)
        release_info.append({'deep_index':di, 'window':win, 'mag':mag, 'E':Ecell, 'dE_dt':local_dE_dt})
    plot_results(surface_groups, [(p[0],p[1],p[2]) for p in deep_peaks], release_info, out_html=args.out_html)
    print("[deepquake_v2] done.")

# -------------------- CLI --------------------
def cli():
    p = argparse.ArgumentParser(description="Deepquake v2: improved deep-source mapper")
    p.add_argument("--date", type=str, default=None, help="end date YYYY-MM-DD for data fetch (default: today)")
    p.add_argument("--days", type=int, default=DEFAULT_DAYS, help="how many days to fetch (recommended 60-120)")
    p.add_argument("--min-mag", type=float, default=2.5, help="min magnitude to fetch (use 2.5+ for stable results)")
    p.add_argument("--local-cache", type=str, default=LOCAL_CACHE_DEFAULT, help="local cache path (use uploaded file if present)")
    p.add_argument("--force-fetch", action="store_true", help="ignore local cache and fetch USGS directly")
    p.add_argument("--nx", type=int, default=60, help="grid X resolution (lon)")
    p.add_argument("--ny", type=int, default=40, help="grid Y resolution (lat)")
    p.add_argument("--pad-deg", type=float, default=1.0, help="pad degrees around event bbox for grid")
    p.add_argument("--L-h", type=float, default=150.0, help="horizontal kernel lengthscale (km)")
    p.add_argument("--H-depth", type=float, default=1000.0, help="depth attenuation lengthscale (km)")
    p.add_argument("--horiz-radius", type=float, default=1000.0, help="max horiz radius for event influence (km)")
    p.add_argument("--lam", type=float, default=1e10, help="Tikhonov regularization lambda")
    p.add_argument("--cg-rtol", type=float, default=1e-6, dest="cg_rtol", help="CG relative tolerance")
    p.add_argument("--cg-maxiter", type=int, default=800, dest="cg_maxiter", help="CG max iter")
    p.add_argument("--show-threshold", type=float, default=6.5, help="M threshold for showing candidates (super-events)")
    p.add_argument("--cluster-eps-km", type=float, default=25.0, help="clustering radius km for grouping exits")
    p.add_argument("--top-deep-peaks", type=int, default=8, help="how many deep peaks to extract (projection grid cells)")
    p.add_argument("--link-max-km", type=float, default=800.0, help="max distance to link surface exit to deep source (km)")
    p.add_argument("--surface-depth-cut", type=float, default=100.0, help="exclude events deeper than this from surface mapping")
    p.add_argument("--local-growth-km", type=float, default=200.0, help="radius for local dE/dt proxy (km)")
    p.add_argument("--deep-model", type=int, default=1, choices=[1,2], help="1 simple depth kernel, 2 channel-like deeper model")
    p.add_argument("--out-html", type=str, default=OUTPUT_HTML_DEFAULT, help="output html file name")
    args = p.parse_args()
    if os.path.exists(LOCAL_CACHE_DEFAULT) and args.local_cache == LOCAL_CACHE_DEFAULT:
        args.local_cache = LOCAL_CACHE_DEFAULT
    run_deepquake(args)

if __name__ == "__main__":
    cli()

