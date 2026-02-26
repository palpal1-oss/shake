import xml.etree.ElementTree as ET
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime, timedelta
from datetime import datetime, timedelta, UTC
import numpy as np


pio.renderers.default = "browser"

# ==============================
# НАСТРОЙКИ
# ==============================
FETCH_DAYS = 10      # сколько дней истории грузим
EVENT_DAYS = 11      # считаем "активными" событиями (звёзды)
MIN_MAG = 1.0

# ==============================
# ПОЛЕ ВЛИЯНИЯ
# ==============================
USE_INFLUENCE_FIELD = True
SIGMA = 5.0        # радиус влияния (градусы)
GRID_STEP = 1.0    # разрешение сетки (меньше = точнее, но медленнее)

STAR_SIZE_FAR = 8     # большие звезды (видно издалека)
STAR_SIZE_NEAR = 3     # нормальный размер

# ==============================
# ПРОГНОЗ
# ==============================
FORECAST_DAYS = [1, 2, 3]   # окна прогноза
TAU_DAYS = 7.0              # время жизни напряжения

# ==============================
# 1. ЧИТАЕМ ВУЛКАНЫ (XML)
# ==============================
print("Loading volcanoes...")

tree = ET.parse("volcanoes.xml")
root = tree.getroot()

volcanoes = []

for v in root.findall("volcano"):
    volcanoes.append({
        "name": v.find("name").text,
        "lat": float(v.find("latitude").text),
        "lon": float(v.find("longitude").text)
    })

df_volcano = pd.DataFrame(volcanoes)

# ==============================
# 2. ЗАГРУЖАЕМ ЗЕМЛЕТРЯСЕНИЯ (ПО ДНЯМ — СТАБИЛЬНО)
# ==============================
def load_earthquakes(days, min_mag):

    end_date = datetime.now(UTC)
    all_data = []

    print(f"Loading earthquakes day-by-day ({days} days)...")

    for i in range(days):

        day_end = end_date - timedelta(days=i)
        day_start = day_end - timedelta(days=1)

        url = (
            "https://earthquake.usgs.gov/fdsnws/event/1/query.csv?"
            f"starttime={day_start.strftime('%Y-%m-%d')}"
            f"&endtime={day_end.strftime('%Y-%m-%d')}"
            f"&minmagnitude={min_mag}"
            "&orderby=time"
        )

        try:
            df_day = pd.read_csv(url)

            if len(df_day) == 0:
                continue

            df_day = df_day.rename(columns={
                "depth": "depth",
                "latitude": "lat",
                "longitude": "lon",
                "mag": "mag",
                "place": "place",
                "time": "time"
            })

            df_day["time"] = pd.to_datetime(df_day["time"], utc=True)

            all_data.append(
                df_day[["lat","lon","mag","place","time","depth"]]
            )

            print(f"Day {i+1}/{days} loaded: {len(df_day)} events")

        except Exception as e:
            print(f"Skipped day {i+1} (network issue)")

    if len(all_data) == 0:
        return pd.DataFrame(
            columns=["lat", "lon", "mag", "place", "time"]
        )

    df = pd.concat(all_data, ignore_index=True)

    print("Total earthquakes loaded:", len(df))
    return df



def build_influence_field(df, sigma, step, future_shift_days=0):

    print(f"Building field (forecast +{future_shift_days}d)...")

    now = datetime.now(UTC) + timedelta(days=future_shift_days)

    lats = np.arange(-90, 90, step)
    lons = np.arange(-180, 180, step)

    grid_lon, grid_lat = np.meshgrid(lons, lats)
    field = np.zeros_like(grid_lat, dtype=float)

    if len(df) == 0:
        return grid_lat, grid_lon, field

    for _, row in df.iterrows():

        # --- временной вес ---
        dt_days = (now - row["time"]).total_seconds() / 86400
        time_weight = np.exp(-(dt_days / TAU_DAYS)**1.6)

        if time_weight < 0.01:
            continue

        # --- расстояние ---
        dlat = grid_lat - row["lat"]
        dlon = grid_lon - row["lon"]
        r2 = dlat**2 + dlon**2

        spatial = np.exp(-r2 / (2 * sigma**2))

        # усиление кластеров (если рядом много событий)
        cluster_boost = 1 + 0.4 * np.exp(-r2 / (8 * sigma ** 2))

        # используем лог-энергию чтобы слабые события тоже влияли
        energy = row["energy_log"] ** 2

        mag_boost = 1 + 0.6 * np.tanh(row["mag"] - 2.5)
        energy *= mag_boost

        local_response = energy * spatial * time_weight * cluster_boost

        field += local_response

        # псевдо-инверсия (выявляет скрытые зоны)
        delta = local_response - np.mean(local_response)
        field += 0.25 * np.maximum(delta, 0)



    # --- мягкое глобальное восстановление поля ---
    field += 0.15 * (
            np.roll(field, 1, axis=0) +
            np.roll(field, -1, axis=0) +
            np.roll(field, 1, axis=1) +
            np.roll(field, -1, axis=1)
    ) / 4

    field = np.power(np.maximum(field, 1e-12), 0.35)

    return grid_lat, grid_lon, field


# ==============================
# НОРМАЛИЗАЦИЯ ПО ЛОКАЛЬНОМУ ФОНУ
# ==============================
def compute_anomaly(field):

    print("Computing anomaly field...")

    # локальное среднее (фон региона)
    background = (
        np.roll(field, 5, axis=0) +
        np.roll(field, -5, axis=0) +
        np.roll(field, 5, axis=1) +
        np.roll(field, -5, axis=1) +
        field
    ) / 5

    # избегаем деления на ноль
    background += 1e-9

    anomaly = field / background

    # мягкое сжатие диапазона
    anomaly = np.power(anomaly, 0.7)

    return anomaly

df_quakes = load_earthquakes(FETCH_DAYS, MIN_MAG)

# ==============================
# ЭНЕРГИЯ СОБЫТИЙ (главный фикс)
# ==============================

# энергия растёт экспоненциально
df_quakes["energy"] = 10 ** (1.5 * df_quakes["mag"] + 4.8)

# логарифм — чтобы поле не взрывалось
df_quakes["energy_log"] = np.log10(df_quakes["energy"] + 1)

# ==============================
# УЧЁТ ГЛУБИНЫ (делаем ОДИН РАЗ!)
# ==============================

L_DEPTH = 70.0  # км

df_quakes["energy_eff"] = (
    df_quakes["energy"]
    * np.exp(-df_quakes["depth"] / L_DEPTH)
)

df_quakes["energy_log"] = np.log10(df_quakes["energy_eff"] + 1)

print("Volcanoes:", len(df_volcano))
print("Earthquakes:", len(df_quakes))

# ==============================
# 3. ВЫДЕЛЯЕМ АКТИВНЫЕ СОБЫТИЯ
# ==============================

event_limit = datetime.now(UTC) - timedelta(days=EVENT_DAYS)

df_events = df_quakes[df_quakes["time"] >= event_limit]
df_old = df_quakes[df_quakes["time"] < event_limit]

print("Recent events:", len(df_events))

if USE_INFLUENCE_FIELD:
    grid_lat, grid_lon, field = build_influence_field(
        df_quakes, SIGMA, GRID_STEP
    )

field = compute_anomaly(field)

# ==============================
# 3.5 СТРОИМ ПРОГНОЗНЫЕ ПОЛЯ
# ==============================

def extract_hotspots(grid_lat, grid_lon, field,
                     percentile=99.7,
                     min_mag=3.5):

    # берём только самые сильные области поля
    limit = np.percentile(field, percentile)
    mask = field >= limit

    lat = grid_lat[mask]
    lon = grid_lon[mask]
    strength = field[mask]

    # --- перевод поля → магнитуда ---
    # обратная Gutenberg–Richter
    mag_pred = (np.log10(strength + 1e-12) - 4.8) / 1.5

    # ФИЛЬТР ПО МАГНИТУДЕ
    keep = mag_pred >= min_mag

    return lat[keep], lon[keep], mag_pred[keep]


forecast_points = {}

for d in FORECAST_DAYS:

    glat, glon, fld = build_influence_field(
        df_quakes,
        SIGMA,
        GRID_STEP,
        future_shift_days=d
    )

    fld = compute_anomaly(fld)

    lat_h, lon_h, mag_h = extract_hotspots(glat, glon, fld)
    forecast_points[d] = (lat_h, lon_h, mag_h)

print(f"Forecast +{d}d points:", len(mag_h))

print("Forecast windows built")

# ==============================
# 4. СТРОИМ ГЛОБУС
# ==============================

fig = go.Figure()

# --- ПОЛЕ ВЛИЯНИЯ ---
if USE_INFLUENCE_FIELD:


    # оставляем только сильные области поля
    threshold = np.percentile(field, 85)

    mask = field >= threshold

    fig.add_trace(
        go.Scattergeo(
            lat=grid_lat[mask],
            lon=grid_lon[mask],
            mode="markers",
            marker=dict(
                size=np.clip(mag_h * 3, 6, 18),
                color=field[mask],
                colorscale="Hot",
                opacity=0.35,
                colorbar=dict(title="Stress")
            ),
            name="Stress Field"
        )
    )

# --- вулканы ---
fig.add_trace(
    go.Scattergeo(
        lat=df_volcano["lat"],
        lon=df_volcano["lon"],
        text=df_volcano["name"],
        mode="markers",
        marker=dict(
            size=4,
            color="orange"
        ),
        name="Volcanoes"
    )
)

# --- старые землетрясения ---
fig.add_trace(
    go.Scattergeo(
        lat=df_old["lat"],
        lon=df_old["lon"],
        text=df_old["place"],
        mode="markers",
        marker=dict(
            size=df_old["mag"] * 2,
            color="red",
            opacity=0.4
        ),
        name=f"Earthquakes ({FETCH_DAYS} days)"
    )
)

# активные события
# BIG STARS (дальний масштаб)
fig.add_trace(
    go.Scattergeo(
        lat=df_events["lat"],
        lon=df_events["lon"],
        text=df_events["place"],
        mode="markers",
        marker=dict(
            size=df_events["mag"] * STAR_SIZE_FAR,
            color="red",
            symbol="star",
            line=dict(width=1)
        ),
        name="Recent Events (far)",
        visible=True
    )
)

#  NORMAL STARS (ближний масштаб)
fig.add_trace(
    go.Scattergeo(
        lat=df_events["lat"],
        lon=df_events["lon"],
        text=df_events["place"],
        mode="markers",
        marker=dict(
            size=df_events["mag"] * STAR_SIZE_NEAR,
            color="red",
            symbol="star",
            line=dict(width=1)
        ),
        name="Recent Events (near)",
        visible=False
    )
)

# ==============================
# ПРОГНОЗНЫЕ ОКНА
# ==============================

forecast_colors = {
    1: "yellow",
    2: "orange",
    3: "purple"
}

for d in FORECAST_DAYS:

    lat_h, lon_h, mag_h = forecast_points[d]

    # текст возле звезды
    labels = [f"M{m:.1f}" for m in mag_h]

    # hover информация
    hover = [
        f"Forecast +{d} day<br>Expected magnitude: {m:.2f}"
        for m in mag_h
    ]

    fig.add_trace(
        go.Scattergeo(
            lat=lat_h,
            lon=lon_h,
            mode="markers+text",  
            text=labels,
            textposition="top center",
            textfont=dict(size=9, color="white"),

            hovertext=hover,
            hoverinfo="text",

            marker=dict(
                size=np.clip(mag_h * 3, 4, 18),
                color=forecast_colors[d],
                symbol="star-diamond",
                opacity=0.85
            ),
            name=f"Forecast window +{d} day"
        )
    )

# ==============================
# 5. ВИД ГЛОБУСА
# ==============================
fig.update_layout(
    title="Global Volcano + Earthquake Field",
    geo=dict(
        projection_type="orthographic",
        showland=True,
        showocean=True,
        showcountries=True
    ),
    mapbox_style="carto-positron",
    mapbox_zoom=1,
    mapbox_center={"lat":0, "lon":0},
)

# ==============================
# КНОПКИ УПРАВЛЕНИЯ СЛОЯМИ
# ==============================

fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="left",
            x=0.02,
            y=1.20,
            showactive=True,
            buttons=[

                
                dict(
                    label="FULL VIEW",
                    method="update",
                    args=[{
                        "visible": [True] * len(fig.data)
                    }]
                ),

                
                dict(
                    label="VOLCANO MODE",
                    method="update",
                    args=[{
                        "visible": (
                                [False,  # 0 Stress Field
                                 True,  # 1 Volcanoes
                                 False,  # 2 Old earthquakes
                                 True,  # 3 Big stars
                                 True]  # 4 Near stars
                                +
                                [True] * (len(fig.data) - 5)  # прогноз остаётся
                        )
                    }]
                ),
            ],
        )
    ]
)

fig.show()