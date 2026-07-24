"""
Core pipeline for the J&J Norderstedt Deutschlandticket commute analysis.

Steps covered:
  1. Synthetic employee generation (population-weighted, no real data)
  2. Public transport connection assessment via the Transitous routing API
     (open MOTIS instance serving official HVV / DELFI GTFS feeds)
  3. Door-to-door commute time calculation
  4. Commute-time grouping
  5. Deutschlandticket adoption scoring
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

# Johnson & Johnson Medical GmbH, Robert-Koch-Straße 1, 22851 Norderstedt
# (geocoded via OpenStreetMap / Nominatim)
WORKPLACE = {"name": "J&J Medical GmbH (Norderstedt)", "lat": 53.68652, "lon": 10.04696}

TRANSITOUS_URL = "https://api.transitous.org/api/v1/plan"

# Target arrival: a typical working Monday, 08:00 local time
ARRIVAL_TIME = "2026-07-27T08:00:00+02:00"

DEUTSCHLANDTICKET_PRICE_EUR = 58.0   # monthly price (2026)
CAR_COST_PER_KM_EUR = 0.30           # fuel + wear, conservative ADAC-style figure
WORKDAYS_PER_MONTH = 21
ROAD_DETOUR_FACTOR = 1.35            # road distance ≈ 1.35 × straight line
CAR_SPEED_KMH = 42.0                 # average urban/suburban door-to-door speed
CAR_OVERHEAD_MIN = 6.0               # parking + walk at both ends

# ----------------------------------------------------------------------------
# 1. Synthetic employees
# ----------------------------------------------------------------------------

# (area, population_in_thousands, lat, lon, sigma_deg)
# Population figures are approximate public statistics used only as sampling
# weights. sigma controls residential spread around the area centroid.
AREAS = [
    # Hamburg city districts
    ("Rahlstedt",           92, 53.602, 10.157, 0.014),
    ("Billstedt",           70, 53.540, 10.100, 0.012),
    ("Winterhude",          56, 53.596, 10.000, 0.008),
    ("Eimsbüttel",          58, 53.575,  9.952, 0.008),
    ("Barmbek-Nord",        41, 53.605, 10.040, 0.007),
    ("Barmbek-Süd",         36, 53.583, 10.044, 0.007),
    ("Wandsbek",            36, 53.581, 10.084, 0.008),
    ("Bramfeld",            52, 53.615, 10.072, 0.010),
    ("Langenhorn",          46, 53.660, 10.010, 0.011),
    ("Niendorf",            42, 53.620,  9.950, 0.010),
    ("Fuhlsbüttel",         13, 53.634, 10.016, 0.006),
    ("Ohlsdorf",            16, 53.625, 10.031, 0.006),
    ("Alsterdorf",          15, 53.611, 10.007, 0.006),
    ("Eppendorf",           25, 53.593,  9.983, 0.006),
    ("Hoheluft",            25, 53.583,  9.975, 0.005),
    ("Altona-Altstadt",     30, 53.548,  9.945, 0.006),
    ("Ottensen",            37, 53.552,  9.918, 0.006),
    ("St. Pauli",           22, 53.550,  9.963, 0.005),
    ("City (Alt-/Neustadt)",15, 53.550,  9.990, 0.005),
    ("Rotherbaum/Harvestehude", 33, 53.570, 9.990, 0.006),
    ("Uhlenhorst",          19, 53.573, 10.017, 0.005),
    ("Hamm",                39, 53.554, 10.057, 0.007),
    ("Horn",                39, 53.552, 10.090, 0.007),
    ("Eilbek",              22, 53.568, 10.045, 0.005),
    ("Poppenbüttel",        24, 53.659, 10.084, 0.009),
    ("Sasel",               24, 53.653, 10.112, 0.009),
    ("Volksdorf",           21, 53.649, 10.163, 0.010),
    ("Farmsen-Berne",       36, 53.606, 10.117, 0.009),
    ("Steilshoop",          19, 53.610, 10.058, 0.006),
    ("Harburg",             26, 53.460,  9.983, 0.010),
    ("Wilhelmsburg",        58, 53.495, 10.010, 0.012),
    ("Bergedorf",           36, 53.489, 10.212, 0.012),
    ("Lokstedt",            30, 53.603,  9.957, 0.007),
    ("Schnelsen",           30, 53.633,  9.920, 0.009),
    ("Eidelstedt",          34, 53.606,  9.905, 0.009),
    ("Stellingen",          26, 53.592,  9.928, 0.008),
    ("Wellingsbüttel",      11, 53.641, 10.080, 0.006),
    ("Hummelsbüttel",       18, 53.648, 10.041, 0.008),
    ("Duvenstedt/Lemsahl",  12, 53.708, 10.105, 0.010),
    # Surrounding towns (Kreise Segeberg, Pinneberg, Stormarn)
    ("Norderstedt",         80, 53.700, 10.010, 0.016),
    ("Quickborn",           22, 53.727,  9.903, 0.010),
    ("Henstedt-Ulzburg",    28, 53.794,  9.978, 0.012),
    ("Kaltenkirchen",       24, 53.837,  9.960, 0.010),
    ("Ahrensburg",          34, 53.674, 10.226, 0.010),
    ("Bad Bramstedt",       15, 53.919,  9.882, 0.009),
    ("Pinneberg",           44, 53.655,  9.789, 0.010),
    ("Elmshorn",            51, 53.754,  9.653, 0.010),
    ("Bargteheide",         16, 53.728, 10.256, 0.009),
    ("Bad Oldesloe",        25, 53.811, 10.374, 0.010),
    ("Wedel",               34, 53.583,  9.700, 0.009),
    ("Glinde/Barsbüttel",   35, 53.545, 10.200, 0.011),
]


def generate_employees(n: int = 350, seed: int = 42) -> pd.DataFrame:
    """Sample synthetic employee home locations, weighted by area population."""
    rng = np.random.default_rng(seed)
    areas = pd.DataFrame(AREAS, columns=["area", "pop_k", "lat", "lon", "sigma"])
    weights = areas["pop_k"] / areas["pop_k"].sum()
    idx = rng.choice(len(areas), size=n, p=weights)

    rows = []
    for i, a_idx in enumerate(idx):
        a = areas.iloc[a_idx]
        lat = rng.normal(a["lat"], a["sigma"])
        lon = rng.normal(a["lon"], a["sigma"] * 1.6)  # lon degrees are shorter at 53.6°N
        rows.append({
            "employee_id": f"E{i+1:04d}",
            "area": a["area"],
            "home_lat": round(float(lat), 5),
            "home_lon": round(float(lon), 5),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 2 & 3. Public transport routing (door-to-door)
# ----------------------------------------------------------------------------

@dataclass
class RouteResult:
    pt_minutes: float | None
    transfers: int | None
    walk_minutes: float | None
    modes: str | None
    first_stop_name: str | None
    first_stop_lat: float | None
    first_stop_lon: float | None
    ok: bool


def _parse_best_itinerary(payload: dict) -> RouteResult:
    its = payload.get("itineraries") or []
    if not its:
        return RouteResult(None, None, None, None, None, None, None, False)
    best = min(its, key=lambda it: it.get("duration", 1e12))
    legs = best.get("legs", [])
    walk_s = sum(l.get("duration", 0) for l in legs if l.get("mode") == "WALK")
    transit_legs = [l for l in legs if l.get("mode") != "WALK"]
    modes = "+".join(dict.fromkeys(l.get("mode", "?") for l in transit_legs)) or "WALK"
    first_stop = transit_legs[0]["from"] if transit_legs else None
    return RouteResult(
        pt_minutes=round(best.get("duration", 0) / 60.0, 1),
        transfers=int(best.get("transfers", max(len(transit_legs) - 1, 0))),
        walk_minutes=round(walk_s / 60.0, 1),
        modes=modes,
        first_stop_name=first_stop.get("name") if first_stop else None,
        first_stop_lat=first_stop.get("lat") if first_stop else None,
        first_stop_lon=first_stop.get("lon") if first_stop else None,
        ok=True,
    )


def query_route(lat: float, lon: float, session: requests.Session,
                retries: int = 3) -> RouteResult:
    """One door-to-door PT itinerary request, arriving Monday 08:00."""
    params = {
        "fromPlace": f"{lat},{lon}",
        "toPlace": f"{WORKPLACE['lat']},{WORKPLACE['lon']}",
        "time": ARRIVAL_TIME,
        "arriveBy": "true",
        "numItineraries": 4,
    }
    for attempt in range(retries):
        try:
            r = session.get(TRANSITOUS_URL, params=params, timeout=30)
            if r.status_code == 200:
                return _parse_best_itinerary(r.json())
        except requests.RequestException:
            pass
        time.sleep(1.5 * (attempt + 1))
    return RouteResult(None, None, None, None, None, None, None, False)


def route_all(employees: pd.DataFrame, max_workers: int = 4) -> pd.DataFrame:
    """Query routes for all employees with polite, bounded concurrency."""
    session = requests.Session()
    session.headers["User-Agent"] = "commute-case-study (educational assessment)"
    results: dict[str, RouteResult] = {}

    def work(row):
        return row.employee_id, query_route(row.home_lat, row.home_lon, session)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(work, row) for row in employees.itertuples()]
        for i, fut in enumerate(as_completed(futures), 1):
            emp_id, res = fut.result()
            results[emp_id] = res
            if i % 50 == 0:
                print(f"  routed {i}/{len(futures)}")

    rec = []
    for row in employees.itertuples():
        r = results[row.employee_id]
        rec.append({
            "employee_id": row.employee_id,
            "pt_minutes": r.pt_minutes,
            "transfers": r.transfers,
            "walk_minutes": r.walk_minutes,
            "modes": r.modes,
            "first_stop_name": r.first_stop_name,
            "first_stop_lat": r.first_stop_lat,
            "first_stop_lon": r.first_stop_lon,
            "route_ok": r.ok,
        })
    return employees.merge(pd.DataFrame(rec), on="employee_id")


# ----------------------------------------------------------------------------
# Car baseline + 4. commute-time grouping + 5. adoption score
# ----------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


TIME_BANDS = ["≤ 30 min", "31–45 min", "46–60 min", "> 60 min"]


def band(minutes: float) -> str:
    if minutes <= 30:
        return TIME_BANDS[0]
    if minutes <= 45:
        return TIME_BANDS[1]
    if minutes <= 60:
        return TIME_BANDS[2]
    return TIME_BANDS[3]


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add car baseline, cost comparison, time band, and adoption score."""
    df = df.copy()
    df["dist_km"] = [
        round(haversine_km(r.home_lat, r.home_lon, WORKPLACE["lat"], WORKPLACE["lon"]), 2)
        for r in df.itertuples()
    ]
    road_km = df["dist_km"] * ROAD_DETOUR_FACTOR
    df["car_minutes"] = (road_km / CAR_SPEED_KMH * 60 + CAR_OVERHEAD_MIN).round(1)
    df["car_cost_month_eur"] = (road_km * 2 * WORKDAYS_PER_MONTH * CAR_COST_PER_KM_EUR).round(0)
    df["savings_month_eur"] = (df["car_cost_month_eur"] - DEUTSCHLANDTICKET_PRICE_EUR).round(0)

    df["time_band"] = df["pt_minutes"].apply(lambda m: band(m) if pd.notna(m) else None)
    df["time_ratio"] = (df["pt_minutes"] / df["car_minutes"]).round(2)

    # ---- Adoption score (0–100), interpretable weighted model -------------
    # Rationale: mode-choice research consistently finds relative travel time,
    # transfer count, and access/egress walking to be the dominant drivers of
    # public transport uptake; cost savings add a monetary incentive.
    def score(r):
        if pd.isna(r.pt_minutes):
            return np.nan
        s = 100.0
        s -= 38 * max(r.time_ratio - 1.0, 0)          # PT slower than car → strong penalty
        s -= 6.5 * max((r.transfers or 0) - 1, 0)     # each transfer beyond the first
        s -= 0.9 * max((r.walk_minutes or 0) - 10, 0) # walking beyond 10 min door-to-door
        s -= 0.35 * max(r.pt_minutes - 45, 0)         # absolute-time fatigue beyond 45 min
        s += min(max(r.savings_month_eur, 0) / 12, 12)  # up to +12 for monthly savings
        return float(np.clip(s, 0, 100))

    df["adoption_score"] = df.apply(score, axis=1).round(1)
    df["adoption_class"] = pd.cut(
        df["adoption_score"], bins=[-0.1, 40, 65, 100.1],
        labels=["Low", "Medium", "High"],
    )
    return df
