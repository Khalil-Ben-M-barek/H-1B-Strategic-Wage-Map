from functools import lru_cache
import json
import os

import dash
from dash import Input, Output, State, ctx, dash_table, dcc, html
import flask
import numpy as np
import pandas as pd
import plotly.graph_objects as go

LOCAL_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATA_DIR = os.environ.get("H1B_DATA_DIR", LOCAL_DATA_DIR)

FILE_NAMES = {
    "occupations": "occupations.json",
    "geography": "geography.json",
    "counties": "counties.json",
    "cost_of_living": "cost_of_living.json",
    "counties_10m": "counties_10m.json",
    "wages": "wages.json"
}

def resolve_files(data_dir):
    base = str(data_dir).replace("\\", "/")
    if not base.endswith("/"):
        base += "/"
    return {k: base + v for k, v in FILE_NAMES.items()}

DEFAULT_TARGET_SALARY = 100000
HEX_CLICK_MARKER_SIZE = 34
BUCKET_ORDER = ["Below L1", "L1", "L2", "L3", "L4"]
BUCKET_INDEX = {b: i for i, b in enumerate(BUCKET_ORDER)}

LEVEL_COLORS = {
    "Below L1": "black",
    "L1": "red",
    "L2": "yellow",
    "L3": "green",
    "L4": "blue"
}

NO_DATA_COLOR = "#bebebe"
DEACTIVATED_COLOR = "#ffffff"
DEACTIVATED_BORDER = "#cccccc"
HEX_BORDER_COLOR = "#c9c9c9"
LABEL_COLOR = "#2b2b2b"

COMBINE_MODES = {
    "average": "Average across selected jobs",
    "strictest": "Strictest (max requirement across jobs)",
    "lenient": "Most lenient (min requirement across jobs)"
}

FIPS_TO_STATE = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
    "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI",
    "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY"
}

STATE_HEX_LAYOUT = {
    "AK": (1, 0), "ME": (23, 0),
    "VT": (20, 1), "NH": (22, 1),
    "WA": (3, 2), "MT": (5, 2), "ND": (7, 2), "MN": (9, 2), "WI": (11, 2),
    "MI": (15, 2), "NY": (19, 2), "MA": (21, 2), "RI": (23, 2),
    "ID": (4, 3), "WY": (6, 3), "SD": (8, 3), "IA": (10, 3), "IL": (12, 3),
    "IN": (14, 3), "OH": (16, 3), "PA": (18, 3), "NJ": (20, 3), "CT": (22, 3),
    "OR": (3, 4), "NV": (5, 4), "CO": (7, 4), "NE": (9, 4), "MO": (11, 4),
    "KY": (13, 4), "WV": (15, 4), "MD": (17, 4), "DE": (19, 4), "DC": (23, 4),
    "CA": (4, 5), "AZ": (6, 5), "UT": (8, 5), "KS": (10, 5), "AR": (12, 5),
    "TN": (14, 5), "VA": (16, 5), "NC": (18, 5),
    "NM": (7, 6), "OK": (9, 6), "LA": (11, 6), "MS": (13, 6), "AL": (15, 6),
    "SC": (17, 6),
    "TX": (8, 7), "GA": (16, 7),
    "HI": (1, 8), "FL": (17, 8)
}

COL_UNIT = 0.5
ROW_UNIT = 0.85
HEX_SIZE = 0.42

STATE_CENTERS = {
    "AL": (32.8066, -86.7911, 6), "AK": (61.3707, -152.4044, 3.5), "AZ": (33.7297, -111.4312, 6),
    "AR": (34.9697, -92.3731, 6), "CA": (36.1162, -119.6815, 5), "CO": (39.0598, -105.3111, 6),
    "CT": (41.5977, -72.7553, 8), "DE": (39.3185, -75.5071, 8), "DC": (38.9072, -77.0369, 10),
    "FL": (27.7662, -81.6867, 6), "GA": (33.0406, -83.6430, 6), "HI": (21.0943, -157.4983, 6),
    "ID": (44.2404, -114.4788, 6), "IL": (40.3494, -88.9861, 6), "IN": (39.8494, -86.2582, 6),
    "IA": (42.0115, -93.2105, 6), "KS": (38.5266, -96.7264, 6), "KY": (37.6681, -84.6700, 6),
    "LA": (31.1695, -91.8678, 6), "ME": (44.6939, -69.3819, 6), "MD": (39.0639, -76.8021, 7),
    "MA": (42.2301, -71.5301, 7), "MI": (43.3266, -84.5360, 6), "MN": (45.6944, -93.9001, 6),
    "MS": (31.1695, -90.0495, 6), "MO": (38.4560, -92.2883, 6), "MT": (46.9219, -110.4543, 5),
    "NE": (41.1253, -98.2680, 6), "NV": (38.3135, -117.0553, 6), "NH": (43.4524, -71.5638, 7),
    "NJ": (40.2989, -74.5210, 7), "NM": (34.8405, -106.2484, 6), "NY": (42.1657, -74.9480, 6),
    "NC": (35.6300, -79.8064, 6), "ND": (47.5289, -99.7840, 6), "OH": (40.3887, -82.7649, 6),
    "OK": (35.5653, -96.9289, 6), "OR": (44.5720, -122.0709, 6), "PA": (40.5907, -77.2097, 6),
    "RI": (41.6808, -71.5117, 8), "SC": (33.8568, -80.9450, 6), "SD": (44.2997, -99.4388, 6),
    "TN": (35.7478, -86.6923, 6), "TX": (31.0544, -97.5634, 5), "UT": (40.1500, -111.8624, 6),
    "VT": (44.0458, -72.7106, 7), "VA": (37.7693, -78.1699, 6), "WA": (47.4009, -121.4904, 6),
    "WV": (38.4912, -80.9544, 6), "WI": (44.2685, -89.6165, 6), "WY": (42.7559, -107.3024, 6)
}

non_sortable_column_ids = ["rank", "custom_rank", "title", "code"]
table_css = [
    {'selector': f'th[data-dash-column="{col}"] span.column-header--sort', 'rule': 'display: none !important'}
    for col in non_sortable_column_ids
]

def normalize_soc_code(code):
    if not code:
        return ""
    code_str = str(code).strip()
    if "." in code_str:
        code_str = code_str.split(".")[0]
    return code_str

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_all_data(files):
    for label, filepath in files.items():
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Missing required data file '{label}' target path: {filepath}. "
                "Please verify the folder contents or set the H1B_DATA_DIR environment variable."
            )
    return {
        "occ": pd.DataFrame(load_json(files["occupations"])).assign(
            code=lambda d: d["code"].astype(str)
        ),
        "geo_raw": load_json(files["geography"]),
        "counties_raw": load_json(files["counties"]),
        "col_raw": load_json(files["cost_of_living"]),
        "wages_raw": load_json(files["wages"]),
        "counties_10m_raw": load_json(files["counties_10m"])
    }

def parse_primary_state(area_name):
    if not area_name or "," not in area_name:
        return None
    suffix = area_name.split(",")[-1].strip()
    token = suffix.split()[0] if suffix.split() else ""
    parts = token.split("-")
    first_part = parts[0].upper()
    if len(first_part) == 2 and first_part.isalpha():
        return first_part
    return None

def build_area_to_state(geo_raw, col_raw, counties_raw):
    area_to_state = {}
    for code in geo_raw:
        state = geo_raw[code].get("state")
        if state:
            area_to_state[str(code)] = state.upper()
    for details in counties_raw.values():
        code = str(details.get("area"))
        area_name = details.get("areaName")
        primary_state = parse_primary_state(area_name)
        if code and primary_state:
            area_to_state[code] = primary_state
    for code in col_raw:
        code_str = str(code)
        if code_str not in area_to_state:
            area_name = col_raw[code].get("areaName")
            primary_state = parse_primary_state(area_name)
            if primary_state:
                area_to_state[code_str] = primary_state
    return area_to_state

def build_area_to_county(counties_raw):
    rows = []
    for key in counties_raw:
        details = counties_raw[key]
        rows.append({
            "county_key": key,
            "county": details.get("county", key),
            "state": (details.get("state") or "").upper(),
            "area_code": str(details.get("area")),
            "area_name": details.get("areaName", "")
        })
    return pd.DataFrame(rows)

def build_wage_thresholds_df(wages_raw):
    rows = []
    for area_code in wages_raw:
        occ_dict = wages_raw[area_code]
        for occ_code in occ_dict:
            levels = occ_dict[occ_code]
            if not levels or len(levels) < 4 or any(v is None for v in levels[:4]):
                continue
            l1, l2, l3, l4 = [float(v) for v in levels[:4]]
            if any(v < 10000 or v >= 5000000 for v in (l1, l2, l3, l4)):
                continue
            rows.append({
                "area_code": str(area_code), "occupation_code": normalize_soc_code(occ_code),
                "L1": l1, "L2": l2, "L3": l3, "L4": l4
            })
    return pd.DataFrame(rows)

COUNTY_SUFFIXES = [
    " city and borough", " census area", " municipality", " municipio",
    " county", " parish", " borough", " city"
]

def normalize_county_name(name):
    if not name:
        return ""
    n = name.strip().lower()
    for suf in COUNTY_SUFFIXES:
        if n.endswith(suf):
            n = n[: -len(suf)]
            break
    return n.strip()

def _decode_arc(raw_arc, scale, translate):
    coords = []
    x = y = 0
    for i, (dx, dy) in enumerate(raw_arc):
        if i == 0:
            x, y = dx, dy
        else:
            x += dx
            y += dy
        if scale is not None:
            coords.append([x * scale[0] + translate[0], y * scale[1] + translate[1]])
        else:
            coords.append([x, y])
    return coords

def _ring_from_arc_indices(arc_indices, decoded_arcs):
    ring = []
    for idx in arc_indices:
        if idx < 0:
            pts = list(reversed(decoded_arcs[~idx]))
        else:
            pts = decoded_arcs[idx]
        if ring and pts:
            pts = pts[1:]
        ring.extend(pts)
    if ring and ring[0] != ring[-1]:
        ring.append(ring[0])
    return ring

def _geometry_arcs_to_coordinates(geom_type, arcs_field, decoded_arcs):
    if geom_type == "Polygon":
        return [_ring_from_arc_indices(ring, decoded_arcs) for ring in arcs_field]
    if geom_type == "MultiPolygon":
        return [
            [_ring_from_arc_indices(ring, decoded_arcs) for ring in polygon]
            for polygon in arcs_field
        ]
    return None

def topology_to_geojson(topology, object_name=None):
    transform = topology.get("transform")
    scale = transform["scale"] if transform else None
    translate = transform["translate"] if transform else None
    decoded_arcs = [_decode_arc(arc, scale, translate) for arc in topology.get("arcs", [])]
    objects = topology.get("objects", {})
    if object_name is None:
        if "counties" in objects:
            object_name = "counties"
        else:
            object_name = max(
                objects, key=lambda k: len(objects[k].get("geometries", []))
            )
    features = []
    for geom in objects[object_name].get("geometries", []):
        gtype = geom.get("type")
        coords = _geometry_arcs_to_coordinates(gtype, geom.get("arcs", []), decoded_arcs)
        if coords is None:
            continue
        fid = str(geom.get("id", "")).zfill(5) if geom.get("id") is not None else ""
        props = dict(geom.get("properties", {}))
        if not props.get("name") and props.get("NAME"):
            props["name"] = props["NAME"]
        props["fips"] = fid
        features.append({
            "type": "Feature",
            "id": fid,
            "properties": props,
            "geometry": {"type": gtype, "coordinates": coords}
        })
    return {"type": "FeatureCollection", "features": features}, object_name

def build_county_shapes_df(counties_geojson):
    rows = []
    for feat in counties_geojson["features"]:
        fips = str(feat["id"]).zfill(5)
        if len(fips) < 5:
            continue
        state_fips = fips[:2]
        name = feat["properties"].get("name", "") or feat["properties"].get("NAME", "")
        rows.append({
            "fips": fips,
            "name": name,
            "state": FIPS_TO_STATE.get(state_fips),
            "norm_name": normalize_county_name(name)
        })
    return pd.DataFrame(rows)

def match_county_shapes_to_wage_areas(df_shapes, df_county_map):
    wage_side = df_county_map.copy()
    wage_side["norm_name"] = wage_side["county"].apply(normalize_county_name)
    merged = df_shapes.merge(
        wage_side[["norm_name", "state", "area_code", "county"]],
        on=["norm_name", "state"], how="left"
    )
    return merged

def compute_area_level(df_wages, occ_codes, combine_mode):
    normalized_occ_codes = [normalize_soc_code(c) for c in occ_codes]
    sub = df_wages[
        df_wages["occupation_code"].isin(occ_codes) |
        df_wages["occupation_code"].isin(normalized_occ_codes)
    ]
    if sub.empty:
        return pd.DataFrame(columns=["area_code", "L1", "L2", "L3", "L4"])
    if combine_mode == "strictest":
        return sub.groupby("area_code")[["L1", "L2", "L3", "L4"]].max().reset_index()
    elif combine_mode == "lenient":
        return sub.groupby("area_code")[["L1", "L2", "L3", "L4"]].min().reset_index()
    else:
        return sub.groupby("area_code")[["L1", "L2", "L3", "L4"]].mean().reset_index()

def aggregate_area_levels_to_group(area_levels, area_to_group, group_col, agg_method="median"):
    df = area_levels.copy()
    df[group_col] = df["area_code"].map(area_to_group)
    df = df.dropna(subset=[group_col])
    if df.empty:
        return pd.DataFrame(columns=[group_col, "L1", "L2", "L3", "L4", "n_areas"])
    agg_fn = "median" if agg_method == "median" else "mean"
    return df.groupby(group_col).agg(
        L1=("L1", agg_fn), L2=("L2", agg_fn), L3=("L3", agg_fn), L4=("L4", agg_fn),
        n_areas=("area_code", "nunique")
    ).reset_index()

def levels_to_buckets(df, salary):
    df = df.copy()
    if df.empty:
        df["bucket"] = None
        return df
    for col in ["L1", "L2", "L3", "L4"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    l1, l2, l3, l4 = df["L1"].values, df["L2"].values, df["L3"].values, df["L4"].values
    has_nan = np.isnan(l1) | np.isnan(l2) | np.isnan(l3) | np.isnan(l4)
    buckets = np.full(len(df), None, dtype=object)
    buckets[salary < l1] = "Below L1"
    buckets[(salary >= l1) & (salary < l2)] = "L1"
    buckets[(salary >= l2) & (salary < l3)] = "L2"
    buckets[(salary >= l3) & (salary < l4)] = "L3"
    buckets[salary >= l4] = "L4"
    buckets[has_nan] = None
    df["bucket"] = buckets
    return df

@lru_cache(maxsize=128)
def _compute_area_level_cached(occ_codes_tuple, combine_mode):
    if DATA is None:
        return pd.DataFrame(columns=["area_code", "L1", "L2", "L3", "L4"])
    return compute_area_level(DATA.df_wages, occ_codes_tuple, combine_mode).copy()

@lru_cache(maxsize=128)
def _state_levels_for_cached(occ_codes_tuple, combine_mode, salary, agg_method="median"):
    area_levels = _compute_area_level_cached(occ_codes_tuple, combine_mode)
    state_levels = aggregate_area_levels_to_group(area_levels, DATA.area_to_state, "state", agg_method)
    return levels_to_buckets(state_levels, salary)

@lru_cache(maxsize=128)
def _county_levels_for_cached(occ_codes_tuple, combine_mode, salary):
    area_levels = _compute_area_level_cached(occ_codes_tuple, combine_mode)
    merged = DATA.df_county_shapes.merge(area_levels, on="area_code", how="left")
    return levels_to_buckets(merged, salary)

def hex_vertices(cx, cy, size):
    angles_deg = [0, 60, 120, 180, 240, 300]
    pts_x = [cx + size * np.cos(np.radians(a)) for a in angles_deg]
    pts_y = [cy + size * np.sin(np.radians(a)) for a in angles_deg]
    pts_x.append(pts_x[0])
    pts_y.append(pts_y[0])
    return pts_x, pts_y

def state_hex_center(state):
    col, row = STATE_HEX_LAYOUT[state]
    return col * COL_UNIT, -row * ROW_UNIT

def format_money(v):
    return "n/a" if v is None or pd.isna(v) else f"${v:,.0f}"

def _clean_salary_string(value):
    if value is None:
        return None
    return str(value).replace(",", "").replace("$", "").strip()

def parse_number(value, default):
    cleaned = _clean_salary_string(value)
    if not cleaned:
        return default
    try:
        return float(cleaned)
    except ValueError:
        return default

def classify_bucket(bucket, state, excluded_states, allowed_buckets):
    is_missing = bucket is None or (not isinstance(bucket, str) and pd.isna(bucket))
    is_excluded = state in excluded_states
    is_hidden = (not is_missing) and bucket not in allowed_buckets
    if is_excluded or is_hidden:
        return "excluded", is_excluded
    if is_missing:
        return "no_data", is_excluded
    return bucket, is_excluded

def bucket_fill_color(category):
    if category == "excluded":
        return DEACTIVATED_COLOR, DEACTIVATED_BORDER
    if category == "no_data":
        return NO_DATA_COLOR, HEX_BORDER_COLOR
    return LEVEL_COLORS[category], HEX_BORDER_COLOR

def _batch_hex_traces(fig, state_entries, is_mobile=False):
    from collections import defaultdict
    groups = defaultdict(lambda: {"xs": [], "ys": []})
    cx_list, cy_list, texts, hover_list, custom_list = [], [], [], [], []
    label_size = 9 if is_mobile else 12
    for entry in state_entries:
        state = entry["state"]
        cx, cy = state_hex_center(state)
        xs, ys = hex_vertices(cx, cy, HEX_SIZE)
        key = (entry["fill_color"], entry["border_color"])
        groups[key]["xs"].extend(xs + [None])
        groups[key]["ys"].extend(ys + [None])
        cx_list.append(cx)
        cy_list.append(cy)
        texts.append(state)
        hover_list.append(entry["hover_text"])
        custom_list.append([state, entry["hover_text"]])
    for (fill_color, border_color), pts in groups.items():
        fig.add_trace(go.Scatter(
            x=pts["xs"], y=pts["ys"], mode="lines", fill="toself",
            fillcolor=fill_color, line=dict(color=border_color, width=1.2),
            hoverinfo="skip", showlegend=False
        ))
    fig.add_trace(go.Scatter(
        x=cx_list, y=cy_list, mode="text", text=texts,
        textfont=dict(size=label_size, color=LABEL_COLOR, family="Arial, sans-serif"),
        hoverinfo="skip", showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=cx_list, y=cy_list, mode="markers",
        marker=dict(size=HEX_CLICK_MARKER_SIZE, color="rgba(0,0,0,0.02)"),
        hoverinfo="text", text=hover_list, customdata=custom_list, showlegend=False
    ))

def add_bucket_legend(fig, x=1.0, y=0.5, yanchor="middle"):
    for bucket in BUCKET_ORDER:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=14, color=LEVEL_COLORS[bucket], symbol="square"),
            name=bucket, showlegend=True
        ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=14, color=NO_DATA_COLOR, symbol="square",
                     line=dict(color=HEX_BORDER_COLOR, width=1)),
        name="No data", showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(size=14, color=DEACTIVATED_COLOR, symbol="square",
                     line=dict(color=DEACTIVATED_BORDER, width=1)),
        name="Excluded", showlegend=True
    ))
    fig.update_layout(legend=dict(title="Triggerable level", x=x, y=y,
                                    xanchor="left", yanchor=yanchor))

LEGEND_ITEMS = ([(b, LEVEL_COLORS[b], HEX_BORDER_COLOR) for b in BUCKET_ORDER]
                + [("No data", NO_DATA_COLOR, HEX_BORDER_COLOR),
                    ("Excluded", DEACTIVATED_COLOR, DEACTIVATED_BORDER)])

def build_legend_html():
    return html.Div(className="map-legend", children=[
        html.Div("Triggerable level", className="legend-title"),
        *[
            html.Div(className="legend-row", children=[
                html.Span(className="legend-swatch",
                           style={"backgroundColor": color, "borderColor": border}),
                html.Span(label)
            ])
            for label, color, border in LEGEND_ITEMS
        ]
    ])

def build_explore_legend_html():
    return html.Div(className="explore-legend-wrapper", children=[
        html.Div("Triggerable level", className="explore-legend-title"),
        html.Div(className="explore-legend-row-container", children=[
            html.Div(className="explore-legend-item", children=[
                html.Span(className="explore-legend-swatch", style={"backgroundColor": color, "borderColor": border}),
                html.Span(label, className="explore-legend-label")
            ])
            for label, color, border in LEGEND_ITEMS
        ])
    ])

def is_mobile_request():
    try:
        if not flask.has_request_context():
            return False
        headers = flask.request.headers
        if "?1" in headers.get("Sec-Ch-Ua-Mobile", ""):
            return True
        user_agent = headers.get("User-Agent", "").lower()
        mobile_keywords = ["android", "webos", "iphone", "ipad", "ipod", "blackberry", "iemobile", "opera mini", "mobile", "mobi"]
        return any(kw in user_agent for kw in mobile_keywords)
    except Exception:
        return False

def _finalize_hex_figure(fig, height=560, margin=None, title=None, show_legend=False, layout_id=""):
    fig.update_xaxes(range=[-0.1, 12.1], visible=False, showgrid=False, zeroline=False, fixedrange=True)
    fig.update_yaxes(range=[-7.4, 0.6], visible=False, showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1, fixedrange=True)
    title_layout = None
    if title:
        if isinstance(title, dict):
            title_layout = title
        else:
            title_layout = dict(text=title)
    layout = dict(
        margin=margin or {"r": 140 if show_legend else 20, "t": 30 if title else 10, "l": 10, "b": 10},
        clickmode="event", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        height=height, showlegend=show_legend,
        dragmode=False,
        title=title_layout
    )
    fig.update_layout(**layout)
    return fig

def build_state_hex_figure(state_levels, salary, excluded_states, allowed_buckets, occ_label="selected job(s)", show_legend=True, map_id=""):
    row_by_state = {r["state"]: r for r in state_levels.to_dict("records")}
    excluded_states = set(excluded_states or [])
    allowed_buckets = set(allowed_buckets) if allowed_buckets else set(BUCKET_ORDER)
    fig = go.Figure()
    hover_by_id = {}
    state_entries = []
    for state in STATE_HEX_LAYOUT:
        row_data = row_by_state.get(state)
        bucket = row_data["bucket"] if row_data is not None else None
        category, is_excluded = classify_bucket(bucket, state, excluded_states, allowed_buckets)
        fill_color, border = bucket_fill_color(category)
        if row_data is None:
            base_hover = f"<b>{state}</b><br>No wage data for {occ_label}"
        else:
            base_hover = (
                f"<b>{state}</b> ({int(row_data['n_areas'])} area(s), median)<br>"
                f"Target salary: {format_money(salary)}<br>"
                f"Triggerable level: <b>{bucket}</b><br>"
                f"L1 {format_money(row_data['L1'])} | L2 {format_money(row_data['L2'])} | "
                f"L3 {format_money(row_data['L3'])} | L4 {format_money(row_data['L4'])}"
            )
        hover_by_id[state] = base_hover
        trace_hover = base_hover + "<br><i>(click to " + ("re-enable" if is_excluded else "exclude") + ")</i>"
        state_entries.append({
            "state": state, "fill_color": fill_color, "border_color": border, "hover_text": trace_hover
        })
    is_mobile = is_mobile_request()
    _batch_hex_traces(fig, state_entries, is_mobile=is_mobile)
    if show_legend:
        add_bucket_legend(fig)
    fig.update_layout(meta={"hover_by_id": hover_by_id, "map_kind": "hex"})
    if is_mobile:
        margin = {"r": 5, "t": 10, "l": 5, "b": 5}
        height = 240
    else:
        margin = {"r": 10, "t": 10, "l": 10, "b": 10} if map_id in ("compare-map-a", "compare-map-b") else None
        height = 400 if map_id in ("compare-map-a", "compare-map-b") else 560
    
    return _finalize_hex_figure(fig, height=height, margin=margin, show_legend=show_legend, layout_id=map_id)

DIVERGING_BLUE = (33, 102, 172)
DIVERGING_WHITE = (230, 230, 230)
DIVERGING_RED = (178, 24, 43)
MAX_BUCKET_DIFF = len(BUCKET_ORDER) - 1

def _diverging_color(t):
    t = min(max(t, 0.0), 1.0)
    lo, hi, f = (DIVERGING_BLUE, DIVERGING_WHITE, t / 0.5) if t < 0.5 else (DIVERGING_WHITE, DIVERGING_RED, (t - 0.5) / 0.5)
    rgb = [int(lo[i] + (hi[i] - lo[i]) * f) for i in range(3)]
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"

def build_diff_hex_figure(state_levels_a, state_levels_b, label_a, label_b, excluded_states, allowed_buckets):
    a_by_state = {r["state"]: r for r in state_levels_a.to_dict("records")}
    b_by_state = {r["state"]: r for r in state_levels_b.to_dict("records")}
    excluded_states = set(excluded_states or [])
    allowed_buckets = set(allowed_buckets) if allowed_buckets else set(BUCKET_ORDER)
    fig = go.Figure()
    hover_by_id = {}
    state_entries = []
    for state in STATE_HEX_LAYOUT:
        ra, rb = a_by_state.get(state), b_by_state.get(state)
        bucket_a = ra["bucket"] if ra is not None else None
        bucket_b = rb["bucket"] if rb is not None else None
        is_excluded = state in excluded_states
        is_hidden = (bucket_a is not None and bucket_a not in allowed_buckets) or \
                    (bucket_b is not None and bucket_b not in allowed_buckets)
        if is_excluded or is_hidden:
            fill_color, border = DEACTIVATED_COLOR, DEACTIVATED_BORDER
        elif bucket_a is None or bucket_b is None:
            fill_color, border = NO_DATA_COLOR, HEX_BORDER_COLOR
        else:
            diff = BUCKET_INDEX[bucket_a] - BUCKET_INDEX[bucket_b]
            t = (MAX_BUCKET_DIFF - diff) / (2 * MAX_BUCKET_DIFF)
            fill_color, border = _diverging_color(t), HEX_BORDER_COLOR
        if bucket_a is None or bucket_b is None:
            base_hover = f"<b>{state}</b><br>No data for one or both jobs"
        else:
            verdict = ("Same level for both jobs" if bucket_a == bucket_b else
                        f"{label_a} can trigger a higher level" if BUCKET_INDEX[bucket_a] > BUCKET_INDEX[bucket_b] else
                        f"{label_b} can trigger a higher level")
            base_hover = f"<b>{state}</b><br>{label_a}: {bucket_a}<br>{label_b}: {bucket_b}<br>{verdict}"
        hover_by_id[state] = base_hover
        trace_hover = base_hover + "<br><i>(click to " + ("re-enable" if is_excluded else "exclude") + ")</i>"
        state_entries.append({
            "state": state, "fill_color": fill_color, "border_color": border, "hover_text": trace_hover
        })
    is_mobile = is_mobile_request()
    _batch_hex_traces(fig, state_entries, is_mobile=is_mobile)
    if is_mobile:
        colorbar_config = dict(
            orientation="h",
            y=-0.1,
            yanchor="top",
            x=0.5,
            xanchor="center",
            thickness=10,
            len=0.8,
            tickfont=dict(size=8),
            tickvals=[-6, -5, -4, -2, 0, 2, 4],
            ticktext=["No data", "Excluded", "B higher", "", "Same", "", "A higher"]
        )
        margin_config = {"r": 10, "t": 10, "l": 5, "b": 50}
        height = 240
    else:
        colorbar_config = dict(
            title="Level difference", x=1.02, xanchor="left",
            tickvals=[-6, -5, -4, -2, 0, 2, 4],
            ticktext=["No data", "Excluded", f"{label_b} higher", "", "Same", "", f"{label_a} higher"]
        )
        margin_config = {"r": 160, "t": 10, "l": 10, "b": 10}
        height = 400
    diff_band_colors = [NO_DATA_COLOR, DEACTIVATED_COLOR] + [
        _diverging_color((MAX_BUCKET_DIFF - diff) / (2 * MAX_BUCKET_DIFF))
        for diff in range(-MAX_BUCKET_DIFF, MAX_BUCKET_DIFF + 1)
    ]
    diff_color_scale = []
    diff_band_count = len(diff_band_colors)
    for i, color in enumerate(diff_band_colors):
        diff_color_scale.append([i / diff_band_count, color])
        diff_color_scale.append([(i + 1) / diff_band_count, color])
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", showlegend=False,
        marker=dict(
            size=0.1, color=[-6, -5, -4, -2, 0, 2, 4],
            colorscale=diff_color_scale,
            cmin=-6, cmax=4, showscale=True,
            colorbar=colorbar_config
        )
    ))
    fig.update_layout(meta={"hover_by_id": hover_by_id, "map_kind": "hex"}, hovermode="closest")
    fig = _finalize_hex_figure(fig, height=height, margin=margin_config, title=None, show_legend=False, layout_id="compare-map-diff")
    return fig

def get_county_centroid(fips, county_geojson):
    if not county_geojson or "features" not in county_geojson:
        return None
    for feat in county_geojson["features"]:
        if str(feat.get("id")) == str(fips):
            g = feat.get("geometry", {})
            gtype = g.get("type")
            coords = g.get("coordinates")
            if not coords:
                continue
            pts = []
            if gtype == "Polygon":
                for ring in coords:
                    pts.extend(ring)
            elif gtype == "MultiPolygon":
                for poly in coords:
                    for ring in poly:
                        pts.extend(ring)
            if pts:
                lons = [p[0] for p in pts]
                lats = [p[1] for p in pts]
                return sum(lats) / len(lats), sum(lons) / len(lons)
    return None

def _get_map_center_and_zoom(state_filter=None, county_filter=None):
    is_mobile = is_mobile_request()
    default_zoom = 1.9 if is_mobile else 3.0
    lat, lon, zoom = 37.0902, -95.7129, default_zoom
    if county_filter:
        centroid = get_county_centroid(county_filter, DATA.county_geojson)
        if centroid:
            c_zoom = 7.5 if is_mobile else 8.5
            return centroid[0], centroid[1], c_zoom
    if state_filter and state_filter in STATE_CENTERS:
        center_data = STATE_CENTERS[state_filter]
        state_zoom = max(1.5, center_data[2] - 1.0) if is_mobile else center_data[2]
        return center_data[0], center_data[1], state_zoom
    return lat, lon, zoom

def build_county_choropleth_figure(county_geojson, county_levels, salary, allowed_buckets, state_filter=None, show_legend=True, excluded_states=None, excluded_counties=None, map_id="", county_filter=None):
    sub = county_levels.copy()
    if state_filter:
        sub = sub[sub["state"] == state_filter]
    allowed_buckets = set(allowed_buckets) if allowed_buckets else set(BUCKET_ORDER)
    excluded_states = set(excluded_states or [])
    excluded_counties = set(excluded_counties or [])
    sub = sub[sub["fips"].notna()]
    buckets = sub["bucket"].values
    fips_arr = sub["fips"].values
    states = sub["state"].values
    counties = sub["county"].values if "county" in sub.columns else sub["name"].values
    names = sub["name"].values if "name" in sub.columns else counties
    l1s = sub["L1"].values
    l2s = sub["L2"].values
    l3s = sub["L3"].values
    l4s = sub["L4"].values
    z, locations, hover_text = [], [], []
    for bucket, fips, state, county, name, l1, l2, l3, l4 in zip(
        buckets, fips_arr, states, counties, names, l1s, l2s, l3s, l4s
    ):
        state_excluded = state in excluded_states
        fips_str = str(fips).zfill(5) if str(fips).replace(".0","").isdigit() else str(fips)
        county_excluded = fips_str in excluded_counties or str(fips) in excluded_counties
        is_valid = isinstance(bucket, str) and bucket in BUCKET_INDEX
        hidden = is_valid and bucket not in allowed_buckets
        is_disabled = state_excluded or county_excluded or hidden
        z_val = -2 if is_disabled else (-1 if not is_valid else BUCKET_INDEX[bucket])
        locations.append(fips_str)
        z.append(z_val)
        display_county = county if (isinstance(county, str) and county.strip()) else (name if isinstance(name, str) else None)
        if display_county is None or (isinstance(display_county, float)):
            display_county = "?"
        label = f"<b>{display_county}, {state or '?'}</b><br>"
        if state_excluded:
            txt = label + "Excluded (state disabled on the map)"
            txt += "<br><i>(click to re-enable state to view)</i>"
        elif county_excluded:
            txt = label + "Excluded (county disabled on the map)"
            txt += "<br><i>(click to re-enable)</i>"
        elif bucket is None or pd.isna(bucket):
            txt = label + "No wage data"
            txt += "<br><i>(click to exclude)</i>"
        else:
            txt = (
                f"{label}Target salary: {format_money(salary)}<br>Triggerable level: <b>{bucket}</b><br>"
                f"L1 {format_money(l1)} | L2 {format_money(l2)} | "
                f"L3 {format_money(l3)} | L4 {format_money(l4)}"
                + ("<br><i>(hidden by bucket filter)</i>" if hidden else "")
            )
            txt += "<br><i>(click to exclude)</i>"
        hover_text.append(txt)
    if not locations:
        fig = go.Figure()
        fig.update_layout(
            annotations=[dict(text="No county shapes matched for this state yet.",
            showarrow=False, font=dict(size=14))],
            height=560
        )
        return fig
    bands = ["Excluded", "No data"] + BUCKET_ORDER
    n = len(bands)
    band_colors = [DEACTIVATED_COLOR, NO_DATA_COLOR] + [LEVEL_COLORS[b] for b in BUCKET_ORDER]
    colorscale = []
    for i, color in enumerate(band_colors):
        colorscale.append([i / n, color])
        colorscale.append([(i + 1) / n, color])
    fig = go.Figure(go.Choroplethmap(
        geojson=get_county_geojson_url(state_filter), locations=locations, z=z,
        featureidkey="id", colorscale=colorscale, zmin=-2, zmax=4,
        marker_line_color=HEX_BORDER_COLOR, marker_line_width=0.4,
        text=hover_text, hoverinfo="text",
        showscale=show_legend,
        customdata=locations,
        colorbar=dict(
            title="Wage level", x=1.02, xanchor="left",
            tickvals=[-2, -1, 0, 1, 2, 3, 4],
            ticktext=bands
        ) if show_legend else None
    ))
    lat, lon, zoom = _get_map_center_and_zoom(state_filter, county_filter)
    uirevision_val = f"county|{map_id}|{state_filter or ''}|{county_filter or ''}"
    is_mobile = is_mobile_request()
    if is_mobile:
        height = 240
    else:
        height = 400 if map_id in ("compare-map-a", "compare-map-b") else 560
    fig.update_layout(
        map=dict(
            style="carto-positron",
            center=dict(lat=lat, lon=lon),
            zoom=zoom
        ),
        uirevision=uirevision_val,
        margin=(
            {"r": 10, "t": 10, "l": 10, "b": 0} if map_id in ("compare-map-a", "compare-map-b")
            else {"r": 90 if show_legend else 20, "t": 10, "l": 0, "b": 0}
        ),
        height=height,
        meta={"map_kind": "county"}
    )
    return fig

def build_county_diff_figure(county_geojson, county_levels_a, county_levels_b, label_a, label_b, allowed_buckets, state_filter=None, excluded_states=None, excluded_counties=None, county_filter=None):
    merged = county_levels_a.merge(county_levels_b, on=["fips", "county", "state"], suffixes=("_a", "_b"))
    if state_filter:
        merged = merged[merged["state"] == state_filter]
    allowed_buckets = set(allowed_buckets) if allowed_buckets else set(BUCKET_ORDER)
    excluded_states = set(excluded_states or [])
    excluded_counties = set(excluded_counties or [])
    merged = merged[merged["fips"].notna()]
    buckets_a = merged["bucket_a"].values
    buckets_b = merged["bucket_b"].values
    fips_arr = merged["fips"].values
    states = merged["state"].values
    counties = merged["county"].values
    z, locations, hover_text = [], [], []
    for bucket_a, bucket_b, fips, state, county in zip(
        buckets_a, buckets_b, fips_arr, states, counties
    ):
        state_excluded = state in excluded_states
        county_excluded = fips in excluded_counties
        is_disabled = state_excluded or county_excluded
        is_valid_a = isinstance(bucket_a, str) and bucket_a in BUCKET_INDEX
        is_valid_b = isinstance(bucket_b, str) and bucket_b in BUCKET_INDEX
        label = f"<b>{county or '?'}, {state or '?'}</b><br>"
        if is_disabled:
            z_val = -5
            if state_excluded:
                txt = f"{label}Excluded (state disabled on the map)"
            else:
                txt = f"{label}Excluded"
        elif not is_valid_a or not is_valid_b:
            z_val = -6
            txt = f"{label}No wage data for one or both jobs"
        else:
            diff = BUCKET_INDEX[bucket_a] - BUCKET_INDEX[bucket_b]
            z_val = diff
            verdict = ("Same level for both jobs" if bucket_a == bucket_b else
                        f"{label_a} can trigger a higher level" if BUCKET_INDEX[bucket_a] > BUCKET_INDEX[bucket_b] else
                        f"{label_b} can trigger a higher level")
            txt = f"{label}{label_a}: {bucket_a}<br>{label_b}: {bucket_b}<br>{verdict}"
        if state_excluded:
            txt += "<br><i>(click to re-enable state to view)</i>"
        elif county_excluded:
            txt += "<br><i>(click to re-enable)</i>"
        else:
            txt += "<br><i>(click to exclude)</i>"
        hover_text.append(txt)
        fips_str = str(fips).zfill(5) if str(fips).isdigit() else str(fips)
        locations.append(fips_str)
        z.append(z_val)
    if not locations:
        fig = go.Figure()
        fig.update_layout(
            annotations=[dict(text="No county shapes matched.", showarrow=False, font=dict(size=14))],
            height=560
        )
        return fig
    diff_colors = []
    for diff in range(-4, 5):
        t = (MAX_BUCKET_DIFF - diff) / (2 * MAX_BUCKET_DIFF)
        diff_colors.append(_diverging_color(t))
    band_colors = [NO_DATA_COLOR, DEACTIVATED_COLOR] + diff_colors
    n = len(band_colors)
    colorscale = []
    for i, color in enumerate(band_colors):
        colorscale.append([i / n, color])
        colorscale.append([(i + 1) / n, color])
    is_mobile = is_mobile_request()
    if is_mobile:
        colorbar_config = dict(
            orientation="h",
            y=-0.1,
            yanchor="top",
            x=0.5,
            xanchor="center",
            thickness=10,
            len=0.8,
            tickfont=dict(size=8),
            tickvals=[-6, -5, -4, -2, 0, 2, 4],
            ticktext=["No data", "Excluded", "B higher", "", "Same", "", "A higher"]
        )
        margin_config = {"r": 10, "t": 10, "l": 10, "b": 50}
        height = 240
    else:
        colorbar_config = dict(
            title="Level difference", x=1.02, xanchor="left",
            tickvals=[-6, -5, -4, -2, 0, 2, 4],
            ticktext=["No data", "Excluded", f"{label_b} higher", "", "Same", "", f"{label_a} higher"]
        )
        margin_config = {"r": 160, "t": 10, "l": 0, "b": 0}
        height = 500
    fig = go.Figure(go.Choroplethmap(
        geojson=get_county_geojson_url(state_filter), locations=locations, z=z,
        featureidkey="id", colorscale=colorscale, zmin=-6, zmax=4,
        marker_line_color=HEX_BORDER_COLOR, marker_line_width=0.4,
        text=hover_text, hoverinfo="text",
        showscale=True,
        customdata=locations,
        colorbar=colorbar_config
    ))
    lat, lon, zoom = _get_map_center_and_zoom(state_filter, county_filter)
    uirevision_val = f"diff|{state_filter or ''}|{county_filter or ''}"
    fig.update_layout(
        map=dict(
            style="carto-positron",
            center=dict(lat=lat, lon=lon),
            zoom=zoom,
            domain=dict(x=[0, 0.94], y=[0, 1])
        ),
        uirevision=uirevision_val,
        margin=margin_config, height=height,
        meta={"map_kind": "county-diff"}
    )
    return fig

def _rank_sort(records, sort_by=None):
    def get_pin(row):
        typed = str(row.get("custom_rank", "")).strip()
        try:
            return float(typed) if typed else None
        except ValueError:
            return None
    pinned = []
    unpinned = []
    for r in records:
        pin = get_pin(r)
        if pin is not None:
            pinned.append((pin, r))
        else:
            unpinned.append(r)
    sort_dir = None
    if sort_by:
        for s in sort_by:
            if s.get("column_id") == "required_salary":
                sort_dir = s.get("direction")
                break
    if sort_dir == "asc":
        unpinned = sorted(unpinned, key=lambda x: (x.get("required_salary", 0), x.get("title", "").lower()))
    elif sort_dir == "desc":
        unpinned = sorted(unpinned, key=lambda x: (-x.get("required_salary", 0), x.get("title", "").lower()))
    else:
        unpinned = sorted(unpinned, key=lambda x: (x.get("required_salary", 0), x.get("title", "").lower()))
    pinned = sorted(pinned, key=lambda x: x[0])
    final_list = list(unpinned)
    for pin, row in pinned:
        idx = int(pin) - 1
        idx = max(0, min(idx, len(final_list)))
        final_list.insert(idx, row)
    for i, row in enumerate(final_list, start=1):
        row["rank"] = i
    return final_list

class AppData:
    def __init__(self, files):
        raw = load_all_data(files)
        self.df_occ = raw["occ"].copy()
        self.area_to_state = build_area_to_state(raw["geo_raw"], raw["col_raw"], raw["counties_raw"])
        self.df_county_map = build_area_to_county(raw["counties_raw"])
        self.df_wages = build_wage_thresholds_df(raw["wages_raw"])
        county_geojson, obj_name = topology_to_geojson(raw["counties_10m_raw"])
        self.county_geojson = county_geojson
        df_shapes = build_county_shapes_df(county_geojson)
        matched = match_county_shapes_to_wage_areas(df_shapes, self.df_county_map)
        self.df_county_shapes = matched
        self.occ_options = [
            {"label": f"{r.title} ({r.code})", "value": r.code}
            for r in self.df_occ.itertuples()
        ]
        self.state_options = sorted(list(STATE_HEX_LAYOUT.keys()))

    def state_levels_for(self, occ_codes, combine_mode, salary, agg_method="median"):
        occ_codes_tuple = tuple(sorted(occ_codes)) if occ_codes else ()
        return _state_levels_for_cached(occ_codes_tuple, combine_mode, salary, agg_method)

    def county_levels_for(self, occ_codes, combine_mode, salary):
        occ_codes_tuple = tuple(sorted(occ_codes)) if occ_codes else ()
        return _county_levels_for_cached(occ_codes_tuple, combine_mode, salary)

DATA: AppData = None

_GEOJSON_BY_STATE = {}

def get_county_geojson(state_filter=None):
    if DATA is None:
        return {"type": "FeatureCollection", "features": []}
    key = (state_filter or "").upper()
    cached = _GEOJSON_BY_STATE.get(key)
    if cached is not None:
        return cached
    if not key:
        _GEOJSON_BY_STATE[key] = DATA.county_geojson
        return DATA.county_geojson
    state_fips_prefixes = {v: k for k, v in FIPS_TO_STATE.items()}
    target_prefix = state_fips_prefixes.get(key)
    if not target_prefix:
        _GEOJSON_BY_STATE[key] = DATA.county_geojson
        return DATA.county_geojson
    filtered = {
        "type": "FeatureCollection",
        "features": [
            feat for feat in DATA.county_geojson["features"]
            if str(feat.get("id", "")).startswith(target_prefix)
        ]
    }
    _GEOJSON_BY_STATE[key] = filtered
    return filtered

def get_county_geojson_url(state_filter=None):
    if state_filter:
        return f"/assets/counties_10m.json?state={state_filter.upper()}"
    return "/assets/counties_10m.json"

def make_controls():
    return html.Div(className="controls-panel", children=[
        html.Div(className="control-block", children=[
            html.Label("Occupation(s)"),
            dcc.Dropdown(id="occ-select", options=[], multi=True,
                          placeholder="Select one or more occupations...",
                          persistence=True, persistence_type="local")
        ]),
        html.Div(className="control-block", children=[
            html.Label("Combine mode (for multiple jobs)"),
            dcc.Dropdown(id="combine-mode",
                         options=[{"label": v, "value": k} for k, v in COMBINE_MODES.items()],
                         value="average", clearable=False,
                         persistence=True, persistence_type="local")
        ]),
        html.Div(className="control-block", children=[
            html.Label("Target salary ($)"),
            dcc.Input(id="target-salary", type="text", value=str(DEFAULT_TARGET_SALARY),
                       debounce=True, className="salary-field",
                       persistence=True, persistence_type="local"),
            html.Div(id="salary-validation-msg", className="validation-error", style={"marginTop": "4px", "fontSize": "11px"})
        ]),
        html.Div(className="control-block", children=[
            html.Label("Exclude states"),
            dcc.Dropdown(id="exclude-states", options=[], multi=True,
                          placeholder="Select states to exclude",
                          persistence=True, persistence_type="local")
        ]),
        html.Div(className="control-block", children=[
            html.Label("Toggle wage-level buckets"),
            dcc.Checklist(id="bucket-filter",
                          options=[{"label": f" {b}", "value": b} for b in BUCKET_ORDER],
                          value=BUCKET_ORDER, inline=True,
                          persistence=True, persistence_type="local")
        ]),
        html.Div(className="control-block", children=[
            html.Label("View mode"),
            dcc.RadioItems(id="view-mode", options=[
                {"label": " Map explorer", "value": "explore"},
                {"label": " Compare two jobs", "value": "compare"},
                {"label": " Rank occupations", "value": "rank"}
            ], value="explore", inline=True,
            persistence=True, persistence_type="local")
        ]),
        html.Div(className="control-block", children=[
            html.Label("Map detail mode"),
            dcc.RadioItems(id="map-level", options=[
                {"label": " State overview (simplified)", "value": "state"},
                {"label": " County detail (exact)", "value": "county"}
            ], value="state", inline=True,
            persistence=True, persistence_type="local")
        ]),
        html.Div(className="control-block", style={"minWidth": "340px"}, children=[
            html.Div(style={"display": "flex", "gap": "10px"}, children=[
                html.Div(style={"flex": "1"}, children=[
                    html.Label("Inspect a state"),
                    dcc.Dropdown(id="inspect-state", options=[], placeholder="Whole country...",
                                 persistence=True, persistence_type="local")
                ]),
                html.Div(style={"flex": "1"}, children=[
                    html.Label("Select a county"),
                    dcc.Dropdown(id="inspect-county", options=[], placeholder="Select county...",
                                 persistence=True, persistence_type="local")
                ])
            ]),
            html.Div(style={"display": "flex", "gap": "6px", "marginTop": "6px"}, children=[
                html.Button("Reset Entire Map Exclusions", id="btn-reset-all", n_clicks=0,
                            style={"fontSize": "11px", "padding": "2px 6px", "cursor": "pointer", "flex": "1"}),
                html.Button("Reset Inspected State Exclusions", id="btn-reset-state", n_clicks=0,
                            style={"fontSize": "11px", "padding": "2px 6px", "cursor": "pointer", "flex": "1"})
            ])
        ])
    ])

def build_layout():
    return html.Div(className="app-shell", children=[
        dcc.Store(id="excluded-counties", data=[], storage_type="local"),
        dcc.Store(id="excluded-occupations", data=[], storage_type="local"),
        dcc.Store(id="table-pins", data={}, storage_type="local"),
        html.Div(className="app-header", children=[
            html.H1("H-1B Strategic Wage Map"),
            html.P("Pick a job and a target salary to see which DOL prevailing "
                    "wage level that salary triggers, state by state or county "
                    "by county. You can click any state hexagon or county on the map to exclude or re-include it.")
        ]),
        make_controls(),
        html.Div(id="view-explore", children=[
            html.Div(className="phone-help", children=[
                html.Div("Phone controls", className="phone-help-title"),
                html.Ul([
                    html.Li("Tap a state or county to enable or disable it."),
                    html.Li("Touch and hold to inspect wage information."),
                    html.Li("Slide while holding to inspect nearby regions."),
                    html.Li("Pinch to zoom."),
                    html.Li("Use two fingers to move the map.")
                ])
            ]),
            html.Div(className="map-panel", children=[
                dcc.Graph(id="state-hex-map", responsive=True, config={"displayModeBar": False, "scrollZoom": True}),
                html.Div(id="mobile-tooltip-explore", className="mobile-tooltip"),
                build_explore_legend_html()
            ])
        ]),
        html.Div(id="view-compare", style={"display": "none"}, children=[
            html.Div(className="phone-help", children=[
                html.Div("Phone controls", className="phone-help-title"),
                html.Ul([
                    html.Li("Tap a state or county to enable or disable it."),
                    html.Li("Touch and hold to inspect wage information."),
                    html.Li("Slide while holding to inspect nearby regions."),
                    html.Li("Pinch to zoom."),
                    html.Li("Use two fingers to move the map.")
                ])
            ]),
            html.Div(className="compare-controls", children=[
                html.Div([html.Label("Job A"), dcc.Dropdown(id="compare-occ-a", options=[], persistence=True, persistence_type="local")]),
                html.Div([html.Label("Job B"), dcc.Dropdown(id="compare-occ-b", options=[], persistence=True, persistence_type="local")])
            ]),
            html.Div(className="compare-maps", children=[
                html.Div(className="compare-map-slot", children=[
                    dcc.Graph(id="compare-map-a", responsive=True, config={"displayModeBar": False, "scrollZoom": True}),
                    html.Div(id="mobile-tooltip-compare-a", className="mobile-tooltip")
                ]),
                html.Div(className="compare-legend-col", children=build_legend_html()),
                html.Div(className="compare-map-slot", children=[
                    dcc.Graph(id="compare-map-b", responsive=True, config={"displayModeBar": False, "scrollZoom": True}),
                    html.Div(id="mobile-tooltip-compare-b", className="mobile-tooltip")
                ])
            ]),
            html.Div(id="compare-map-diff-title", className="compare-diff-title"),
            dcc.Graph(id="compare-map-diff", responsive=True, config={"displayModeBar": False, "scrollZoom": True}),
            html.Div(id="mobile-tooltip-compare-diff", className="mobile-tooltip")
        ]),
        html.Div(id="view-rank", style={"display": "none"}, children=[
            html.Div(className="rank-controls", children=[
                html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "16px"}, children=[
                    html.Div(style={"flex": "1", "minWidth": "200px"}, children=[
                        html.Label("Filter ranking to one state"),
                        dcc.Dropdown(id="rank-state-filter", options=[], placeholder="All states", persistence=True, persistence_type="local")
                    ]),
                    html.Div(style={"flex": "1", "minWidth": "200px"}, children=[
                        html.Label("Minimum Salary ($)"),
                        dcc.Input(id="rank-min-salary", type="text", value="60000", className="salary-field", persistence=True, persistence_type="local"),
                        html.Div(id="rank-min-salary-validation-msg", className="validation-error", style={"marginTop": "4px", "fontSize": "11px"})
                    ]),
                    html.Div(style={"flex": "1", "minWidth": "200px"}, children=[
                        html.Label("Maximum Salary ($)"),
                        dcc.Input(id="rank-max-salary", type="text", value="200000", className="salary-field", persistence=True, persistence_type="local"),
                        html.Div(id="rank-max-salary-validation-msg", className="validation-error", style={"marginTop": "4px", "fontSize": "11px"})
                    ]),
                    html.Div(style={"flex": "1", "minWidth": "200px"}, children=[
                        html.Label("Desired Level"),
                        dcc.Dropdown(id="rank-desired-level", options=[
                            {"label": b, "value": b} for b in ["L1", "L2", "L3", "L4"]
                        ], value="L3", persistence=True, persistence_type="local")
                    ]),
                    html.Div(style={"flex": "1", "minWidth": "200px", "display": "flex", "flexDirection": "column", "justifyContent": "flex-end", "gap": "6px"}, children=[
                        html.Button("Reset Deleted Occupations", id="reset-excluded-occupations-btn", n_clicks=0, style={"fontSize": "11px", "padding": "4px 8px", "cursor": "pointer", "width": "100%"}),
                        html.Button("Reset Pinned Occupations", id="reset-pinned-occupations-btn", n_clicks=0, style={"fontSize": "11px", "padding": "4px 8px", "cursor": "pointer", "width": "100%"})
                    ])
                ])
            ]),
            html.P("\"Rank\" is the current order (best match first). Type a "
                    "number in \"Pin to position\" to move that row there. "
                    "It re-sorts as soon as you press enter or tab, or click away.",
                    className="hint-text", style={"marginBottom": "8px"}),
            html.Div(className="rank-table-container", children=[
                html.Div(id="rank-table-validation-msg", className="validation-error"),
                dash_table.DataTable(
                    id="rank-table",
                    columns=[
                        {"name": "Rank", "id": "rank"},
                        {"name": "Pin to position", "id": "custom_rank", "editable": True},
                        {"name": "Occupation", "id": "title"},
                        {"name": "SOC Code", "id": "code"},
                        {"name": "Salary to Trigger Level ($)", "id": "required_salary", "type": "numeric", "format": {"specifier": ",.0f"}}
                    ],
                    css=table_css,
                    sort_action="custom",
                    sort_by=[],
                    row_deletable=True,
                    page_action="native",
                    page_size=1000,
                    style_table={"height": "600px", "overflowY": "auto"},
                    style_cell={"fontFamily": "Arial, sans-serif", "fontSize": 13, "padding": "6px"},
                    style_header={"fontWeight": "bold"},
                    style_data_conditional=[
                        {"if": {"column_id": "custom_rank"}, "backgroundColor": "#fffbe6", "cursor": "text"},
                        {"if": {"column_id": "rank"}, "color": "#777", "fontWeight": "bold"}
                    ],
                    persistence=True,
                    persistence_type="local",
                    persisted_props=["sort_by"]
                )
            ])
        ])
    ])

APP_CSS = """
.app-shell { font-family: Arial, sans-serif; max-width: 1500px; margin: 0 auto; padding: 16px 16px; }
.app-header h1 { margin-bottom: 4px; }
.app-header p { color: #555; margin-top: 0; }
.controls-panel { display: flex; flex-wrap: wrap; gap: 16px; background: #f7f7f9; box-sizing: border-box;
    border: 1px solid #e2e2e6; border-radius: 10px; padding: 16px; margin-bottom: 16px; }
.control-block { flex: 1 1 240px; min-width: 220px; }
.controls-panel > .control-block:nth-child(-n+5) { flex: 1 1 calc((100% - 64px) / 5); min-width: 0; }
.controls-panel > .control-block:nth-child(6) { flex: 0 0 360px; min-width: 360px; max-width: 360px; }
.controls-panel > .control-block:nth-child(7) { flex: 0 0 500px; min-width: 500px; max-width: 500px; }
.controls-panel > .control-block:nth-child(8) { flex: 1 1 0; min-width: 340px; }
.control-block label { font-weight: 600; font-size: 13px; display: block; margin-bottom: 6px; }
.salary-field { width: 100%; box-sizing: border-box; padding: 6px; border: 1px solid #ccc; border-radius: 4px; }

.map-panel {
    border: 1px solid #e2e2e6;
    border-radius: 10px;
    padding: 8px;
    display: flex;
    flex-direction: column;
    align-items: center;
    clear: both;
}

#state-hex-map {
    width: 100% !important;
    height: 560px !important;
    margin-bottom: 16px !important;
}

#compare-map-diff {
    width: 100% !important;
    height: 500px !important;
    margin-bottom: 32px !important;
}

.hint-text { color: #777; font-size: 12px; text-align: center; margin: 4px 0 0; }
.compare-controls { display: flex; gap: 24px; margin-bottom: 8px; }
.compare-controls > div { flex: 1; }
.compare-maps { display: flex; align-items: center; justify-content: space-between !important; width: 100% !important; box-sizing: border-box !important; }
.compare-legend-col { flex: 0 0 150px !important; display: flex !important; align-items: center !important; justify-content: center !important; height: 400px !important; }
.compare-maps > div:not(.compare-legend-col) {
    width: calc(50% - 75px) !important;
    flex: 0 0 calc(50% - 75px) !important;
    min-width: 0 !important;
    height: 400px !important;
}

.compare-maps .js-plotly-plot,
.compare-maps .plotly {
    width: 100% !important;
    max-width: 100% !important;
    height: 100% !important;
}

.map-legend { display: flex; flex-direction: column; gap: 6px; padding: 12px;
    border: 1px solid #e2e2e6; border-radius: 8px; background: #fafafa; }
.legend-title { font-weight: 600; font-size: 12px; margin-bottom: 2px; }
.legend-row { display: flex; align-items: center; gap: 8px; font-size: 12px; color: #333; }
.legend-swatch { width: 14px; height: 14px; border: 1px solid #999; border-radius: 2px;
    display: inline-block; flex: 0 0 14px; }
.rank-controls { margin-bottom: 16px; }
.validation-error { color: #d9534f; font-weight: bold; margin-bottom: 8px; font-size: 13px; }

.compare-diff-title {
    font-weight: bold;
    font-size: 16px;
    text-align: center;
    margin-top: 24px;
    margin-bottom: 8px;
    color: #2b2b2b;
    width: 100%;
}

.explore-legend-wrapper {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    padding: 12px;
    border: 1px solid #e2e2e6;
    border-radius: 8px;
    background: #fafafa;
    margin-top: 16px;
    width: 100%;
    box-sizing: border-box;
}
.explore-legend-title {
    font-weight: 600;
    font-size: 13px;
    text-align: center;
    width: 100%;
    border-bottom: 1px solid #eaeaea;
    padding-bottom: 4px;
    margin-bottom: 4px;
}
.explore-legend-row-container {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 16px;
    width: 100%;
}
.explore-legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: #333;
}
.explore-legend-swatch {
    width: 14px;
    height: 14px;
    border: 1px solid #999;
    border-radius: 2px;
    display: inline-block;
    flex: 0 0 14px;
}

.js-plotly-plot .mapboxgl-canvas,
.js-plotly-plot .maplibregl-canvas {
    cursor: grab !important;
}
.js-plotly-plot .mapboxgl-canvas:active,
.js-plotly-plot .maplibregl-canvas:active {
    cursor: grabbing !important;
}

.js-plotly-plot:has(.hovertext) .nsewdrag {
    cursor: pointer !important;
}
.js-plotly-plot:has(.hovertext) .mapboxgl-canvas,
.js-plotly-plot:has(.hovertext) .maplibregl-canvas {
    cursor: pointer !important;
}

.js-plotly-plot .plotly .hoverlayer {
    pointer-events: none !important;
}

.phone-help {
    display: none;
}
.mobile-tooltip {
    display: none;
    box-sizing: border-box;
    width: 100%;
    margin: 0;
    padding: 0;
    border: none;
    height: 0;
    overflow: hidden;
}
.mobile-tooltip.is-active {
    display: block;
    height: auto;
    margin: 10px 0 8px 0;
    padding: 8px 10px;
    background: #f7f7f9;
    color: #2b2b2b;
    border: 1px solid #d0d7de;
    border-radius: 8px;
    font-size: 13px;
    line-height: 1.4;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    word-break: break-word;
}
.mobile-tooltip .tt-title {
    font-weight: 700;
    font-size: 13px;
    margin-bottom: 2px;
}
.mobile-tooltip .tt-body {
    color: #333;
    font-size: 12px;
}
.mobile-tooltip .tt-hint {
    margin-top: 4px;
    font-size: 11px;
    color: #777;
    font-style: italic;
}

@media (max-width: 767px) {
    .phone-help {
        display: block !important;
        background: #f0f4f8;
        border: 1px solid #d0d7de;
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 10px;
        font-size: 12px;
        color: #333;
    }
    .phone-help-title {
        font-weight: 700;
        font-size: 13px;
        margin-bottom: 4px;
    }
    .phone-help ul {
        margin: 0;
        padding-left: 18px;
    }
    .phone-help li {
        margin: 2px 0;
    }
    .mobile-tooltip:not(.is-active) {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        border: none !important;
    }
    .compare-map-slot {
        display: flex;
        flex-direction: column;
        width: 100%;
        min-width: 0;
    }
    .compare-maps .compare-map-slot .js-plotly-plot,
    .compare-maps .compare-map-slot .plotly {
        width: 100% !important;
        height: 240px !important;
    }
    .js-plotly-plot .hoverlayer,
    .js-plotly-plot .hovertext {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }
    #state-hex-map,
    #compare-map-a,
    #compare-map-b,
    #compare-map-diff,
    #state-hex-map .js-plotly-plot,
    #compare-map-a .js-plotly-plot,
    #compare-map-b .js-plotly-plot,
    #compare-map-diff .js-plotly-plot,
    #state-hex-map .mapboxgl-canvas,
    #compare-map-a .mapboxgl-canvas,
    #compare-map-b .mapboxgl-canvas,
    #compare-map-diff .mapboxgl-canvas,
    #state-hex-map .maplibregl-canvas,
    #compare-map-a .maplibregl-canvas,
    #compare-map-b .maplibregl-canvas,
    #compare-map-diff .maplibregl-canvas {
        touch-action: none !important;
    }
    .compare-diff-title {
        font-size: 13px;
        margin-top: 16px;
        margin-bottom: 4px;
    }
    .app-shell {
        padding: 8px !important;
    }
    .controls-panel {
        flex-direction: column !important;
        gap: 12px !important;
        padding: 12px !important;
    }
    .control-block {
        flex: 1 1 100% !important;
        width: 100% !important;
        min-width: 0 !important;
    }
    .compare-controls {
        flex-direction: column !important;
        gap: 12px !important;
    }
    .compare-maps {
        display: flex !important;
        flex-direction: column !important;
        align-items: stretch !important;
        gap: 12px !important;
    }
    .compare-maps > div:not(.compare-legend-col) {
        flex: none !important;
        width: 100% !important;
        height: auto !important;
    }
    .compare-map-slot {
        height: auto !important;
    }
    .compare-legend-col {
        flex: none !important;
        width: 100% !important;
        margin: 8px 0 !important;
        order: 2 !important;
        height: auto !important;
    }
    .compare-maps > div:nth-child(1) {
        order: 1 !important;
    }
    .compare-maps > div:nth-child(3) {
        order: 3 !important;
    }
    .map-legend {
        display: flex !important;
        flex-wrap: wrap !important;
        flex-direction: row !important;
        justify-content: center !important;
        gap: 10px !important;
        width: 100% !important;
        max-width: 100% !important;
        padding: 8px !important;
        box-sizing: border-box !important;
    }
    .legend-title {
        flex: 0 0 100% !important;
        font-size: 10px !important;
        margin-bottom: 2px !important;
        text-align: center !important;
        border-bottom: 1px solid #eaeaea !important;
        padding-bottom: 4px !important;
    }
    .legend-row {
        display: flex !important;
        align-items: center !important;
        gap: 4px !important;
        font-size: 9px !important;
    }
    .legend-swatch {
        width: 10px !important;
        height: 10px !important;
        flex: 0 0 10px !important;
    }
    #state-hex-map {
        height: 240px !important;
        width: 100% !important;
        margin-bottom: 8px !important;
    }
    #compare-map-diff {
        flex: none !important;
        width: 100% !important;
        height: 350px !important;
        margin-top: 10px !important;
        margin-bottom: 16px !important;
    }
    .rank-table-container {
        width: 100% !important;
        overflow-x: auto !important;
        -webkit-overflow-scrolling: touch;
    }
    .rank-table-container .dash-spreadsheet-container .dash-spreadsheet {
        min-width: 650px !important;
    }
}
"""

def register_callbacks(app: dash.Dash):

    @app.callback(
        Output("occ-select", "options"), Output("exclude-states", "options"),
        Output("compare-occ-a", "options"), Output("compare-occ-b", "options"),
        Output("rank-state-filter", "options"), Output("inspect-state", "options"),
        Output("inspect-county", "options"),
        Input("occ-select", "id")
    )
    def populate_static_options(_):
        if DATA is None:
            raise dash.exceptions.PreventUpdate
        state_opts = [{"label": s, "value": s} for s in DATA.state_options]
        county_rows = DATA.df_county_shapes.dropna(subset=["name", "state", "fips"]).to_dict("records")
        county_opts = sorted(
            [{"label": f"{r['name']}, {r['state']}", "value": r["fips"]} for r in county_rows],
            key=lambda x: x["label"]
        )
        return (DATA.occ_options, state_opts, DATA.occ_options, DATA.occ_options,
                state_opts, state_opts, county_opts)

    @app.callback(
        Output("view-explore", "style"), Output("view-compare", "style"), Output("view-rank", "style"),
        Input("view-mode", "value")
    )
    def switch_view(mode):
        if DATA is None:
            raise dash.exceptions.PreventUpdate
        hide, show = {"display": "none"}, {"display": "block"}
        return (show if mode == "explore" else hide,
                show if mode == "compare" else hide,
                show if mode == "rank" else hide)

    _CLICKABLE_MAP_IDS = ["state-hex-map", "compare-map-a", "compare-map-b", "compare-map-diff"]

    @app.callback(
        Output("exclude-states", "value"),
        Output("excluded-counties", "data"),
        [Input(map_id, "clickData") for map_id in _CLICKABLE_MAP_IDS],
        State("exclude-states", "value"),
        State("excluded-counties", "data"),
        prevent_initial_call=True
    )
    def handle_map_click(*args):
        if DATA is None:
            raise dash.exceptions.PreventUpdate
        click_by_id = dict(zip(_CLICKABLE_MAP_IDS, args[:-2]))
        excluded_states = list(args[-2] or [])
        excluded_counties = list(args[-1] or [])
        click_data = click_by_id.get(ctx.triggered_id)
        if not click_data or not click_data.get("points"):
            return dash.no_update, dash.no_update
        point = click_data["points"][0]
        custom_val = point.get("customdata")
        while isinstance(custom_val, (list, tuple)) and len(custom_val) > 0:
            custom_val = custom_val[0]
        if custom_val is None:
            custom_val = point.get("location")
        if not custom_val:
            return dash.no_update, dash.no_update
        clicked_str = str(custom_val).strip()
        if len(clicked_str) == 2 and clicked_str.isalpha():
            if clicked_str in excluded_states:
                excluded_states.remove(clicked_str)
            else:
                excluded_states.append(clicked_str)
        elif clicked_str.isdigit():
            clicked_str = clicked_str.zfill(5)
            if len(clicked_str) != 5:
                return dash.no_update, dash.no_update
            if clicked_str in excluded_counties:
                excluded_counties.remove(clicked_str)
            else:
                excluded_counties.append(clicked_str)
        else:
            return dash.no_update, dash.no_update
        return excluded_states, excluded_counties

    @app.callback(
        Output("salary-validation-msg", "children"),
        Output("rank-min-salary-validation-msg", "children"),
        Output("rank-max-salary-validation-msg", "children"),
        Input("target-salary", "value"),
        Input("rank-min-salary", "value"),
        Input("rank-max-salary", "value")
    )
    def validate_all_salaries(target_val, min_val, max_val):
        if DATA is None:
            raise dash.exceptions.PreventUpdate
        def run_validation(value, empty_error_msg, format_error_msg):
            cleaned = _clean_salary_string(value)
            if not cleaned:
                return empty_error_msg
            try:
                val = float(cleaned)
                if val < 0:
                    return "⚠️ Salary must be a positive number."
                return ""
            except ValueError:
                return format_error_msg
        target_msg = run_validation(
            target_val,
            "⚠️ Please enter a salary.",
            f"⚠️ Invalid salary format. Using default (${DEFAULT_TARGET_SALARY:,.0f})."
        )
        min_msg = run_validation(
            min_val,
            "⚠️ Please enter a minimum salary.",
            "⚠️ Invalid salary format. Using default ($0)."
        )
        max_msg = run_validation(
            max_val,
            "⚠️ Please enter a maximum salary.",
            "⚠️ Invalid salary format. Using default (no maximum limit)."
        )
        return target_msg, min_msg, max_msg

    @app.callback(
        Output("state-hex-map", "figure"),
        Input("occ-select", "value"), Input("combine-mode", "value"),
        Input("target-salary", "value"), Input("exclude-states", "value"),
        Input("excluded-counties", "data"),
        Input("bucket-filter", "value"), Input("map-level", "value"),
        Input("inspect-state", "value"),
        Input("inspect-county", "value")
    )
    def update_explore_map(occ_codes, combine_mode, salary, excluded, excluded_counties, buckets, level, inspect_state, inspect_county):
        if DATA is None:
            raise dash.exceptions.PreventUpdate
        occ_codes = occ_codes or []
        if not occ_codes:
            return _empty_figure("Select at least one occupation above.")
        salary_val = parse_number(salary, DEFAULT_TARGET_SALARY)
        combine_mode = combine_mode or "average"
        if level == "county":
            county_levels = DATA.county_levels_for(occ_codes, combine_mode, salary_val)
            geojson = get_county_geojson(inspect_state)
            return build_county_choropleth_figure(
                geojson, county_levels, salary_val, buckets,
                state_filter=inspect_state, excluded_states=excluded, excluded_counties=excluded_counties, map_id="state-hex-map", county_filter=inspect_county,
                show_legend=False
            )
        state_levels = DATA.state_levels_for(occ_codes, combine_mode, salary_val)
        return build_state_hex_figure(state_levels, salary_val, excluded, buckets, show_legend=False, map_id="state-hex-map")

    @app.callback(
        Output("compare-map-a", "figure"), Output("compare-map-b", "figure"),
        Output("compare-map-diff", "figure"),
        Output("compare-map-diff-title", "children"),
        Input("compare-occ-a", "value"), Input("compare-occ-b", "value"),
        Input("target-salary", "value"), Input("exclude-states", "value"),
        Input("excluded-counties", "data"),
        Input("bucket-filter", "value"), Input("map-level", "value"),
        Input("inspect-state", "value"),
        Input("inspect-county", "value"),
        Input("view-mode", "value")
    )
    def update_compare(occ_a, occ_b, salary, excluded, excluded_counties, buckets, level, inspect_state, inspect_county, view_mode):
        if DATA is None:
            raise dash.exceptions.PreventUpdate
        if view_mode != "compare":
            raise dash.exceptions.PreventUpdate
        salary_val = parse_number(salary, DEFAULT_TARGET_SALARY)
        if not occ_a or not occ_b:
            empty = _empty_figure("Pick a job for both A and B.")
            return empty, empty, empty, ""
        label_a, label_b = _occ_label(occ_a), _occ_label(occ_b)
        state_levels_a = DATA.state_levels_for([occ_a], "average", salary_val)
        state_levels_b = DATA.state_levels_for([occ_b], "average", salary_val)
        if level == "county":
            county_levels_a = DATA.county_levels_for([occ_a], "average", salary_val)
            county_levels_b = DATA.county_levels_for([occ_b], "average", salary_val)
            geojson = get_county_geojson(inspect_state)
            fig_a = build_county_choropleth_figure(
                geojson, county_levels_a, salary_val, buckets,
                state_filter=inspect_state, show_legend=False, excluded_states=excluded, excluded_counties=excluded_counties, map_id="compare-map-a", county_filter=inspect_county
            )
            fig_b = build_county_choropleth_figure(
                geojson, county_levels_b, salary_val, buckets,
                state_filter=inspect_state, show_legend=False, excluded_states=excluded, excluded_counties=excluded_counties, map_id="compare-map-b", county_filter=inspect_county
            )
            fig_diff = build_county_diff_figure(
                geojson, county_levels_a, county_levels_b,
                label_a, label_b, buckets, state_filter=inspect_state,
                excluded_states=excluded, excluded_counties=excluded_counties, county_filter=inspect_county
            )
            title_text = f"County Difference: {label_a} vs {label_b}"
        else:
            fig_a = build_state_hex_figure(state_levels_a, salary_val, excluded, buckets, occ_label=label_a, show_legend=False, map_id="compare-map-a")
            fig_b = build_state_hex_figure(state_levels_b, salary_val, excluded, buckets, occ_label=label_b, show_legend=False, map_id="compare-map-b")
            fig_diff = build_diff_hex_figure(state_levels_a, state_levels_b, label_a, label_b, excluded, buckets)
            title_text = f"Difference: {label_a} vs {label_b}"
        return fig_a, fig_b, fig_diff, title_text

    @app.callback(
        Output("rank-table", "data"),
        Output("excluded-occupations", "data"),
        Output("table-pins", "data"),
        Output("rank-table", "sort_by"),
        Output("rank-table-validation-msg", "children"),
        Input("view-mode", "value"),
        Input("rank-state-filter", "value"),
        Input("rank-min-salary", "value"),
        Input("rank-max-salary", "value"),
        Input("rank-desired-level", "value"),
        Input("rank-table", "data_timestamp"),
        Input("rank-table", "sort_by"),
        Input("reset-excluded-occupations-btn", "n_clicks"),
        Input("reset-pinned-occupations-btn", "n_clicks"),
        State("rank-table", "data"),
        State("excluded-occupations", "data"),
        State("table-pins", "data")
    )
    def update_rank_table(view_mode, state_filter, min_salary, max_salary, desired_level, _ts, sort_by, reset_clicks, reset_pin_clicks, current_table_data, excluded_occupations, table_pins):
        if DATA is None:
            raise dash.exceptions.PreventUpdate
        if view_mode != "rank":
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        triggered_id = ctx.triggered_id
        excluded_occupations = excluded_occupations or []
        table_pins = table_pins or {}
        validation_msg = ""
        triggered_prop = ""
        if ctx.triggered:
            triggered_prop = ctx.triggered[0]["prop_id"]
        sort_by = sort_by or []
        if triggered_prop == "rank-table.sort_by" and not sort_by:
            sort_by = [{"column_id": "required_salary", "direction": "asc"}]
        else:
            for s in sort_by:
                if not s.get("direction"):
                    s["direction"] = "asc"
        
        if current_table_data:
            for row in current_table_data:
                val = str(row.get("custom_rank", "")).strip()
                if val:
                    try:
                        float(val)
                    except ValueError:
                        validation_msg = "⚠️ Warning: Pin to position must be a number (e.g. 1, 2)."
        if triggered_id == "reset-excluded-occupations-btn":
            excluded_occupations = []
        elif triggered_id == "reset-pinned-occupations-btn":
            table_pins = {}
        min_salary_val = parse_number(min_salary, 0.0)
        max_salary_val = parse_number(max_salary, float("inf"))
        desired_level = desired_level or "L3"
        df_state_wages = DATA.df_wages.copy()
        df_state_wages["state"] = df_state_wages["area_code"].map(DATA.area_to_state)
        df_state_wages = df_state_wages.dropna(subset=["state"])
        df_group = df_state_wages[df_state_wages["state"] == state_filter] if state_filter else df_state_wages
        df_agg = df_group.groupby("occupation_code").agg(
            L1=("L1", "median"), L2=("L2", "median"), L3=("L3", "median"), L4=("L4", "median")
        ).reset_index()
        df_merged = DATA.df_occ.merge(df_agg, left_on="code", right_on="occupation_code", how="inner")
        if df_merged.empty:
            return [], excluded_occupations, table_pins, sort_by, validation_msg
        l1, l2, l3, l4 = df_merged["L1"].values, df_merged["L2"].values, df_merged["L3"].values, df_merged["L4"].values
        if desired_level == "L1":
            t_d = l1
        elif desired_level == "L2":
            t_d = l2
        elif desired_level == "L3":
            t_d = l3
        else:
            t_d = l4
        valid_mask = (t_d >= min_salary_val) & (t_d <= max_salary_val) & (~pd.isna(t_d))
        df_filtered = df_merged[valid_mask].copy()
        if df_filtered.empty:
            return [], excluded_occupations, table_pins, sort_by, validation_msg
        S = t_d[valid_mask]
        df_filtered["required_salary"] = S
        expected_codes = set(df_filtered["code"].tolist())
        if triggered_id == "rank-table" and current_table_data is not None:
            for row in current_table_data:
                code = row.get("code")
                pin = row.get("custom_rank")
                if code:
                    if pin not in (None, ""):
                        table_pins[code] = pin
                    elif code in table_pins:
                        del table_pins[code]
            current_codes = {r.get("code") for r in current_table_data if r.get("code")}
            newly_deleted = (expected_codes - set(excluded_occupations)) - current_codes
            if newly_deleted:
                excluded_occupations.extend(list(newly_deleted))
                for code in newly_deleted:
                    if code in table_pins:
                        del table_pins[code]
                df_filtered = df_filtered[~df_filtered["code"].isin(excluded_occupations)]
            else:
                resorted = _rank_sort(current_table_data, sort_by)
                return resorted, excluded_occupations, table_pins, sort_by, validation_msg
        df_filtered = df_filtered[~df_filtered["code"].isin(excluded_occupations)]
        df_filtered["custom_rank"] = df_filtered["code"].map(table_pins).fillna("")
        records = df_filtered.to_dict("records")
        resorted = _rank_sort(records, sort_by)
        return resorted, excluded_occupations, table_pins, sort_by, validation_msg

    @app.callback(
        Output("map-level", "value"),
        Output("view-mode", "value"),
        Input("inspect-state", "value"),
        Input("inspect-county", "value"),
        Input("occ-select", "value"),
        State("map-level", "value"),
        State("view-mode", "value"),
        prevent_initial_call=True
    )
    def handle_view_and_level_updates(inspect_state_val, inspect_county_val, occ_select_val, current_level, current_view):
        if DATA is None:
            raise dash.exceptions.PreventUpdate
        triggered_id = ctx.triggered_id
        if not triggered_id:
            return dash.no_update, dash.no_update
        next_level = current_level
        next_view = current_view
        if triggered_id == "occ-select":
            if occ_select_val:
                next_view = "explore"
            return next_level, next_view
        if triggered_id in ("inspect-state", "inspect-county"):
            if not inspect_state_val and not inspect_county_val:
                return dash.no_update, dash.no_update
            if current_view == "rank":
                next_view = "explore"
                next_level = "county"
            elif current_level == "state":
                next_level = "county"
            return next_level, next_view
        return dash.no_update, dash.no_update

    @app.callback(
        Output("inspect-state", "value"),
        Output("inspect-county", "value"),
        Input("inspect-state", "value"),
        Input("inspect-county", "value"),
        prevent_initial_call=True
    )
    def sync_state_and_county_dropdowns(state_val, county_val):
        if DATA is None:
            raise dash.exceptions.PreventUpdate
        triggered_id = ctx.triggered_id
        if not triggered_id:
            return dash.no_update, dash.no_update
        if triggered_id == "inspect-state":
            if not state_val:
                return dash.no_update, None
            if county_val:
                state_fips_prefixes = {v: k for k, v in FIPS_TO_STATE.items()}
                prefix = state_fips_prefixes.get(state_val.upper())
                if prefix and not str(county_val).startswith(prefix):
                    return dash.no_update, None
            return dash.no_update, dash.no_update
        elif triggered_id == "inspect-county":
            if not county_val:
                return dash.no_update, dash.no_update
            state_fips = str(county_val)[:2]
            state_abbr = FIPS_TO_STATE.get(state_fips)
            if state_abbr:
                return state_abbr, dash.no_update
        return dash.no_update, dash.no_update

    @app.callback(
        Output("compare-occ-a", "value"),
        Output("compare-occ-b", "value"),
        Input("occ-select", "value"),
        State("compare-occ-a", "value"),
        State("compare-occ-b", "value"),
        prevent_initial_call=True
    )
    def sync_compare_view_defaults(occ_vals, current_a, current_b):
        if DATA is None:
            raise dash.exceptions.PreventUpdate
        if not occ_vals:
            return current_a, current_b
        next_a = current_a
        next_b = current_b
        if len(occ_vals) >= 1:
            next_a = occ_vals[0]
        if len(occ_vals) >= 2:
            next_b = occ_vals[1]
        return next_a, next_b

    @app.callback(
        Output("exclude-states", "value", allow_duplicate=True),
        Output("excluded-counties", "data", allow_duplicate=True),
        Input("btn-reset-all", "n_clicks"),
        Input("btn-reset-state", "n_clicks"),
        State("inspect-state", "value"),
        State("exclude-states", "value"),
        State("excluded-counties", "data"),
        prevent_initial_call=True
    )
    def reset_exclusions(reset_all_clicks, reset_state_clicks, inspect_state, current_states, current_counties):
        if DATA is None:
            raise dash.exceptions.PreventUpdate
        triggered_id = ctx.triggered_id
        if not triggered_id:
            return dash.no_update, dash.no_update
        states = list(current_states or [])
        counties = list(current_counties or [])
        if triggered_id == "btn-reset-all":
            return [], []
        elif triggered_id == "btn-reset-state":
            if not inspect_state:
                return dash.no_update, dash.no_update
            if inspect_state in states:
                states.remove(inspect_state)
            state_to_fips = {v: k for k, v in FIPS_TO_STATE.items()}
            prefix = state_to_fips.get(inspect_state)
            if prefix:
                counties = [c for c in counties if not str(c).startswith(prefix)]
            return states, counties
        return dash.no_update, dash.no_update

def _occ_label(occ_code):
    row = DATA.df_occ[DATA.df_occ["code"] == occ_code]
    return row.iloc[0]["title"] if not row.empty else occ_code

def _empty_figure(message):
    fig = go.Figure()
    fig.update_layout(
        annotations=[dict(text=message, showarrow=False, font=dict(size=16))],
        height=560, xaxis={"visible": False}, yaxis={"visible": False}
    )
    return fig

MOBILE_INTERACTION_JS = r"""
<script>
(function () {
  if (window.__h1bMobileControllerInstalled) return;
  window.__h1bMobileControllerInstalled = true;

  const LONG_PRESS_MS = 320;
  const MOVE_CANCEL_PX = 28;
  const MAP_IDS = ["state-hex-map", "compare-map-a", "compare-map-b", "compare-map-diff"];
  const TOOLTIP_IDS = {
    "state-hex-map": "mobile-tooltip-explore",
    "compare-map-a": "mobile-tooltip-compare-a",
    "compare-map-b": "mobile-tooltip-compare-b",
    "compare-map-diff": "mobile-tooltip-compare-diff"
  };
  const fingerCount = {};
  document.addEventListener("mouseleave", function (e) {
    if (e.target && e.target.classList && e.target.classList.contains("js-plotly-plot")) {
      const hoverlayer = e.target.querySelector(".hoverlayer");
      if (hoverlayer) {
        hoverlayer.style.display = "none";
      }
    }
  }, true);

  document.addEventListener("mouseenter", function (e) {
    if (e.target && e.target.classList && e.target.classList.contains("js-plotly-plot")) {
      const hoverlayer = e.target.querySelector(".hoverlayer");
      if (hoverlayer) {
        hoverlayer.style.display = "";
      }
    }
  }, true);

  function isMobileUI() {
    return (window.matchMedia && (
      window.matchMedia("(max-width: 767px)").matches ||
      window.matchMedia("(pointer: coarse)").matches
    )) || ("ontouchstart" in window);
  }

  function escapeHtml(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function formatTooltipHtml(raw) {
    if (!raw) return "";
    let lines = String(raw).replace(/<\/?b>/gi, "").split(/<br\s*\/?>(?:\n)?/gi)
      .map(l => l.replace(/<[^>]*>/g, "").trim())
      .filter(Boolean)
      .filter(l => !/^\(click to /i.test(l) && !/^\(hidden by/i.test(l));

    if (!lines.length) return "";
    if (/^\d{3,5}$/.test(lines[0]) && lines.length === 1) {
      lines = ["County " + lines[0], "No wage data for this area"];
    }
    return '<div class="tt-title">' + escapeHtml(lines[0]) + "</div>"
      + (lines.length > 1 ? '<div class="tt-body">' + lines.slice(1).map(escapeHtml).join("<br>") + "</div>" : "");
  }

  function showTooltip(graphId, raw) {
    Object.keys(TOOLTIP_IDS).forEach(function (gid) {
      const el = document.getElementById(TOOLTIP_IDS[gid]);
      if (!el) return;
      if (gid === graphId && raw) {
        const html = formatTooltipHtml(raw);
        el.innerHTML = html;
        el.classList.toggle("is-active", !!html);
      } else {
        el.innerHTML = "";
        el.classList.remove("is-active");
      }
    });
  }

  function keepTooltip(graphId) {
    const el = document.getElementById(TOOLTIP_IDS[graphId]);
    if (el && el.innerHTML) el.classList.add("is-active");
  }

  function plotlyGd(id) {
    const el = document.getElementById(id);
    return el && (el.classList.contains("js-plotly-plot") ? el : el.querySelector(".js-plotly-plot"));
  }

  function getMapboxMap(gd) {
    if (!gd || !gd._fullLayout) return null;
    const mb = gd._fullLayout.mapbox || gd._fullLayout.map;
    return mb ? (mb._subplot && mb._subplot.map) || mb._map || mb.map : null;
  }

  function normalizeFips(fid) {
    const s = String(fid || "").trim();
    return (/^\d+$/.test(s) && s.length < 5) ? s.padStart(5, "0") : s;
  }

  function hoverLookup(gd) {
    if (!gd) return {};
    if (gd.__h1bHoverIdx) return gd.__h1bHoverIdx;

    const idx = {};
    const meta = (gd._fullLayout && gd._fullLayout.meta) || {};
    const hoverById = meta.hover_by_id || {};
    Object.keys(hoverById).forEach(function (k) {
      const val = hoverById[k];
      idx[k] = val;
      idx[normalizeFips(k)] = val;
    });

    (gd.data || []).forEach(function (t) {
      const locs = t.locations || [], texts = t.text || t.hovertext || [], cds = t.customdata || [];
      const maxLen = Math.max(locs.length, cds.length, Array.isArray(texts) ? texts.length : 0);
      for (let j = 0; j < maxLen; j++) {
        let tx = Array.isArray(texts) ? texts[j] : texts;
        let loc = locs[j];
        const cd = cds[j];
        if (Array.isArray(cd)) {
          if (cd[1]) tx = cd[1];
          if (cd[0] != null) loc = cd[0];
        }
        if (loc != null && tx) {
          const s = String(loc);
          idx[s] = String(tx);
          idx[normalizeFips(s)] = String(tx);
        }
      }
    });

    gd.__h1bHoverIdx = idx;
    return idx;
  }

  function lookupText(gd, fid) {
    const idx = hoverLookup(gd);
    return idx[normalizeFips(fid)] || idx[String(fid).trim()] || null;
  }

  function hitTestHex(gd, clientX, clientY) {
    if (!gd || !gd._fullData || !gd._fullLayout) return null;
    const rect = gd.getBoundingClientRect();
    const xa = gd._fullLayout.xaxis, ya = gd._fullLayout.yaxis;
    if (!xa || !ya || !xa.range || !ya.range || !xa._length || !ya._length) return null;
    const px = clientX - rect.left, py = clientY - rect.top;
    const xMin = Number(xa.range[0]), xMax = Number(xa.range[1]);
    const yMin = Number(ya.range[0]), yMax = Number(ya.range[1]);
    const xOffset = Number(xa._offset || 0), yOffset = Number(ya._offset || 0);
    const xLen = Number(xa._length), yLen = Number(ya._length);
    const dataX = xMin + ((px - xOffset) / xLen) * (xMax - xMin);
    const dataY = yMax - ((py - yOffset) / yLen) * (yMax - yMin);
    if (!Number.isFinite(dataX) || !Number.isFinite(dataY)) return null;

    let best = null, bestDist = Infinity;
    gd._fullData.forEach(function (t) {
      if (!t || !t.customdata || !t.x || !t.y || !t.x.length || !t.y.length || !t.mode || t.mode.indexOf("markers") < 0) return;
      for (let i = 0; i < Math.min(t.x.length, t.y.length, t.customdata.length); i++) {
        const cx = Number(t.x[i]), cy = Number(t.y[i]);
        if (!Number.isFinite(cx) || !Number.isFinite(cy)) continue;
        const dist = Math.hypot(cx - dataX, cy - dataY);
        if (dist < bestDist) {
          const cd = t.customdata[i];
          if (Array.isArray(cd) && cd[0] != null && cd[1]) {
            bestDist = dist;
            best = { id: String(cd[0]), text: String(cd[1]) };
          }
        }
      }
    });
    return bestDist <= 0.9 ? best : null;
  }

  function hitTestMapbox(gd, clientX, clientY) {
    const map = getMapboxMap(gd);
    if (!map || !map.queryRenderedFeatures) return null;
    const rect = (gd.querySelector(".mapboxgl-canvas, .maplibregl-canvas") || gd).getBoundingClientRect();
    const x = clientX - rect.left, y = clientY - rect.top;

    const layerIds = (map.getStyle()?.layers || [])
      .map(l => l.id)
      .filter(id => /choropleth|plotly|fill/i.test(id));

    const features = map.queryRenderedFeatures([x, y], layerIds.length ? { layers: layerIds } : undefined) || [];

    for (const f of features) {
      const p = f.properties || {};
      const candidates = [];
      if (f.id != null) candidates.push(f.id);
      if (p.fips != null) candidates.push(p.fips);
      if (p.GEOID != null) candidates.push(p.GEOID);
      if (p.STATEFP != null && p.COUNTYFP != null) {
        candidates.push(String(p.STATEFP).padStart(2, "0") + String(p.COUNTYFP).padStart(3, "0"));
      }

      for (const fid of candidates) {
        if (fid) {
          const text = lookupText(gd, fid);
          if (text) return { id: normalizeFips(fid), text: text };
        }
      }

      const name = p.name || p.NAME || p.NAMELSAD;
      if (name) {
        const fid2 = candidates[0] != null ? normalizeFips(candidates[0]) : "";
        return { id: fid2, text: "<b>" + name + "</b><br>No wage data for this area" };
      }
    }
    return null;
  }

  function hitTest(gd, clientX, clientY) {
    return getMapboxMap(gd) ? hitTestMapbox(gd, clientX, clientY) : hitTestHex(gd, clientX, clientY);
  }

  function configureMapboxGestures(gd, graphId) {
    const map = getMapboxMap(gd);
    if (!map) return;
    const key = graphId || "map";

    if (map.touchZoomRotate) {
      map.touchZoomRotate.enable();
      map.touchZoomRotate.disableRotation?.();
    }

    if (map.dragPan) {
      const two = (fingerCount[key] || 0) >= 2;
      two ? map.dragPan.enable() : map.dragPan.disable();
    }

    if (map.__h1bBound) return;
    map.__h1bBound = true;

    const canvas = map.getCanvas?.() || map.getContainer?.();
    if (!canvas) return;

    function sync(e) {
      fingerCount[key] = e?.touches?.length || 0;
      if (map.dragPan) {
        fingerCount[key] >= 2 ? map.dragPan.enable() : map.dragPan.disable();
      }
    }

    canvas.addEventListener("touchstart", sync, { passive: true, capture: true });
    canvas.addEventListener("touchmove", sync, { passive: true, capture: true });
    canvas.addEventListener("touchend", sync, { passive: true, capture: true });
    canvas.addEventListener("touchcancel", function () {
      fingerCount[key] = 0;
      map.dragPan?.disable();
    }, { passive: true, capture: true });
  }

  function bindTouchEvents(root, graphId) {
    let state = "NORMAL", lpTimer = null, suppressClickUntil = 0;
    let startX = 0, startY = 0, lastFeature = null;
    let fingerDown = false, longPressFired = false, multiTouch = false;

    const clearLP = () => { if (lpTimer) { clearTimeout(lpTimer); lpTimer = null; } };

    function beginInspect(x, y) {
      longPressFired = true;
      state = "INSPECTING";
      const gd = plotlyGd(graphId);
      configureMapboxGestures(gd, graphId);
      const feat = hitTest(gd, x, y);
      if (feat) {
        lastFeature = feat;
        showTooltip(graphId, feat.text);
      }
    }

    root.addEventListener("touchstart", function (e) {
      if (!isMobileUI()) return;
      if (e.touches.length > 1) {
        multiTouch = true;
        clearLP();
        getMapboxMap(plotlyGd(graphId))?.dragPan?.enable();
        return;
      }
      multiTouch = false;
      fingerDown = true;
      longPressFired = false;
      startX = e.touches[0].clientX;
      startY = e.touches[0].clientY;
      clearLP();
      lpTimer = setTimeout(() => {
        lpTimer = null;
        if (fingerDown && !multiTouch) beginInspect(startX, startY);
      }, LONG_PRESS_MS);
    }, { passive: true, capture: true });

    root.addEventListener("touchmove", function (e) {
      if (!isMobileUI()) return;
      if (e.touches.length > 1) { multiTouch = true; clearLP(); return; }
      if (!fingerDown) return;

      const t = e.touches[0];
      const dist = Math.hypot(t.clientX - startX, t.clientY - startY);
      if (state !== "INSPECTING") {
        if (dist > MOVE_CANCEL_PX) clearLP();
        return;
      }
      if (e.cancelable) e.preventDefault();
      const feat = hitTest(plotlyGd(graphId), t.clientX, t.clientY);
      if (feat && (!lastFeature || feat.id !== lastFeature.id)) {
        lastFeature = feat;
        showTooltip(graphId, feat.text);
      }
    }, { passive: false, capture: true });

    root.addEventListener("touchend", function (e) {
      if (!isMobileUI()) return;
      clearLP();
      if (multiTouch) {
        multiTouch = !!e.touches?.length;
        if (!multiTouch) {
          fingerDown = false;
          longPressFired = false;
          getMapboxMap(plotlyGd(graphId))?.dragPan?.disable();
        }
        return;
      }
      if (!fingerDown) return;
      fingerDown = false;
      if (longPressFired || state === "INSPECTING") {
        state = "TOOLTIP_FROZEN";
        keepTooltip(graphId);
        longPressFired = false;
        suppressClickUntil = Date.now() + 400;
        if (e.cancelable) e.preventDefault();
        e.stopPropagation();
        return;
      }
      if (state !== "TOOLTIP_FROZEN") state = "NORMAL";
    }, { passive: false, capture: true });

    root.addEventListener("touchcancel", function () {
      clearLP();
      fingerDown = false;
      longPressFired = false;
      multiTouch = false;
      if (state === "INSPECTING") {
        state = "TOOLTIP_FROZEN";
        keepTooltip(graphId);
      }
    }, { passive: true, capture: true });

    root.addEventListener("click", function (e) {
      if (isMobileUI() && Date.now() < suppressClickUntil) {
        e.stopPropagation();
        e.preventDefault();
      }
    }, true);

    root.addEventListener("contextmenu", function (e) {
      if (isMobileUI()) {
        e.preventDefault();
        e.stopPropagation();
      }
    }, true);
  }

  function attachController(graphId) {
    const root = document.getElementById(graphId);
    if (!root) return;

    if (!root.__h1bTouchBound) {
      root.__h1bTouchBound = true;
      root.style.touchAction = "none";
      bindTouchEvents(root, graphId);
    }

    const gd = plotlyGd(graphId);
    if (gd && !gd.__h1bAfterplotBound) {
      gd.__h1bAfterplotBound = true;
      gd.on("plotly_afterplot", function () {
        gd.__h1bHoverIdx = null;
        configureMapboxGestures(gd, graphId);
        showTooltip(graphId, null);
      });
      configureMapboxGestures(gd, graphId);
    }
  }

  function scan() {
    if (isMobileUI()) MAP_IDS.forEach(attachController);
  }

  function boot() {
    scan();
    new MutationObserver(scan).observe(document.body, { childList: true, subtree: true });
    let n = 0;
    const iv = setInterval(function () {
      scan();
      if (++n >= 20) clearInterval(iv);
    }, 500);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
</script>
"""


def create_app(data_dir=None):
    global DATA
    files = resolve_files(data_dir or DATA_DIR)
    DATA = AppData(files)
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    app.title = "H-1B Strategic Wage Map"
    app.layout = build_layout()
    app.index_string = app.index_string.replace(
        "</head>",
        f"<style>{APP_CSS}</style></head>"
    ).replace(
        "</body>",
        f"{MOBILE_INTERACTION_JS}</body>"
    )
    get_county_geojson(None)
    _geojson_http_cache = {}

    @app.server.route("/assets/counties_10m.json")
    def serve_counties_geojson():
        state_filter = flask.request.args.get("state")
        key = (state_filter or "").upper()
        body = _geojson_http_cache.get(key)
        if body is None:
            body = json.dumps(get_county_geojson(state_filter), separators=(",", ":"))
            _geojson_http_cache[key] = body
        resp = flask.Response(body, mimetype="application/json")
        resp.headers["Cache-Control"] = "public, max-age=3600"
        return resp
    register_callbacks(app)
    return app

try:
    app = create_app()
    server = app.server
except Exception as e:
    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H2("Initialization Error"),
        html.P(f"Failed to load datasets: {e}"),
        html.P("Please verify that your JSON data files are located inside a 'data/' subdirectory or at the fallbacked path.")
    ])
    server = app.server

if __name__ == "__main__":
    app.run(debug=True)