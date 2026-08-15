from functools import lru_cache
import json
import os

import flask
import pandas as pd
from utils.constants import FIPS_TO_STATE, STATE_HEX_LAYOUT

LOCAL_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
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
        from utils.wage_logic import _state_levels_for_cached
        occ_codes_tuple = tuple(sorted(occ_codes)) if occ_codes else ()
        return _state_levels_for_cached(occ_codes_tuple, combine_mode, salary, agg_method)

    def county_levels_for(self, occ_codes, combine_mode, salary):
        from utils.wage_logic import _county_levels_for_cached
        occ_codes_tuple = tuple(sorted(occ_codes)) if occ_codes else ()
        return _county_levels_for_cached(occ_codes_tuple, combine_mode, salary)

DATA: AppData = AppData(resolve_files(DATA_DIR))

_GEOJSON_BY_STATE = {}

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

def get_county_geojson(state_filter=None):
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