from functools import lru_cache

import numpy as np
import pandas as pd
from utils.constants import (
    BUCKET_ORDER, BUCKET_INDEX, LEVEL_COLORS, HEX_BORDER_COLOR,
    DEACTIVATED_COLOR, DEACTIVATED_BORDER, NO_DATA_COLOR
)
from utils.data_loader import normalize_soc_code, DATA

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