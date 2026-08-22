import numpy as np
import pandas as pd
import plotly.graph_objects as go
from utils.constants import (
    STATE_HEX_LAYOUT, STATE_CENTERS, BUCKET_ORDER, BUCKET_INDEX, LEVEL_COLORS,
    HEX_BORDER_COLOR, NO_DATA_COLOR, DEACTIVATED_COLOR, DEACTIVATED_BORDER,
    LABEL_COLOR
)
from utils.data_loader import is_mobile_request, DATA, get_county_geojson
from utils.wage_logic import classify_bucket, bucket_fill_color

COL_UNIT = 0.5
ROW_UNIT = 0.85
HEX_SIZE = 0.42
HEX_CLICK_MARKER_SIZE = 34
MAX_BUCKET_DIFF = len(BUCKET_ORDER) - 1

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

def build_county_choropleth_figure(county_geojson, county_levels, salary, allowed_buckets, state_filter=None, show_legend=True, excluded_states=None, excluded_counties=None, map_id="", county_filter=None, uirevision_val=None, force=False, force_camera=False, override_center=None, override_zoom=None):
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
        geojson=county_geojson, locations=locations, z=z,
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

    if uirevision_val is None:
        uirevision_val = f"county|{map_id}|{state_filter or ''}|{county_filter or ''}"

    if override_center is not None and override_zoom is not None:
        map_center = override_center
        map_zoom = override_zoom
    else:
        lat, lon, zoom = _get_map_center_and_zoom(state_filter, county_filter)
        map_center = dict(lat=lat, lon=lon)
        map_zoom = zoom

    is_mobile = is_mobile_request()
    height = 240 if is_mobile else (400 if map_id in ("compare-map-a", "compare-map-b") else 560)

    map_layout = {
        "style": "carto-positron",
        "center": map_center,
        "zoom": map_zoom,
        "uirevision": uirevision_val,
    }

    fig.update_layout(
        map=map_layout,
        uirevision=uirevision_val,
        margin=(
            {"r": 10, "t": 10, "l": 10, "b": 0} if map_id in ("compare-map-a", "compare-map-b")
            else {"r": 90 if show_legend else 20, "t": 10, "l": 0, "b": 0}
        ),
        height=height,
        meta={"map_kind": "county"}
    )

    return fig

def build_county_diff_figure(county_geojson, county_levels_a, county_levels_b, label_a, label_b, allowed_buckets, state_filter=None, excluded_states=None, excluded_counties=None, county_filter=None, uirevision_val=None, force=False, force_camera=False, override_center=None, override_zoom=None):
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
        geojson=county_geojson, locations=locations, z=z,
        featureidkey="id", colorscale=colorscale, zmin=-6, zmax=4,
        marker_line_color=HEX_BORDER_COLOR, marker_line_width=0.4,
        text=hover_text, hoverinfo="text",
        showscale=True,
        customdata=locations,
        colorbar=colorbar_config
    ))

    if uirevision_val is None:
        uirevision_val = f"diff|{state_filter or ''}|{county_filter or ''}"

    if override_center is not None and override_zoom is not None:
        map_center = override_center
        map_zoom = override_zoom
    else:
        lat, lon, zoom = _get_map_center_and_zoom(state_filter, county_filter)
        map_center = dict(lat=lat, lon=lon)
        map_zoom = zoom

    map_layout = {
        "style": "carto-positron",
        "center": map_center,
        "zoom": map_zoom,
        "uirevision": uirevision_val,
        "domain": dict(x=[0, 0.94], y=[0, 1]),
    }

    fig.update_layout(
        map=map_layout,
        uirevision=uirevision_val,
        margin=margin_config, height=height,
        meta={"map_kind": "county-diff"}
    )

    return fig

def _empty_figure(message):
    fig = go.Figure()
    fig.update_layout(
        annotations=[dict(text=message, showarrow=False, font=dict(size=16))],
        height=560, xaxis={"visible": False}, yaxis={"visible": False}
    )
    return fig