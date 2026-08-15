import dash
from dash import Input, Output, State, ctx
import pandas as pd
from utils.constants import BUCKET_ORDER, FIPS_TO_STATE, DEFAULT_TARGET_SALARY
from utils.data_loader import DATA, get_county_geojson
from utils.wage_logic import parse_number, _rank_sort, _clean_salary_string
from components.figures import (
    build_county_choropleth_figure,
    build_county_diff_figure,
    build_state_hex_figure,
    build_diff_hex_figure,
    _empty_figure
)
from components.layouts import _occ_label

_CLICKABLE_MAP_IDS = ["state-hex-map", "compare-map-a", "compare-map-b", "compare-map-diff"]

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
        triggered_id = ctx.triggered_id
        if triggered_id in ("inspect-state", "inspect-county", "map-level") or not triggered_id:
            uirevision_val = f"force|{inspect_state or ''}|{inspect_county or ''}|{level}"
        else:
            uirevision_val = f"county|state-hex-map|{inspect_state or ''}|{inspect_county or ''}"
        if level == "county":
            county_levels = DATA.county_levels_for(occ_codes, combine_mode, salary_val)
            geojson = get_county_geojson(inspect_state)
            return build_county_choropleth_figure(
                geojson, county_levels, salary_val, buckets,
                state_filter=inspect_state, excluded_states=excluded, excluded_counties=excluded_counties, map_id="state-hex-map", county_filter=inspect_county,
                show_legend=False, uirevision_val=uirevision_val
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
        triggered_id = ctx.triggered_id
        if triggered_id in ("inspect-state", "inspect-county", "map-level", "view-mode") or not triggered_id:
            uirevision_val = f"force|{inspect_state or ''}|{inspect_county or ''}|{level}"
        else:
            uirevision_val = f"county|compare|{inspect_state or ''}|{inspect_county or ''}"
        label_a, label_b = _occ_label(occ_a), _occ_label(occ_b)
        state_levels_a = DATA.state_levels_for([occ_a], "average", salary_val)
        state_levels_b = DATA.state_levels_for([occ_b], "average", salary_val)
        if level == "county":
            county_levels_a = DATA.county_levels_for([occ_a], "average", salary_val)
            county_levels_b = DATA.county_levels_for([occ_b], "average", salary_val)
            geojson = get_county_geojson(inspect_state)
            fig_a = build_county_choropleth_figure(
                geojson, county_levels_a, salary_val, buckets,
                state_filter=inspect_state, show_legend=False, excluded_states=excluded, excluded_counties=excluded_counties, map_id="compare-map-a", county_filter=inspect_county,
                uirevision_val=uirevision_val
            )
            fig_b = build_county_choropleth_figure(
                geojson, county_levels_b, salary_val, buckets,
                state_filter=inspect_state, show_legend=False, excluded_states=excluded, excluded_counties=excluded_counties, map_id="compare-map-b", county_filter=inspect_county,
                uirevision_val=uirevision_val
            )
            fig_diff = build_county_diff_figure(
                geojson, county_levels_a, county_levels_b,
                label_a, label_b, buckets, state_filter=inspect_state,
                excluded_states=excluded, excluded_counties=excluded_counties, county_filter=inspect_county,
                uirevision_val=uirevision_val
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