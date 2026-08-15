from dash import dash_table, dcc, html
from utils.constants import BUCKET_ORDER, COMBINE_MODES, DEFAULT_TARGET_SALARY
from utils.data_loader import DATA
from utils.constants import LEVEL_COLORS, HEX_BORDER_COLOR, NO_DATA_COLOR, DEACTIVATED_COLOR, DEACTIVATED_BORDER

non_sortable_column_ids = ["rank", "custom_rank", "title", "code"]
table_css = [
    {'selector': f'th[data-dash-column="{col}"] span.column-header--sort', 'rule': 'display: none !important'}
    for col in non_sortable_column_ids
]

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

def make_controls():
    return html.Div(className="controls-scale-wrap", children=[
    html.Div(className="controls-panel", children=[
        html.Div(className="controls-row row-1", children=[
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
        ]),
        html.Div(className="controls-row row-2", children=[
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
            html.Div(className="control-block control-block-inspect", children=[
                html.Div(className="inspect-row", children=[
                    html.Div(className="inspect-field", children=[
                        html.Label("Inspect a state"),
                        dcc.Dropdown(id="inspect-state", options=[], placeholder="Whole country...",
                                     persistence=True, persistence_type="local")
                    ]),
                    html.Div(className="inspect-field", children=[
                        html.Label("Select a county"),
                        dcc.Dropdown(id="inspect-county", options=[], placeholder="Select county...",
                                     persistence=True, persistence_type="local")
                    ])
                ]),
                html.Div(className="inspect-reset-row", children=[
                    html.Button("Reset Entire Map Exclusions", id="btn-reset-all", n_clicks=0,
                                className="inspect-reset-btn"),
                    html.Button("Reset Inspected State Exclusions", id="btn-reset-state", n_clicks=0,
                                className="inspect-reset-btn")
                ])
            ])
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

def _occ_label(occ_code):
    row = DATA.df_occ[DATA.df_occ["code"] == occ_code]
    return row.iloc[0]["title"] if not row.empty else occ_code