import json

from components.layouts import build_layout
from callbacks.map_callbacks import register_callbacks
import dash
import flask
from utils.data_loader import get_county_geojson

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "H-1B Strategic Wage Map"
app.layout = build_layout()

server = app.server

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