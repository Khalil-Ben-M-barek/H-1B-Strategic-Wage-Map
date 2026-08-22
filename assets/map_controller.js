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
  let frozenGraphId = null;
  let lastFrozenRaw = {};

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

  function reapplyFrozen() {
    if (!frozenGraphId) return;
    const el = document.getElementById(TOOLTIP_IDS[frozenGraphId]);
    if (!el) return;
    const raw = lastFrozenRaw[frozenGraphId];
    if (raw) {
      el.innerHTML = formatTooltipHtml(raw);
      el.classList.add("is-active");
    }
  }

  function showTooltip(graphId, raw) {
    if (!raw) {
      if (frozenGraphId) {
        reapplyFrozen();
      }
      return;
    }
    Object.keys(TOOLTIP_IDS).forEach(function (gid) {
      const el = document.getElementById(TOOLTIP_IDS[gid]);
      if (!el) return;
      if (gid === graphId) {
        el.innerHTML = formatTooltipHtml(raw);
        el.classList.add("is-active");
        frozenGraphId = graphId;
        lastFrozenRaw[graphId] = raw;
      } else {
        el.innerHTML = "";
        el.classList.remove("is-active");
        delete lastFrozenRaw[gid];
      }
    });
  }

  function keepTooltip(graphId) {
    frozenGraphId = graphId;
    reapplyFrozen();
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

  function installPlotlyClickGuard(gd, root) {
    if (!gd || gd.__h1bClickGuardInstalled) return;
    if (typeof gd.emit !== "function") return;

    const originalEmit = gd.emit;
    gd.__h1bOriginalEmit = originalEmit;
    gd.emit = function () {
      const eventName = arguments[0];
      if (eventName === "plotly_click" && isMobileUI() && root.__h1bBlockPlotlyClick) {
        return false;
      }
      return originalEmit.apply(this, arguments);
    };
    gd.__h1bClickGuardInstalled = true;
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
    let state = "NORMAL", lpTimer = null;
    let startX = 0, startY = 0, lastFeature = null;
    let fingerDown = false, longPressFired = false, multiTouch = false;

    root.__h1bBlockPlotlyClick = false;

    const clearLP = () => { if (lpTimer) { clearTimeout(lpTimer); lpTimer = null; } };

    function beginInspect(x, y) {
      longPressFired = true;
      state = "INSPECTING";
      root.__h1bBlockPlotlyClick = true;
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
        longPressFired = false;
        fingerDown = false;
        if (state === "INSPECTING") state = "NORMAL";
        root.__h1bBlockPlotlyClick = true;
        getMapboxMap(plotlyGd(graphId))?.dragPan?.enable();
        return;
      }
      multiTouch = false;
      fingerDown = true;
      longPressFired = false;
      startX = e.touches[0].clientX;
      startY = e.touches[0].clientY;

      const gd = plotlyGd(graphId);
      if (hitTest(gd, startX, startY)) {
        root.__h1bBlockPlotlyClick = false;
      }

      clearLP();
      lpTimer = setTimeout(() => {
        lpTimer = null;
        if (fingerDown && !multiTouch) beginInspect(startX, startY);
      }, LONG_PRESS_MS);
    }, { passive: true, capture: true });

    root.addEventListener("touchmove", function (e) {
      if (!isMobileUI()) return;
      if (e.touches.length > 1) {
        multiTouch = true;
        clearLP();
        longPressFired = false;
        if (state === "INSPECTING") state = "NORMAL";
        root.__h1bBlockPlotlyClick = true;
        return;
      }
      if (!fingerDown || multiTouch) return;

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
        multiTouch = !!(e.touches && e.touches.length);
        if (!multiTouch) {
          fingerDown = false;
          longPressFired = false;
          state = "NORMAL";
          getMapboxMap(plotlyGd(graphId))?.dragPan?.disable();
        }
        return;
      }
      if (!fingerDown) return;
      fingerDown = false;
      if (longPressFired || state === "INSPECTING") {
        root.__h1bBlockPlotlyClick = true;
        state = "TOOLTIP_FROZEN";
        keepTooltip(graphId);
        longPressFired = false;
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
      state = "NORMAL";
      root.__h1bBlockPlotlyClick = true;
    }, { passive: true, capture: true });

    root.addEventListener("click", function (e) {
      if (isMobileUI() && root.__h1bBlockPlotlyClick) {
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

  document.addEventListener("click", function (e) {
    if (!isMobileUI()) return;
    let node = e.target;
    while (node && node !== document) {
      if (node.__h1bBlockPlotlyClick) {
        e.preventDefault();
        e.stopImmediatePropagation();
        return;
      }
      node = node.parentNode;
    }
  }, true);

  function attachController(graphId) {
    const root = document.getElementById(graphId);
    if (!root) return;

    const gd = plotlyGd(graphId);

    if (!root.__h1bTouchBound) {
      root.__h1bTouchBound = true;
      root.style.touchAction = "none";
      bindTouchEvents(root, graphId);
    }

    if (gd) {
      installPlotlyClickGuard(gd, root);

      if (!gd.__h1bAfterplotBound) {
        gd.__h1bAfterplotBound = true;
        gd.on("plotly_afterplot", function () {
          gd.__h1bHoverIdx = null;
          configureMapboxGestures(gd, graphId);
          if (frozenGraphId === graphId) {
            reapplyFrozen();
          }
          try {
            const map = getMapboxMap(gd);
            const mapLayout = gd._fullLayout && (gd._fullLayout.map || gd._fullLayout.mapbox);
            const uirevision = (gd.layout && gd.layout.uirevision) || "";
            if (map && mapLayout && String(uirevision).startsWith("force")) {
              const c = mapLayout.center, z = mapLayout.zoom;
              if (c && z != null) {
                const lon = (c.lon !== undefined) ? c.lon : c.lng;
                if (lon !== undefined && c.lat !== undefined) {
                  map.jumpTo({ center: [lon, c.lat], zoom: z });
                }
              }
            }
          } catch (e) {}
        });
        configureMapboxGestures(gd, graphId);
      }
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

(function () {
  if (window.__h1bControlsScaleInstalled) return;
  window.__h1bControlsScaleInstalled = true;
  const DESIGN_W = 1460;

  function isPhone() {
    return window.matchMedia && window.matchMedia("(max-width: 767px)").matches;
  }

  function fitControls() {
    const wrap = document.querySelector(".controls-scale-wrap");
    const panel = document.querySelector(".controls-panel");
    if (!wrap || !panel) return;

    if (isPhone()) {
      panel.style.width = "";
      panel.style.transform = "";
      wrap.style.height = "";
      return;
    }

    panel.style.width = DESIGN_W + "px";
    panel.style.transform = "none";
    wrap.style.height = "auto";

    const naturalH = panel.offsetHeight;
    const avail = wrap.clientWidth;
    if (avail <= 0) return;

    const s = Math.min(1, avail / DESIGN_W);
    panel.style.transformOrigin = "top left";
    panel.style.transform = "scale(" + s + ")";
    wrap.style.height = Math.ceil(naturalH * s) + "px";
  }

  function scheduleFit() {
    requestAnimationFrame(function () {
      requestAnimationFrame(fitControls);
    });
  }

  function bootScale() {
    scheduleFit();
    window.addEventListener("resize", scheduleFit);
    if (window.visualViewport) {
      window.visualViewport.addEventListener("resize", scheduleFit);
    }
    new MutationObserver(scheduleFit).observe(document.body, {
      childList: true, subtree: true, attributes: true, attributeFilter: ["class", "style"]
    });
    [100, 300, 800, 1500].forEach(function (ms) {
      setTimeout(scheduleFit, ms);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bootScale);
  } else {
    bootScale();
  }
})();

document.addEventListener('dblclick', e => {
    const gd = e.target.closest('.js-plotly-plot');
    const m = gd && gd._fullLayout && (gd._fullLayout.mapbox || gd._fullLayout.map);
    const map = m && ((m._subplot && m._subplot.map) || m._map || m.map);
    if (map) {
      e.stopImmediatePropagation(); e.preventDefault();
      map.jumpTo({ center: [-95.7129, 37.0902], zoom: (window.innerWidth < 768) ? 1.9 : 3.0 });
    }
  }, true);