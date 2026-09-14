/* Мониторинг природных пожаров — веб-интерфейс (MapLibre GL, без сборки). */
(() => {
  "use strict";

  // ------------------------------------------------------------------ утилиты
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const SEV = {
    1: { label: "слабая", color: "#f2c14e" },
    2: { label: "средняя", color: "#ec7c30" },
    3: { label: "сильная", color: "#b3261e" },
  };
  const HOT = { D: "#ff5a24", N: "#9b5de5" };
  const QUERY = "#38c2b4";
  const TL_BURN = "#d9442e";
  const MONTHS = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"];
  const MONTHS_SHORT = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"];

  const fmt = (v, d = 0) =>
    v == null || Number.isNaN(Number(v)) ? "—" : Number(v).toLocaleString("ru-RU", { minimumFractionDigits: d, maximumFractionDigits: d });
  const fmtHa = (v) => fmt(v, 2);
  const fmtDate = (s) => (s ? `${s.slice(8, 10)}.${s.slice(5, 7)}.${s.slice(0, 4)}` : "—");
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const plural = (n, one, few, many) => {
    const a = Math.abs(n) % 100, b = a % 10;
    if (a > 10 && a < 20) return many;
    if (b > 1 && b < 5) return few;
    if (b === 1) return one;
    return many;
  };
  const pad = (n) => String(n).padStart(2, "0");
  const lastDay = (y, m) => new Date(Date.UTC(y, m, 0)).getUTCDate(); // m: 1..12

  // ------------------------------------------------------------------ состояние
  const state = {
    meta: null,
    aoi: null,
    bsChips: null,
    source: "model",
    mode: "aoi", // all | aoi | bbox | polygon | file
    geometry: null, // GeoJSON запроса (polygon/file/aoi)
    bbox: null,
    fileName: null,
    lastBody: null,
    report: null,
    scarsFC: null,
    selected: null,
    selectedInfo: null,
    quicklook: "none",
    chipSevOnly: false,
    sortKey: "area_ha",
    sortDir: -1,
    tableLimit: 100,
    reqId: 0,
    drawing: null,
    basemap: "dark",
    layers: { hotspots: true, scars: true, severity: true, chips: false, aoi: true },
  };

  // ------------------------------------------------------------------ карта
  // Подложка — векторные тайлы OpenFreeMap (схема OpenMapTiles, данные OSM) в собственном стиле:
  // вода, застройка, дороги и подписи населённых пунктов. Слой административных границ и подписи
  // стран/регионов намеренно не выводятся. «Спутник» — снимки Esri без слоя границ.
  const OFM = "https://tiles.openfreemap.org";
  const ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services";
  const NAME = ["coalesce", ["get", "name:ru"], ["get", "name"]];
  const BM_VECTOR = ["bm-landuse", "bm-water", "bm-waterway", "bm-road-minor", "bm-road-major"];
  const BM_PLACES = ["bm-place-village", "bm-place-town", "bm-place-city"];
  const BM_THEMES = {
    dark: {
      bg: "#30373d",
      paint: {
        "bm-landuse": { "fill-color": "#3a4248" },
        "bm-water": { "fill-color": "#1a3142" },
        "bm-waterway": { "line-color": "#223d52" },
        "bm-road-minor": { "line-color": "#414a52" },
        "bm-road-major": { "line-color": "#545f69" },
      },
      label: ["#dfe4e8", "#30373d"],
      water: ["#8eaabd", "#1a3142"],
    },
    light: {
      bg: "#eceeef",
      paint: {
        "bm-landuse": { "fill-color": "#e2e5e7" },
        "bm-water": { "fill-color": "#c6d1d9" },
        "bm-waterway": { "line-color": "#c6d1d9" },
        "bm-road-minor": { "line-color": "#f9fafa" },
        "bm-road-major": { "line-color": "#ffffff" },
      },
      label: ["#48525a", "#eceeef"],
      water: ["#7a8d9a", "#c6d1d9"],
    },
    sat: { bg: "#0f1215", paint: {}, label: ["#ffffff", "rgba(0,0,0,0.8)"], water: ["#cfe0ea", "rgba(0,0,0,0.6)"] },
  };

  function basemapSources() {
    return {
      "bm-ofm": {
        type: "vector",
        url: `${OFM}/planet`,
        attribution: "© OpenFreeMap, © OpenMapTiles, данные © участники OpenStreetMap",
      },
      "bm-sat": {
        type: "raster",
        tiles: [`${ESRI}/World_Imagery/MapServer/tile/{z}/{y}/{x}`],
        tileSize: 256,
        maxzoom: 18,
        attribution: "Снимки © Esri, Maxar, Earthstar Geographics",
      },
    };
  }

  function basemapLayers() {
    const T = BM_THEMES.dark.paint;
    const cls = (list) => ["in", ["get", "class"], ["literal", list]];
    const w = (...stops) => ["interpolate", ["linear"], ["zoom"], ...stops];
    return [
      { id: "bm-sat", type: "raster", source: "bm-sat", layout: { visibility: "none" } },
      {
        id: "bm-landuse", type: "fill", source: "bm-ofm", "source-layer": "landuse", minzoom: 8,
        filter: cls(["residential", "suburb", "neighbourhood", "industrial", "commercial", "retail"]),
        paint: { ...T["bm-landuse"], "fill-opacity": w(8, 0.4, 12, 1) },
      },
      { id: "bm-water", type: "fill", source: "bm-ofm", "source-layer": "water", filter: ["!=", ["get", "brunnel"], "tunnel"], paint: T["bm-water"] },
      {
        id: "bm-waterway", type: "line", source: "bm-ofm", "source-layer": "waterway", minzoom: 6,
        filter: ["all", cls(["river", "canal"]), ["!=", ["get", "brunnel"], "tunnel"]],
        paint: { ...T["bm-waterway"], "line-width": w(6, 0.5, 10, 1.3, 14, 3.5) },
      },
      {
        id: "bm-road-minor", type: "line", source: "bm-ofm", "source-layer": "transportation", minzoom: 9,
        filter: cls(["secondary", "tertiary"]), layout: { "line-cap": "round", "line-join": "round" },
        paint: { ...T["bm-road-minor"], "line-width": w(9, 0.4, 14, 2.4) },
      },
      {
        id: "bm-road-major", type: "line", source: "bm-ofm", "source-layer": "transportation", minzoom: 5,
        filter: cls(["motorway", "trunk", "primary"]), layout: { "line-cap": "round", "line-join": "round" },
        paint: { ...T["bm-road-major"], "line-width": w(5, 0.4, 9, 1.1, 14, 3.6) },
      },
    ];
  }

  function basemapLabelLayers() {
    const [c, h] = BM_THEMES.dark.label;
    const [wc, wh] = BM_THEMES.dark.water;
    const place = (id, klass, minzoom, size) => ({
      id, type: "symbol", source: "bm-ofm", "source-layer": "place", minzoom,
      filter: ["==", ["get", "class"], klass],
      layout: {
        "text-field": NAME,
        "text-font": ["Noto Sans Regular"],
        "text-size": ["interpolate", ["linear"], ["zoom"], ...size],
        "symbol-sort-key": ["coalesce", ["get", "rank"], 99],
        "text-padding": 4,
        "text-max-width": 8,
      },
      paint: { "text-color": c, "text-halo-color": h, "text-halo-width": 1.3, "text-halo-blur": 0.5 },
    });
    return [
      {
        id: "bm-water-name", type: "symbol", source: "bm-ofm", "source-layer": "water_name",
        filter: ["==", ["geometry-type"], "Point"],
        layout: { "text-field": NAME, "text-font": ["Noto Sans Italic"], "text-size": 11.5, "text-letter-spacing": 0.08, "text-max-width": 6 },
        paint: { "text-color": wc, "text-halo-color": wh, "text-halo-width": 1 },
      },
      place("bm-place-village", "village", 10.5, [10.5, 10.5, 14, 12.5]),
      place("bm-place-town", "town", 7, [7, 10.5, 12, 13.5]),
      place("bm-place-city", "city", 4, [4, 11, 8, 13.5, 12, 16]),
    ];
  }
  const EMPTY = { type: "FeatureCollection", features: [] };

  function buildStyle() {
    const sources = basemapSources();
    const layers = [{ id: "bg", type: "background", paint: { "background-color": BM_THEMES.dark.bg } }, ...basemapLayers()];
    for (const id of ["anchor", "aoi", "mask", "chips-bs", "chips-af", "scars", "scar-pts", "hotspots", "query", "draw"])
      sources[id] = { type: "geojson", data: EMPTY };
    const sevColor = ["match", ["get", "dominant_severity"], 1, SEV[1].color, 2, SEV[2].color, 3, SEV[3].color, "#999"];
    layers.push(
      // пустые «якорные» слои: задают место вставки растров и подписей подложки
      { id: "anchor-rasters", type: "fill", source: "anchor", paint: { "fill-opacity": 0 } },
      { id: "mask-fill", type: "fill", source: "mask", paint: { "fill-color": "#1b2229", "fill-opacity": 0.16 } },
      {
        id: "chips-af-line", type: "line", source: "chips-af", layout: { visibility: "none" },
        paint: { "line-color": "#6b5b95", "line-width": 1, "line-dasharray": [3, 2], "line-opacity": 0.8 },
      },
      {
        id: "chips-bs-line", type: "line", source: "chips-bs", layout: { visibility: "none" },
        paint: { "line-color": "#374151", "line-width": ["interpolate", ["linear"], ["zoom"], 6, 0.8, 11, 1.6], "line-dasharray": [2, 2] },
      },
      {
        id: "aoi-line", type: "line", source: "aoi",
        paint: { "line-color": "#1b2229", "line-width": 1.6, "line-dasharray": [4, 2.5], "line-opacity": 0.75 },
      },
      {
        id: "scars-fill", type: "fill", source: "scars", minzoom: 7,
        paint: {
          "fill-color": sevColor,
          "fill-opacity": ["interpolate", ["linear"], ["zoom"], 7, 0.7, 10.4, 0.6, 10.6, 0.06],
        },
      },
      {
        id: "scars-line", type: "line", source: "scars", minzoom: 7,
        paint: {
          "line-color": "#4a140d",
          "line-width": ["interpolate", ["linear"], ["zoom"], 7, 0.4, 10, 0.9, 14, 1.8],
          "line-opacity": ["interpolate", ["linear"], ["zoom"], 7, 0.35, 10, 0.85],
        },
      },
      ...basemapLabelLayers(),
      { id: "anchor-labels", type: "fill", source: "anchor", paint: { "fill-opacity": 0 } },
      {
        id: "scar-pts", type: "circle", source: "scar-pts", maxzoom: 9.5,
        paint: {
          "circle-color": sevColor,
          "circle-radius": ["interpolate", ["linear"], ["sqrt", ["get", "area_ha"]], 0, 2, 10, 3, 40, 5.5, 120, 9],
          "circle-stroke-color": "#3b0d08",
          "circle-stroke-width": 0.7,
          "circle-opacity": ["interpolate", ["linear"], ["zoom"], 7.5, 0.9, 9.4, 0],
          "circle-stroke-opacity": ["interpolate", ["linear"], ["zoom"], 7.5, 0.9, 9.4, 0],
        },
      },
      { id: "query-fill", type: "fill", source: "query", paint: { "fill-color": QUERY, "fill-opacity": 0.04 } },
      { id: "query-line", type: "line", source: "query", paint: { "line-color": QUERY, "line-width": 2.2 } },
      {
        id: "sel-line", type: "line", source: "scars", filter: ["==", ["get", "scar_id"], ""],
        paint: { "line-color": QUERY, "line-width": ["interpolate", ["linear"], ["zoom"], 7, 2, 13, 3.5] },
      },
      {
        // мягкое свечение под термоточками
        id: "hotspots-glow", type: "circle", source: "hotspots",
        paint: {
          "circle-color": ["match", ["get", "daynight"], "N", HOT.N, HOT.D],
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 4, 5, 7, 8, 10, 13, 13, 22],
          "circle-blur": 1,
          "circle-opacity": 0.42,
        },
      },
      {
        id: "hotspots", type: "circle", source: "hotspots",
        paint: {
          "circle-color": ["match", ["get", "daynight"], "N", HOT.N, HOT.D],
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 4, 1.8, 7, 2.8, 10, 4.5, 13, 8],
          "circle-stroke-color": "#ffffff",
          "circle-stroke-width": ["interpolate", ["linear"], ["zoom"], 4, 0.4, 10, 1.2],
          "circle-opacity": 0.92,
        },
      },
      { id: "draw-fill", type: "fill", source: "draw", filter: ["==", ["geometry-type"], "Polygon"], paint: { "fill-color": QUERY, "fill-opacity": 0.1 } },
      { id: "draw-line", type: "line", source: "draw", paint: { "line-color": QUERY, "line-width": 2, "line-dasharray": [2, 1.5] } },
      {
        id: "draw-pts", type: "circle", source: "draw", filter: ["==", ["geometry-type"], "Point"],
        paint: { "circle-radius": 4, "circle-color": "#fff", "circle-stroke-color": QUERY, "circle-stroke-width": 2 },
      }
    );
    return { version: 8, glyphs: `${OFM}/fonts/{fontstack}/{range}.pbf`, sources, layers };
  }

  let map;

  function initMap() {
    map = new maplibregl.Map({
      container: "map",
      style: buildStyle(),
      bounds: state.aoi ? bboxOf(state.aoi) : [38, 44, 49, 53],
      fitBoundsOptions: { padding: { top: 30, bottom: 130, left: 30, right: 30 } },
      attributionControl: false,
      maxZoom: 15.5,
      dragRotate: false,
      pitchWithRotate: false,
    });
    map.touchZoomRotate.disableRotation();
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    map.addControl(new maplibregl.ScaleControl({ maxWidth: 110, unit: "metric" }), "bottom-right");
    map.addControl(new maplibregl.AttributionControl({ compact: true }), "bottom-right");
    map.on("load", onMapLoad);
    // подписи масштабной линейки по-русски
    const ruScale = () => {
      const sc = document.querySelector(".maplibregl-ctrl-scale");
      if (!sc) return;
      const t = sc.textContent;
      const r = t.replace(/\s?km$/, "\u00a0км").replace(/(\d)\s?m$/, "$1\u00a0м");
      if (r !== t) sc.textContent = r;
    };
    map.on("move", ruScale);
    map.on("load", ruScale);
    map.on("idle", ruScale);
    // тайлы подложки грузятся из интернета; без него карта остаётся рабочей — предупреждаем один раз
    let tilesWarned = false;
    map.on("error", (e) => {
      const sid = e && e.sourceId;
      if (tilesWarned || !sid || !String(sid).startsWith("bm-")) return;
      tilesWarned = true;
      toast("Подложка карты недоступна: нет доступа к тайлам OpenFreeMap / Esri. Данные и выгрузки работают, карта показана без подложки.", "info", 12000);
    });
  }

  function onMapLoad() {
    const at = document.querySelector(".maplibregl-ctrl-attrib");
    if (at) at.classList.remove("maplibregl-compact-show");
    if (state.aoi) map.getSource("aoi").setData(state.aoi.features.find((f) => f.id === "aoi") || state.aoi.features[0]);
    if (state.bsChips) map.getSource("chips-bs").setData(state.bsChips);
    fetchJSON("/api/v1/chips?kind=af").then((fc) => map.getSource("chips-af").setData(fc)).catch(() => {});
    setBasemap(state.basemap);
    bindMapEvents();
    // по умолчанию — территория мониторинга (AOI); «Весь каталог» — одним нажатием
    if (state.aoi) setMode("aoi");
    else runQuery();
  }

  function setBasemap(k) {
    state.basemap = k;
    const th = BM_THEMES[k];
    const sat = k === "sat";
    map.setLayoutProperty("bm-sat", "visibility", sat ? "visible" : "none");
    map.setPaintProperty("bg", "background-color", th.bg);
    for (const id of BM_VECTOR) {
      map.setLayoutProperty(id, "visibility", sat ? "none" : "visible");
      for (const [prop, v] of Object.entries(th.paint[id] || {})) map.setPaintProperty(id, prop, v);
    }
    for (const id of BM_PLACES) {
      map.setPaintProperty(id, "text-color", th.label[0]);
      map.setPaintProperty(id, "text-halo-color", th.label[1]);
    }
    map.setPaintProperty("bm-water-name", "text-color", th.water[0]);
    map.setPaintProperty("bm-water-name", "text-halo-color", th.water[1]);
    const dark = k === "dark";
    map.setPaintProperty("mask-fill", "fill-color", dark ? "#000000" : "#1b2229");
    map.setPaintProperty("mask-fill", "fill-opacity", dark ? 0.3 : k === "sat" ? 0.35 : 0.16);
    map.setPaintProperty("aoi-line", "line-color", k === "sat" ? "#ffffff" : dark ? "#dfe4e8" : "#1b2229");
    map.setPaintProperty("aoi-line", "line-opacity", dark ? 0.55 : 0.75);
    map.setPaintProperty("scars-line", "line-color", dark ? "#f6a36b" : "#4a140d");
    $$("#basemapSeg button").forEach((b) => b.classList.toggle("on", b.dataset.basemap === k));
  }

  function setLayerVisible(key, on) {
    state.layers[key] = on;
    const ids = {
      hotspots: ["hotspots-glow", "hotspots"],
      scars: ["scars-fill", "scars-line", "scar-pts", "sel-line"],
      chips: ["chips-af-line", "chips-bs-line"],
      aoi: ["aoi-line"],
      severity: [],
    }[key];
    for (const id of ids) if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", on ? "visible" : "none");
    if (key === "severity") refreshSeverityOverlays();
  }

  // ------------------------------------------------------------------ сеть
  async function fetchJSON(url, opts) {
    const r = await fetch(url, opts);
    let data = null;
    try {
      data = await r.json();
    } catch (_) {
      /* пусто */
    }
    if (!r.ok) throw new Error((data && (data.message || data.detail)) || `Ошибка сервера (HTTP ${r.status})`);
    return data;
  }

  function queryBody() {
    const body = { date_from: $("#dateFrom").value, date_to: $("#dateTo").value, source: state.source };
    if (state.mode === "bbox" && state.bbox) body.bbox = state.bbox;
    else if (["polygon", "file", "aoi"].includes(state.mode) && state.geometry) body.geometry = state.geometry;
    return body;
  }

  function setBusy(on) {
    $("#progress").hidden = !on;
    $("#runBtn").classList.toggle("busy", on);
    $("#reportBody").classList.toggle("loading", on);
  }

  function markDirty(on = true) {
    $("#runBtn").classList.toggle("dirty", on);
  }

  let toastTimer;
  function toast(msg, kind = "error", ms) {
    const t = $("#toast");
    t.className = `toast ${kind === "info" ? "info" : ""}`;
    t.dataset.kind = kind === "info" ? "info" : "error";
    t.innerHTML = `${esc(msg)}<button class="x" aria-label="Закрыть">×</button>`;
    t.hidden = false;
    $(".x", t).onclick = () => (t.hidden = true);
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (t.hidden = true), ms || (kind === "info" ? 3500 : 9000));
  }

  async function runQuery() {
    const df = $("#dateFrom").value, dt = $("#dateTo").value;
    if (!df || !dt) return toast("Укажите обе даты периода.");
    if (df > dt) return toast("Дата начала периода позже даты окончания.");
    if (state.mode === "bbox" && !state.bbox) return toast("Нарисуйте прямоугольник на карте или введите координаты bbox.");
    if (["polygon", "file"].includes(state.mode) && !state.geometry) return toast("Сначала задайте полигон на карте или загрузите GeoJSON.");
    const body = queryBody();
    const id = ++state.reqId;
    setBusy(true);
    try {
      const data = await fetchJSON("/api/v1/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...body, include: ["hotspots", "burn_scars", "report"], compare: true }),
      });
      if (id !== state.reqId) return;
      state.lastBody = body;
      applyResult(data);
      markDirty(false);
      // снимаем сообщение об ошибке прошлого запроса, но не информационные (например, о подложке)
      if ($("#toast").dataset.kind !== "info") $("#toast").hidden = true;
    } catch (e) {
      if (id === state.reqId) toast(e.message);
    } finally {
      if (id === state.reqId) setBusy(false);
    }
  }

  // ------------------------------------------------------------------ результат
  function applyResult(data) {
    const rep = data.report;
    state.report = rep;
    state.scarsFC = data.burn_scars;
    map.getSource("hotspots").setData(data.hotspots);
    map.getSource("scars").setData(data.burn_scars);
    map.getSource("scar-pts").setData({
      type: "FeatureCollection",
      features: rep.scars.map((s) => ({
        type: "Feature",
        geometry: { type: "Point", coordinates: [s.center_lon, s.center_lat] },
        properties: { scar_id: s.scar_id, area_ha: s.area_ha, dominant_severity: s.dominant_severity },
      })),
    });
    drawQueryGeometry();
    ensureQueryVisible();
    if (state.selected && !rep.scars.some((s) => s.scar_id === state.selected)) closeScar();
    state.tableLimit = 100;
    renderReport();
    renderTimeline();
    refreshSeverityOverlays();
  }

  function territoryLabel() {
    switch (state.mode) {
      case "aoi":
        return "Граница AOI";
      case "bbox": {
        const b = state.bbox;
        // неразрывные пробелы, чтобы «в. д.» / «с. ш.» не разрывались при переносе строки
        return b ? `Прямоугольник ${fmt(b[0], 2)}–${fmt(b[2], 2)}° в. д. × ${fmt(b[1], 2)}–${fmt(b[3], 2)}° с. ш.` : "Прямоугольник";
      }
      case "polygon": {
        const n = state.geometry ? state.geometry.coordinates[0].length - 1 : 0;
        return `Полигон, ${n} ${plural(n, "вершина", "вершины", "вершин")}`;
      }
      case "file":
        return `Файл ${state.fileName || "GeoJSON"}`;
      default:
        return "Весь каталог";
    }
  }

  // ------------------------------------------------------------------ геометрия запроса на карте
  function polygonsOf(g) {
    if (!g) return [];
    if (g.type === "FeatureCollection") return g.features.flatMap((f) => polygonsOf(f.geometry));
    if (g.type === "Feature") return polygonsOf(g.geometry);
    if (g.type === "Polygon") return [g.coordinates];
    if (g.type === "MultiPolygon") return g.coordinates;
    return [];
  }
  const signedArea = (ring) => {
    let s = 0;
    for (let i = 0; i < ring.length - 1; i++) s += ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1];
    return s / 2;
  };
  function bboxOf(g) {
    let x0 = 180, y0 = 90, x1 = -180, y1 = -90;
    for (const p of polygonsOf(g)) for (const [x, y] of p[0]) {
      x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y);
    }
    return [x0, y0, x1, y1];
  }
  const bboxPolygon = (b) => ({ type: "Polygon", coordinates: [[[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]], [b[0], b[1]]]] });

  function currentQueryGeometry() {
    const body = state.lastBody || {};
    if (body.bbox) return bboxPolygon(body.bbox);
    return body.geometry || null;
  }

  function drawQueryGeometry() {
    const g = currentQueryGeometry();
    if (!g) {
      map.getSource("query").setData(EMPTY);
      map.getSource("mask").setData(EMPTY);
      return;
    }
    const polys = polygonsOf(g);
    map.getSource("query").setData({
      type: "FeatureCollection",
      features: polys.map((p) => ({ type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: p } })),
    });
    const world = [[-180, -85], [180, -85], [180, 85], [-180, 85], [-180, -85]];
    const holes = polys.map((p) => (signedArea(p[0]) > 0 ? p[0].slice().reverse() : p[0]));
    map.getSource("mask").setData({ type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: [world, ...holes] } });
  }

  /** Если область запроса не попала в текущий вид (bbox введён руками, запрос вне данных) — показать её. */
  function ensureQueryVisible() {
    const g = currentQueryGeometry();
    if (!g) return;
    const [x0, y0, x1, y1] = bboxOf(g);
    if (!(x1 > x0 && y1 > y0)) return;
    const b = map.getBounds();
    if (x1 >= b.getWest() && x0 <= b.getEast() && y1 >= b.getSouth() && y0 <= b.getNorth()) return;
    map.fitBounds([[x0, y0], [x1, y1]], { padding: { top: 40, bottom: 140, left: 40, right: 40 }, maxZoom: 12, duration: 700 });
  }

  // ------------------------------------------------------------------ растры степени (LOD) и превью снимков
  const sevOverlays = new Map(); // chip_id -> source/layer id

  function refreshSeverityOverlays() {
    if (!map || !state.report) return;
    const want = new Set();
    const src = state.lastBody ? state.lastBody.source : state.source;
    if (state.layers.severity && map.getZoom() >= 10.4 && state.bsChips) {
      const b = map.getBounds();
      const chipsInResult = new Set(state.report.scars.map((s) => s.chip_id));
      for (const f of state.bsChips.features) {
        const id = f.properties.chip_id;
        if (!chipsInResult.has(id)) continue;
        if (state.quicklook !== "none" && state.selectedInfo && state.selectedInfo.chip.chip_id === id) continue;
        const c = f.properties.corners;
        const xs = c.map((p) => p[0]), ys = c.map((p) => p[1]);
        if (Math.max(...xs) < b.getWest() || Math.min(...xs) > b.getEast() || Math.max(...ys) < b.getSouth() || Math.min(...ys) > b.getNorth()) continue;
        want.add(`${src}|${id}`);
      }
    }
    for (const [key, sid] of sevOverlays) {
      if (!want.has(key)) {
        if (map.getLayer(sid)) map.removeLayer(sid);
        if (map.getSource(sid)) map.removeSource(sid);
        sevOverlays.delete(key);
      }
    }
    let n = 0;
    for (const key of want) {
      if (sevOverlays.has(key) || n > 30) continue;
      n++;
      const [s, id] = key.split("|");
      const f = state.bsChips.features.find((x) => x.properties.chip_id === id);
      const sid = `sev-${s}-${id}`;
      map.addSource(sid, { type: "image", url: `/catalog/${s}/overlays/${id}.png`, coordinates: f.properties.corners });
      map.addLayer(
        { id: sid, type: "raster", source: sid, paint: { "raster-opacity": 0.85 } },
        "mask-fill"
      );
      sevOverlays.set(key, sid);
    }
  }

  function setQuicklook(kind) {
    state.quicklook = kind;
    if (map.getLayer("ql")) map.removeLayer("ql");
    if (map.getSource("ql")) map.removeSource("ql");
    const info = state.selectedInfo;
    if (kind !== "none" && info && info.chip.quicklook) {
      map.addSource("ql", { type: "image", url: `/catalog/${info.chip.quicklook[kind]}`, coordinates: info.chip.corners });
      map.addLayer({ id: "ql", type: "raster", source: "ql", paint: {} }, "anchor-rasters");
    }
    $$("#qlSeg button").forEach((b) => b.classList.toggle("on", b.dataset.ql === kind));
    refreshSeverityOverlays();
  }

  // ------------------------------------------------------------------ выбор контура
  async function selectScar(id, fly = false) {
    state.selected = id;
    map.setFilter("sel-line", ["==", ["get", "scar_id"], id]);
    $$("table.scars tbody tr").forEach((tr) => tr.classList.toggle("sel", tr.dataset.id === id));
    const row = state.report && state.report.scars.find((s) => s.scar_id === id);
    if (fly && row) map.fitBounds(row.bbox, { padding: { top: 90, bottom: 150, left: 340, right: 80 }, maxZoom: 13.5, duration: 900 });
    try {
      const src = state.lastBody ? state.lastBody.source : state.source;
      const info = await fetchJSON(`/api/v1/scars/${encodeURIComponent(id)}?source=${src}`);
      if (state.selected !== id) return;
      state.selectedInfo = info;
      renderScarCard(info, row);
      if (state.quicklook !== "none") setQuicklook(state.quicklook);
    } catch (e) {
      toast(e.message);
    }
  }

  function closeScar() {
    state.selected = null;
    state.selectedInfo = null;
    map.setFilter("sel-line", ["==", ["get", "scar_id"], ""]);
    setQuicklook("none");
    $("#scarCard").hidden = true;
    $$("table.scars tbody tr.sel").forEach((tr) => tr.classList.remove("sel"));
  }

  function sevMiniBar(a, b, c) {
    const t = a + b + c || 1;
    return `<div class="minibar">${[a, b, c].map((v, i) => `<span style="width:${(100 * v) / t}%;background:${SEV[i + 1].color}"></span>`).join("")}</div>`;
  }

  function renderScarCard(info, row) {
    const s = row || info.scar;
    const c = info.chip;
    const conf = s.confidence != null ? `<dt>Уверенность модели</dt><dd>${fmt(s.confidence * 100, 1)} %</dd>` : "";
    const clipped = s.clipped ? `<div class="note">В запрос попадает часть контура: полная площадь ${fmtHa(s.area_total_ha)} га.</div>` : "";
    const hasQl = !!c.quicklook;
    $("#scarCard").innerHTML = `
      <div class="sc-head">
        <div><div class="sc-id">${esc(s.scar_id)}</div>
        <div class="sc-sub">Событие ${esc(c.fire_event_id || "—")} · чип ${esc(c.chip_id)}</div></div>
        <button class="sc-close" aria-label="Закрыть">×</button>
      </div>
      <div class="sc-area num">${fmtHa(s.area_ha)}<small>га${s.clipped ? " в запросе" : ""}</small></div>
      ${sevMiniBar(s.area_low_ha, s.area_moderate_ha, s.area_high_ha)}
      <div class="sc-sev">${[s.area_low_ha, s.area_moderate_ha, s.area_high_ha]
        .map((v, i) => `<div><span class="sw" style="background:${SEV[i + 1].color}"></span>${SEV[i + 1].label}<b class="num">${fmtHa(v)}</b></div>`)
        .join("")}</div>
      <dl class="sc-meta">
        <dt>Снимки до → после</dt><dd>${fmtDate(c.date_pre)} → ${fmtDate(c.date_post)}</dd>
        <dt>Преобладает</dt><dd>${esc(s.dominant_label || "—")} степень</dd>
        <dt>Пикселей 20 м</dt><dd>${fmt(s.n_pixels)}</dd>
        <dt>Площадь считалась</dt><dd>UTM, EPSG:${s.utm_epsg}</dd>
        ${conf}
      </dl>
      ${clipped}
      ${hasQl ? `<div class="card-title">Снимок Sentinel-2 (B12-B8A-B4)</div>
      <div class="seg seg-sm" id="qlSeg">
        <button type="button" data-ql="none">Нет</button><button type="button" data-ql="pre">До пожара</button><button type="button" data-ql="post">После</button>
      </div>` : ""}`;
    $("#scarCard").hidden = false;
    $(".sc-close", $("#scarCard")).onclick = closeScar;
    $$("#qlSeg button").forEach((b) => {
      b.classList.toggle("on", b.dataset.ql === state.quicklook);
      b.onclick = () => setQuicklook(b.dataset.ql);
    });
    enhanceSegs($("#scarCard"));
  }

  // ------------------------------------------------------------------ события карты
  function hotspotPopup(p, lngLat) {
    const t = p.acq_datetime;
    const d = new Date(t);
    const msk = new Date(d.getTime() + 3 * 3600e3);
    const mskStr = `${pad(msk.getUTCHours())}:${pad(msk.getUTCMinutes())} МСК`;
    const sat = { SNPP: "Suomi NPP", NOAA20: "NOAA-20", NOAA21: "NOAA-21" }[p.satellite] || p.satellite;
    const html = `
      <div class="pp-title">Термоточка · ${p.daynight === "N" ? "ночь" : "день"}</div>
      <div class="pp-sub">${fmtDate(t)}, ${t.slice(11, 16)} UTC (${mskStr}) · ${esc(sat)}</div>
      <dl class="pp-grid">
        <dt>Яркостная T, I4</dt><dd>${fmt(p.bright_ti4, 1)} K</dd>
        <dt>Яркостная T, I5</dt><dd>${fmt(p.bright_ti5, 1)} K</dd>
        <dt>ΔT (I4 − I5)</dt><dd>${fmt(p.delta_t, 1)} K</dd>
        <dt>Достоверность</dt><dd>${fmt(p.confidence * 100, 0)} %</dd>
        <dt>Покров</dt><dd>${esc(p.landcover)}</dd>
        <dt>Координаты</dt><dd>${fmt(p.latitude, 4)}° с. ш., ${fmt(p.longitude, 4)}° в. д.</dd>
      </dl>
      <div class="pp-id">${esc(p.hotspot_id)}</div>`;
    new maplibregl.Popup({ maxWidth: "300px", offset: 8 }).setLngLat(lngLat).setHTML(html).addTo(map);
  }

  function chipPopup(p, lngLat) {
    let html;
    if (p.tile_id) {
      const scenes = typeof p.scenes === "string" ? JSON.parse(p.scenes) : p.scenes;
      html = `<div class="pp-title">Сцена VIIRS · тайл 96 км</div>
        <div class="pp-sub">${p.n_scenes} ${plural(p.n_scenes, "съёмка", "съёмки", "съёмок")}: ${fmtDate(p.first)} — ${fmtDate(p.last)}</div>
        <div class="pp-id">${esc(scenes.slice(0, 6).map((s) => s.chip_id).join(", "))}${scenes.length > 6 ? "…" : ""}</div>`;
    } else {
      html = `<div class="pp-title">Чип Sentinel-2 · 10 км</div>
        <div class="pp-sub">${esc(p.chip_id)} · событие ${esc(p.fire_event_id || "—")}</div>
        <dl class="pp-grid"><dt>Снимки до / после</dt><dd>${fmtDate(p.date_pre)} / ${fmtDate(p.date_post)}</dd>
        <dt>Облачность</dt><dd>${fmt((p.cloud_frac || 0) * 100, 1)} %</dd></dl>`;
    }
    new maplibregl.Popup({ maxWidth: "300px" }).setLngLat(lngLat).setHTML(html).addTo(map);
  }

  function bindMapEvents() {
    const interactive = ["hotspots", "scar-pts", "scars-fill", "chips-bs-line", "chips-af-line"];
    map.on("mousemove", (e) => {
      if (state.drawing) return;
      const layers = interactive.filter((l) => map.getLayer(l) && map.getLayoutProperty(l, "visibility") !== "none");
      const f = map.queryRenderedFeatures(e.point, { layers });
      map.getCanvas().style.cursor = f.length ? "pointer" : "";
    });
    map.on("click", (e) => {
      if (state.drawing) return;
      const layers = interactive.filter((l) => map.getLayer(l) && map.getLayoutProperty(l, "visibility") !== "none");
      const fs = map.queryRenderedFeatures(e.point, { layers });
      if (!fs.length) return;
      const pr = (f) => layers.indexOf(f.layer.id);
      fs.sort((a, b) => pr(a) - pr(b));
      const f = fs[0];
      if (f.layer.id === "hotspots") hotspotPopup(f.properties, f.geometry.coordinates);
      else if (f.layer.id === "scar-pts") selectScar(f.properties.scar_id, true);
      else if (f.layer.id === "scars-fill") selectScar(f.properties.scar_id, false);
      else chipPopup(f.properties, e.lngLat);
    });
    map.on("moveend", refreshSeverityOverlays);

    // рисование
    map.on("mousedown", (e) => {
      const d = state.drawing;
      if (!d || d.type !== "rect") return;
      e.preventDefault();
      d.start = e.lngLat;
      d.startPt = e.point;
    });
    map.on("mousemove", (e) => {
      const d = state.drawing;
      if (!d) return;
      if (d.type === "rect" && d.start) {
        const b = [Math.min(d.start.lng, e.lngLat.lng), Math.min(d.start.lat, e.lngLat.lat), Math.max(d.start.lng, e.lngLat.lng), Math.max(d.start.lat, e.lngLat.lat)];
        map.getSource("draw").setData({ type: "Feature", properties: {}, geometry: bboxPolygon(b) });
      } else if (d.type === "polygon" && d.pts.length) {
        updatePolygonDraft([...d.pts, [e.lngLat.lng, e.lngLat.lat]]);
      }
    });
    map.on("mouseup", (e) => {
      const d = state.drawing;
      if (!d || d.type !== "rect" || !d.start) return;
      const dx = Math.abs(e.point.x - d.startPt.x), dy = Math.abs(e.point.y - d.startPt.y);
      if (dx < 6 || dy < 6) {
        d.start = null;
        map.getSource("draw").setData(EMPTY);
        return;
      }
      const b = [Math.min(d.start.lng, e.lngLat.lng), Math.min(d.start.lat, e.lngLat.lat), Math.max(d.start.lng, e.lngLat.lng), Math.max(d.start.lat, e.lngLat.lat)].map((v) => +v.toFixed(5));
      stopDrawing();
      state.bbox = b;
      fillBboxInputs();
      runQuery();
    });
    map.on("click", (e) => {
      const d = state.drawing;
      if (!d || d.type !== "polygon") return;
      const p = [e.lngLat.lng, e.lngLat.lat];
      if (d.pts.length >= 3) {
        const first = map.project(d.pts[0]);
        if (Math.hypot(first.x - e.point.x, first.y - e.point.y) < 10) return finishPolygon();
      }
      if (d.pts.length) {
        const last = map.project(d.pts[d.pts.length - 1]);
        if (Math.hypot(last.x - e.point.x, last.y - e.point.y) < 4) return;
      }
      d.pts.push(p);
      updatePolygonDraft(d.pts);
    });
    map.on("dblclick", (e) => {
      if (state.drawing && state.drawing.type === "polygon") {
        e.preventDefault();
        finishPolygon();
      }
    });
  }

  function updatePolygonDraft(pts) {
    const feats = pts.map((p) => ({ type: "Feature", properties: {}, geometry: { type: "Point", coordinates: p } }));
    if (pts.length >= 3) feats.push({ type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: [[...pts, pts[0]]] } });
    else if (pts.length === 2) feats.push({ type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: pts } });
    map.getSource("draw").setData({ type: "FeatureCollection", features: feats });
  }

  function startDrawing(type) {
    stopDrawing(true);
    state.drawing = { type, pts: [], prevMode: state.mode };
    map.dragPan.disable();
    map.doubleClickZoom.disable();
    map.getCanvas().style.cursor = "crosshair";
    $(`#territorySeg [data-mode="${type === "rect" ? "bbox" : "polygon"}"]`).classList.add("drawing");
    const hint = $("#drawHint");
    hint.innerHTML =
      type === "rect"
        ? "Зажмите кнопку мыши и протяните прямоугольник · <kbd>Esc</kbd> — отмена"
        : "Щелчками задайте вершины · двойной щелчок — готово · <kbd>Backspace</kbd> — убрать вершину · <kbd>Esc</kbd> — отмена";
    hint.hidden = false;
  }

  function stopDrawing(silent) {
    if (!state.drawing) return;
    state.drawing = null;
    map.dragPan.enable();
    setTimeout(() => map.doubleClickZoom.enable(), 350);
    map.getCanvas().style.cursor = "";
    map.getSource("draw").setData(EMPTY);
    $$("#territorySeg button").forEach((b) => b.classList.remove("drawing"));
    $("#drawHint").hidden = true;
  }

  function finishPolygon() {
    const d = state.drawing;
    const pts = d.pts;
    if (pts.length < 3) return toast("Для полигона нужно минимум три вершины.", "info");
    stopDrawing();
    state.geometry = { type: "Polygon", coordinates: [[...pts.map((p) => [+p[0].toFixed(6), +p[1].toFixed(6)]), [+pts[0][0].toFixed(6), +pts[0][1].toFixed(6)]]] };
    runQuery();
  }

  function cancelDrawing() {
    const d = state.drawing;
    if (!d) return;
    stopDrawing();
    // возврат к предыдущей территории, если новая так и не задана
    if ((d.type === "rect" && !state.bbox) || (d.type === "polygon" && !state.geometry)) setMode(d.prevMode === "bbox" || d.prevMode === "polygon" ? "all" : d.prevMode, false);
  }

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (state.drawing) cancelDrawing();
      else if (state.selected) closeScar();
    }
    if (e.key === "Backspace" && state.drawing && state.drawing.type === "polygon" && document.activeElement === document.body) {
      e.preventDefault();
      state.drawing.pts.pop();
      updatePolygonDraft(state.drawing.pts);
    }
  });

  // ------------------------------------------------------------------ территория
  function setMode(mode, run = true) {
    const prev = state.mode;
    $$("#territorySeg button").forEach((b) => b.classList.toggle("on", b.dataset.mode === mode));
    $("#bboxRow").hidden = mode !== "bbox";
    if (mode === "file") {
      $("#fileInput").value = "";
      $("#fileInput").click();
      $$("#territorySeg button").forEach((b) => b.classList.toggle("on", b.dataset.mode === prev));
      return;
    }
    state.mode = mode;
    if (mode === "all") {
      stopDrawing();
      state.geometry = null;
      if (run) runQuery();
    } else if (mode === "aoi") {
      stopDrawing();
      state.geometry = (state.aoi.features.find((f) => f.id === "aoi") || state.aoi.features[0]).geometry;
      if (run) runQuery();
    } else if (mode === "bbox") {
      state.geometry = null;
      startDrawing("rect");
    } else if (mode === "polygon") {
      state.geometry = null;
      startDrawing("polygon");
    }
  }

  function fillBboxInputs() {
    const b = state.bbox || [null, null, null, null];
    ["#bbMinLon", "#bbMinLat", "#bbMaxLon", "#bbMaxLat"].forEach((s, i) => ($(s).value = b[i] ?? ""));
  }

  function readBboxInputs() {
    const v = ["#bbMinLon", "#bbMinLat", "#bbMaxLon", "#bbMaxLat"].map((s) => parseFloat($(s).value));
    if (v.some((x) => Number.isNaN(x))) return;
    state.bbox = v;
    stopDrawing();
    map.getSource("draw").setData({ type: "Feature", properties: {}, geometry: bboxPolygon(v) });
    markDirty();
  }

  async function onFile(e) {
    const f = e.target.files && e.target.files[0];
    if (!f) return;
    try {
      const gj = JSON.parse(await f.text());
      if (!polygonsOf(gj).length) throw new Error("в файле нет полигонов (Polygon/MultiPolygon)");
      stopDrawing();
      state.mode = "file";
      state.fileName = f.name;
      state.geometry = gj.type === "Feature" ? gj.geometry : gj;
      $$("#territorySeg button").forEach((b) => b.classList.toggle("on", b.dataset.mode === "file"));
      $("#bboxRow").hidden = true;
      map.fitBounds(bboxOf(gj), { padding: { top: 60, bottom: 150, left: 60, right: 60 }, duration: 700 });
      runQuery();
    } catch (err) {
      toast(`Не удалось прочитать ${f.name}: ${err.message}`);
    }
  }

  // ------------------------------------------------------------------ микроанимации
  const reduceMotion = () => window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;

  /** Скользящая подложка у сегментированных переключателей: следует за кнопкой .on. */
  function enhanceSegs(root) {
    $$(".seg", root).forEach((seg) => {
      if (seg.querySelector(":scope > .seg-ind")) return;
      const ind = document.createElement("span");
      ind.className = "seg-ind";
      seg.prepend(ind);
      seg.classList.add("has-ind");
      const sync = () => {
        const on = seg.querySelector(":scope > button.on");
        if (!on || !on.offsetWidth) {
          ind.style.opacity = "0";
          return;
        }
        ind.style.opacity = "1";
        ind.style.width = `${on.offsetWidth}px`;
        ind.style.height = `${on.offsetHeight}px`;
        ind.style.transform = `translate(${on.offsetLeft}px, ${on.offsetTop}px)`;
      };
      sync();
      requestAnimationFrame(() => ind.classList.add("ready"));
      new MutationObserver(sync).observe(seg, { subtree: true, attributes: true, attributeFilter: ["class"] });
      new ResizeObserver(sync).observe(seg);
    });
  }

  /** Числа KPI «докручиваются» от прошлого значения к новому. */
  const counted = new Map();
  const countUp = (key, v, d = 0) =>
    `<span class="countup" data-key="${key}" data-v="${v == null || Number.isNaN(Number(v)) ? "" : Number(v)}" data-d="${d}">${fmt(v, d)}</span>`;

  function runCountUps(root) {
    $$(".countup", root).forEach((el) => {
      if (el.dataset.v === "") return;
      const to = Number(el.dataset.v), d = Number(el.dataset.d), key = el.dataset.key;
      const from = counted.has(key) ? counted.get(key) : 0;
      counted.set(key, to);
      if (reduceMotion() || from === to) return;
      const t0 = performance.now(), dur = 900;
      const tick = (t) => {
        const p = Math.min(1, (t - t0) / dur);
        const e = 1 - Math.pow(1 - p, 4);
        el.textContent = fmt(from + (to - from) * e, d);
        if (p < 1 && el.isConnected) requestAnimationFrame(tick);
      };
      el.textContent = fmt(from, d);
      requestAnimationFrame(tick);
    });
  }

  // ------------------------------------------------------------------ справка
  function renderReport() {
    const r = state.report;
    const s = r.summary;
    const q = r.query;
    const srcLabel = q.source === "model" ? "модель" : "эталон";
    const area = q.area_km2 ? `<span class="sep">·</span>${fmt(q.area_km2, 0)} км²` : "";
    $("#reportQuery").innerHTML = `${esc(territoryLabel())}${area}<span class="sep">·</span>${fmtDate(q.date_from)} — ${fmtDate(q.date_to)}<span class="sep">·</span>${srcLabel}`;

    const clippedNote = s.n_scars_clipped
      ? `${s.n_scars_clipped} ${plural(s.n_scars_clipped, "контур обрезан", "контура обрезаны", "контуров обрезаны")} границей запроса`
      : `по ${s.n_bs_chips} ${plural(s.n_bs_chips, "чипу", "чипам", "чипам")} Sentinel-2`;

    // пустой результат: объясняем, почему нулевые числа, и что делать дальше
    const noData = !s.n_scars && !s.n_hotspots;
    let html = "";
    if (noData) {
      const m = state.meta;
      const outside = !s.n_bs_chips && !s.n_af_scenes;
      html += `<div class="nodata"><b>${outside ? "В границах запроса нет данных каталога" : "За этот период гарей и термоточек не найдено"}</b>
        <span>${
          outside
            ? `Каталог покрывает только территорию мониторинга (${fmt(m.n_bs_chips)} ${plural(m.n_bs_chips, "чип", "чипа", "чипов")} Sentinel-2 и ${fmt(m.n_af_scenes)} ${plural(m.n_af_scenes, "сцена", "сцены", "сцен")} VIIRS, ${m.date_min.slice(0, 4)}–${m.date_max.slice(0, 4)}). Выберите «AOI» или «Весь каталог» — или задайте территорию внутри границы AOI.`
            : `В запрос попало ${fmt(s.n_bs_chips)} ${plural(s.n_bs_chips, "чип", "чипа", "чипов")} Sentinel-2 и ${fmt(s.n_af_scenes)} ${plural(s.n_af_scenes, "сцена", "сцены", "сцен")} VIIRS, но событий в них нет. Расширьте период (данные за ${m.date_min.slice(0, 4)}–${m.date_max.slice(0, 4)}) или территорию.`
        }</span></div>`;
    }
    html += `
      <div class="kpis">
        <div class="kpi main"><div class="k-label">Площадь гарей</div><div class="k-val">${countUp("area", s.burned_area_ha, 2)}<small>га</small></div><div class="k-sub">${clippedNote}${s.burned_share_of_query_pct != null ? ` · ${fmt(s.burned_share_of_query_pct, 3)} % площади запроса` : ""}</div></div>
        <div class="kpi"><div class="k-label">Контуры гарей</div><div class="k-val">${countUp("scars", s.n_scars)}</div><div class="k-sub">связные области пикселей</div></div>
        <div class="kpi"><div class="k-label">Термоточки</div><div class="k-val">${countUp("hotspots", s.n_hotspots)}</div><div class="k-sub">днём ${fmt(s.n_hotspots_day)} · ночью ${fmt(s.n_hotspots_night)}</div></div>
      </div>`;

    // степень
    const tot = s.burned_area_ha || 0;
    html += `<section class="sec"><div class="sec-h"><h2>Площадь по степени поражения</h2><span class="aside">га · доля</span></div>
      <div class="sevbar" role="img" aria-label="Распределение площади по степени поражения">${r.severity
        .map((x) => `<span style="width:${tot ? (100 * x.area_ha) / tot : 0}%;background:${x.color}" title="${x.label}: ${fmtHa(x.area_ha)} га"></span>`)
        .join("")}</div>
      <table class="sevtbl"><tbody>${r.severity
        .map((x) => `<tr><td><span class="sw" style="background:${x.color}"></span>${x.label}</td><td class="num">${fmtHa(x.area_ha)} га</td><td class="num pct">${fmt(x.share_pct, 2)} %</td></tr>`)
        .join("")}<tr class="total"><td>Всего</td><td class="num">${fmtHa(tot)} га</td><td class="num pct">${tot ? "100,00 %" : "—"}</td></tr></tbody></table>
      ${s.n_scars_clipped ? `<div class="note">Без обрезки по границе запроса: ${fmtHa(s.burned_area_unclipped_ha)} га.</div>` : ""}
    </section>`;

    // сравнение (при пустом результате сравнивать нечего)
    const c = noData ? null : r.comparison;
    if (c) {
      const pa = c.pixel_agreement, ha = c.hotspot_agreement;
      const m = (v) => (v == null ? "—" : fmt(v, 2));
      const rows = [
        ["Всего", "burned_area_ha", "var(--ink-2)"],
        ["слабая", "area_low_ha", SEV[1].color],
        ["средняя", "area_moderate_ha", SEV[2].color],
        ["сильная", "area_high_ha", SEV[3].color],
      ];
      const mx = Math.max(1, ...rows.flatMap(([, k]) => [c.model[k], c.reference[k]]));
      html += `<section class="sec"><div class="sec-h"><h2>Модель и эталон</h2><span class="aside">попиксельно, 20 м</span></div>
        <div class="metrics">
          <div class="metric"><div class="m-val">${m(pa.iou_burned)}</div><div class="m-lab">IoU гарей</div></div>
          <div class="metric"><div class="m-val">${m(pa.f1)}</div><div class="m-lab">F1 гарей</div></div>
          <div class="metric"><div class="m-val">${m(pa.severity_accuracy_on_common)}</div><div class="m-lab">Точность степени</div></div>
          <div class="metric"><div class="m-val">${m(ha.f1)}</div><div class="m-lab">F1 термоточек</div></div>
        </div>
        <div class="cmp-rows">${rows
          .map(
            ([lab, k, col]) => `<div class="lab">${lab}</div>
            <div class="bars"><div class="bar" style="width:${(100 * c.model[k]) / mx}%;background:${col}"></div><div class="bar ref" style="width:${(100 * c.reference[k]) / mx}%;background:${col}"></div></div>
            <div class="vals">${fmt(c.model[k], 0)}<br>${fmt(c.reference[k], 0)} га</div>`
          )
          .join("")}</div>
        <div class="cmp-legend"><span><i></i>модель</span><span><i class="ref"></i>эталон</span><span>разница площади: ${c.burned_area_diff_pct == null ? "—" : `${c.burned_area_diff_pct > 0 ? "+" : ""}${fmt(c.burned_area_diff_pct, 1)} %`}</span></div>
      </section>`;
    }

    // термоточки во времени
    const hs = r.hotspots;
    if (hs.total) {
      const sats = hs.by_satellite.map((x) => `${x.label} ${fmt(x.count)}`).join(" · ");
      html += `<section class="sec"><div class="sec-h"><h2>Термоточки во времени</h2><span class="aside">${fmt(hs.by_date.length)} ${plural(hs.by_date.length, "дата", "даты", "дат")} съёмки</span></div>
        ${hotspotChart(hs)}
        <div class="note">Спутники: ${esc(sats)}.${hs.bright_ti4_k ? ` Яркостная температура I4: ${fmt(hs.bright_ti4_k.min, 0)}–${fmt(hs.bright_ti4_k.max, 0)} K, среднее ${fmt(hs.bright_ti4_k.mean, 1)} K.` : ""}</div>
      </section>`;
    }

    // покров
    if (r.landcover.length) {
      const mxl = Math.max(...r.landcover.map((x) => x.area_ha)) || 1;
      html += `<section class="sec"><div class="sec-h"><h2>Гари по типам покрова</h2><span class="aside">ESA WorldCover</span></div>
        <div class="hbars">${r.landcover
          .slice(0, 6)
          .map(
            (x) => `<div class="n" title="${esc(x.name)}">${esc(x.name)}</div>
            <div class="t" style="width:${Math.max(2, (100 * x.area_ha) / mxl)}%">${[x.area_low_ha, x.area_moderate_ha, x.area_high_ha]
              .map((v, i) => `<span style="width:${x.area_ha ? (100 * v) / x.area_ha : 0}%;background:${SEV[i + 1].color}"></span>`)
              .join("")}</div>
            <div class="v">${fmt(x.share_pct, 1)} %</div>`
          )
          .join("")}</div>
        ${r.landcover_estimated ? `<div class="note">Для обрезанных контуров площадь по покрову оценена пропорционально числу пикселей.</div>` : ""}
      </section>`;
    }

    // таблица контуров
    html += `<section class="sec"><div class="sec-h"><h2>Контуры гарей</h2><span class="aside">${fmt(r.scars.length)} · щелчок — показать на карте</span></div>
      <div class="tbl-wrap" id="scarTable"></div></section>`;

    // методика и API
    const apiBody = JSON.stringify(state.lastBody || {});
    const curl = `curl -X POST '${location.origin}/api/v1/report?format=json' \\\n  -H 'Content-Type: application/json' \\\n  -d '${apiBody}'`;
    html += `<section class="sec"><details class="method"><summary>Как посчитаны площади</summary>
        <p>${esc(r.method.burned_area)}</p><p>${esc(r.method.time_filter)}</p><p>${esc(r.method.hotspots)}</p><p>${esc(r.method.crs_output)}</p>
      </details></section>
      <section class="sec"><details class="method"><summary>Тот же запрос через REST API</summary>
        <div class="curl-head"><span>curl · POST /api/v1/report</span><button class="copy" type="button">Копировать</button></div>
        <div class="curl">${esc(curl)}</div>
        <p class="note">Документация и примеры: <a href="/docs" target="_blank" rel="noopener">/docs</a>. Выгрузки — /api/v1/burn-scars, /api/v1/hotspots, /api/v1/report.</p>
      </details></section>`;

    const body = $("#reportBody");
    body.innerHTML = html;
    body.scrollTop = 0;
    runCountUps(body);
    renderScarTable();
    const cp = $(".copy", body);
    if (cp)
      cp.onclick = () => {
        navigator.clipboard && navigator.clipboard.writeText(curl).then(() => toast("Команда скопирована в буфер обмена.", "info"));
      };
  }

  function hotspotChart(hs) {
    let bins = hs.by_date.map((d) => ({ key: d.date, day: d.day, night: d.night }));
    let monthly = false;
    if (bins.length > 36) {
      monthly = true;
      const m = new Map();
      for (const d of hs.by_date) {
        const k = d.date.slice(0, 7);
        const v = m.get(k) || { key: k, day: 0, night: 0 };
        v.day += d.day;
        v.night += d.night;
        m.set(k, v);
      }
      bins = [...m.values()];
    }
    const W = 400, H = 96, top = 6, bottom = 18;
    const mx = Math.max(...bins.map((b) => b.day + b.night), 1);
    const bw = Math.max(2, Math.min(26, (W - 2) / bins.length - 2));
    const step = (W - bw) / Math.max(1, bins.length - 1);
    const y = (v) => ((H - top - bottom) * v) / mx;
    const lab = (k) => (monthly ? `${MONTHS_SHORT[+k.slice(5, 7) - 1]} ${k.slice(0, 4)}` : fmtDate(k));
    const bars = bins
      .map((b, i) => {
        const x = bins.length === 1 ? (W - bw) / 2 : i * step;
        const hd = y(b.day), hn = y(b.night);
        const base = H - bottom;
        return `<g><title>${lab(b.key)}: ${b.day + b.night} (день ${b.day}, ночь ${b.night})</title>
          <rect class="bar-d" x="${x.toFixed(1)}" y="${(base - hd).toFixed(1)}" width="${bw.toFixed(1)}" height="${Math.max(hd, b.day ? 1 : 0).toFixed(1)}" rx="1"/>
          <rect class="bar-n" x="${x.toFixed(1)}" y="${(base - hd - hn).toFixed(1)}" width="${bw.toFixed(1)}" height="${Math.max(hn, b.night ? 1 : 0).toFixed(1)}" rx="1"/></g>`;
      })
      .join("");
    const first = bins[0], last = bins[bins.length - 1];
    return `<svg class="minichart" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="Число термоточек по ${monthly ? "месяцам" : "датам"}">
      <line class="axis" x1="0" x2="${W}" y1="${H - bottom + 0.5}" y2="${H - bottom + 0.5}"/>
      <text x="0" y="10">макс. ${fmt(mx)}</text>
      ${bars}
      <text x="0" y="${H - 4}">${lab(first.key)}</text>
      ${bins.length > 1 ? `<text x="${W}" y="${H - 4}" text-anchor="end">${lab(last.key)}</text>` : ""}
    </svg><div class="cmp-legend"><span><i style="background:var(--hot-day)"></i>день</span><span><i style="background:var(--hot-night)"></i>ночь</span><span>${monthly ? "по месяцам" : "по датам съёмки"}</span></div>`;
  }

  function renderScarTable() {
    const r = state.report;
    const cols = [
      ["scar_id", "Контур", ""],
      ["date_post", "Дата", ""],
      ["area_ha", "Площадь, га", "num"],
      ["share_high_pct", "Степень", ""],
    ];
    const k = state.sortKey, dir = state.sortDir;
    const rows = r.scars.slice().sort((a, b) => (a[k] < b[k] ? -dir : a[k] > b[k] ? dir : 0));
    const shown = rows.slice(0, state.tableLimit);
    const head = cols
      .map(([key, lab, cls]) => `<th class="${cls}" data-k="${key}" style="width:${{ scar_id: "40%", date_post: "20%", area_ha: "22%", share_high_pct: "18%" }[key]}" title="${key === "date_post" ? "Дата снимка после пожара" : key === "share_high_pct" ? "Доли слабой, средней и сильной степени; сортировка — по доле сильной" : ""}">${lab}${key === k ? `<span class="arr">${dir < 0 ? "↓" : "↑"}</span>` : ""}</th>`)
      .join("");
    const body = shown
      .map(
        (s) => `<tr data-id="${esc(s.scar_id)}" class="${s.scar_id === state.selected ? "sel" : ""}">
          <td class="mono" title="${esc(s.scar_id)}${s.clipped ? " — обрезан границей запроса" : ""}">${esc(s.scar_id)}${s.clipped ? '<span class="clip-badge">✂</span>' : ""}</td>
          <td class="num">${fmtDate(s.date_post)}</td>
          <td class="num">${fmtHa(s.area_ha)}</td>
          <td>${sevMiniBar(s.area_low_ha, s.area_moderate_ha, s.area_high_ha)}</td></tr>`
      )
      .join("");
    const more = rows.length > shown.length ? `<button type="button" class="more">Показать ещё ${Math.min(200, rows.length - shown.length)} из ${fmt(rows.length - shown.length)}</button>` : "";
    const el = $("#scarTable");
    el.innerHTML = rows.length
      ? `<table class="scars"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>${more}`
      : `<div class="empty">В запросе нет контуров гарей — расширьте территорию или период.</div>`;
    $$("th", el).forEach(
      (th) =>
        (th.onclick = () => {
          const key = th.dataset.k;
          if (state.sortKey === key) state.sortDir = -state.sortDir;
          else {
            state.sortKey = key;
            state.sortDir = key === "scar_id" ? 1 : -1;
          }
          renderScarTable();
        })
    );
    $$("tbody tr", el).forEach((tr) => (tr.onclick = () => selectScar(tr.dataset.id, true)));
    const mb = $(".more", el);
    if (mb)
      mb.onclick = () => {
        state.tableLimit += 200;
        renderScarTable();
      };
  }

  // ------------------------------------------------------------------ лента времени
  const tl = { months: [], i0: 0, i1: 0, drag: null };

  function monthIndex(dateStr) {
    const y = +dateStr.slice(0, 4), m = +dateStr.slice(5, 7);
    const y0 = +tl.months[0].slice(0, 4), m0 = +tl.months[0].slice(5, 7);
    return (y - y0) * 12 + (m - m0);
  }

  function renderTimeline() {
    const meta = state.meta;
    if (!meta || !meta.timeline) return;
    tl.months = meta.timeline.months;
    const src = (state.lastBody && state.lastBody.source) || state.source;
    const ser = meta.timeline.sources[src] || meta.timeline.sources.reference;
    const svg = $("#tlSvg");
    const W = svg.clientWidth || 800, H = svg.clientHeight || 60;
    const n = tl.months.length;
    const lab = 13, upH = Math.round((H - lab) * 0.56), dnH = H - lab - upH - 2;
    const base = upH;
    const step = W / n;
    const bw = Math.max(1, step - (step > 6 ? 2 : 1));
    const mxh = Math.max(...ser.hotspots, 1), mxb = Math.max(...ser.burned_ha, 1);
    let out = "";
    // годы
    let prevY = null;
    const years = [];
    tl.months.forEach((m, i) => {
      const y = m.slice(0, 4);
      if (y !== prevY) years.push({ y, i });
      prevY = y;
    });
    years.forEach((yr, k) => {
      const x0 = yr.i * step, x1 = (k + 1 < years.length ? years[k + 1].i : n) * step;
      if (k) out += `<line x1="${x0}" x2="${x0}" y1="0" y2="${H}" stroke="#ffffff" stroke-opacity="0.07"/>`;
      out += `<g class="yr" data-year="${yr.y}"><rect x="${x0}" y="${H - lab}" width="${x1 - x0}" height="${lab}" fill="transparent"/><text x="${(x0 + x1) / 2}" y="${H - 2}" text-anchor="middle">${yr.y}</text></g>`;
    });
    // выделение периода
    const df = $("#dateFrom").value, dt = $("#dateTo").value;
    if (df && dt) {
      tl.i0 = Math.max(0, Math.min(n - 1, monthIndex(df)));
      tl.i1 = Math.max(0, Math.min(n - 1, monthIndex(dt)));
    }
    const sx0 = Math.min(tl.i0, tl.i1) * step, sx1 = (Math.max(tl.i0, tl.i1) + 1) * step;
    out += `<rect x="${sx0}" y="0" width="${sx1 - sx0}" height="${H - lab}" fill="${QUERY}" fill-opacity="0.12" rx="3"/>`;
    // бары
    for (let i = 0; i < n; i++) {
      const x = i * step + (step - bw) / 2;
      const inSel = i >= Math.min(tl.i0, tl.i1) && i <= Math.max(tl.i0, tl.i1);
      const h1 = ser.hotspots[i] ? Math.max(1.5, ((upH - 2) * ser.hotspots[i]) / mxh) : 0;
      const h2 = ser.burned_ha[i] ? Math.max(1.5, (dnH * ser.burned_ha[i]) / mxb) : 0;
      const op = inSel ? 1 : 0.22;
      if (h1) out += `<rect x="${x.toFixed(1)}" y="${(base - h1).toFixed(1)}" width="${bw.toFixed(1)}" height="${h1.toFixed(1)}" fill="${HOT.D}" opacity="${op}" rx="0.8"/>`;
      if (h2) out += `<rect x="${x.toFixed(1)}" y="${base + 2}" width="${bw.toFixed(1)}" height="${h2.toFixed(1)}" fill="${TL_BURN}" opacity="${op}" rx="0.8"/>`;
    }
    out += `<line x1="0" x2="${W}" y1="${base + 1}" y2="${base + 1}" stroke="#ffffff" stroke-opacity="0.16"/>`;
    out += `<line x1="${sx0}" x2="${sx0}" y1="0" y2="${H - lab}" stroke="${QUERY}" stroke-width="2"/><line x1="${sx1}" x2="${sx1}" y1="0" y2="${H - lab}" stroke="${QUERY}" stroke-width="2"/>`;
    out += `<rect id="tlHover" x="0" y="0" width="${step}" height="${H - lab}" fill="#ffffff" fill-opacity="0.07" visibility="hidden"/>`;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.innerHTML = out;
    tl.step = step;
    tl.W = W;
    tl.ser = ser;
    tl.labH = lab;
    tl.H = H;
    $$(".yr", svg).forEach((g) =>
      g.addEventListener("click", (e) => {
        e.stopPropagation();
        const y = g.dataset.year;
        const season = $("#seasonOnly").checked;
        setDates(season ? `${y}-04-01` : `${y}-01-01`, season ? `${y}-10-31` : `${y}-12-31`);
        runQuery();
      })
    );
  }

  function setDates(a, b) {
    const mn = state.meta.date_min, mx = state.meta.date_max;
    const clamp = (d) => (d < mn ? mn : d > mx ? mx : d);
    $("#dateFrom").value = clamp(a);
    $("#dateTo").value = clamp(b);
    renderTimeline();
  }

  function tlIndex(evt) {
    const svg = $("#tlSvg");
    const r = svg.getBoundingClientRect();
    const x = ((evt.clientX - r.left) / r.width) * tl.W;
    return Math.max(0, Math.min(tl.months.length - 1, Math.floor(x / tl.step)));
  }

  function tlReadout(i) {
    const m = tl.months[i];
    if (!m) return "";
    return `${MONTHS[+m.slice(5, 7) - 1]} ${m.slice(0, 4)}: ${fmt(tl.ser.hotspots[i])} ${plural(tl.ser.hotspots[i], "термоточка", "термоточки", "термоточек")} · ${fmtHa(tl.ser.burned_ha[i])} га гарей`;
  }

  function bindTimeline() {
    const svg = $("#tlSvg");
    svg.addEventListener("pointerdown", (e) => {
      if (e.target.closest(".yr")) return;
      svg.setPointerCapture(e.pointerId);
      const i = tlIndex(e);
      tl.drag = { a: i };
      tl.i0 = tl.i1 = i;
    });
    svg.addEventListener("pointermove", (e) => {
      const i = tlIndex(e);
      const hv = $("#tlHover");
      if (hv) {
        hv.setAttribute("x", i * tl.step);
        hv.setAttribute("visibility", "visible");
      }
      svg.setAttribute("aria-description", tlReadout(i));
      $("#tlReadout").textContent = tlReadout(i);
      $("#timeline").classList.add("hovering");
      if (!tl.drag) return;
      tl.i0 = Math.min(tl.drag.a, i);
      tl.i1 = Math.max(tl.drag.a, i);
      const m0 = tl.months[tl.i0], m1 = tl.months[tl.i1];
      $("#dateFrom").value = `${m0}-01`;
      $("#dateTo").value = `${m1}-${pad(lastDay(+m1.slice(0, 4), +m1.slice(5, 7)))}`;
      renderTimeline();
    });
    svg.addEventListener("pointerleave", () => {
      const hv = $("#tlHover");
      if (hv) hv.setAttribute("visibility", "hidden");
      $("#tlReadout").textContent = "";
      $("#timeline").classList.remove("hovering");
    });
    svg.addEventListener("pointerup", () => {
      if (!tl.drag) return;
      tl.drag = null;
      const m0 = tl.months[tl.i0], m1 = tl.months[tl.i1];
      setDates(`${m0}-01`, `${m1}-${pad(lastDay(+m1.slice(0, 4), +m1.slice(5, 7)))}`);
      runQuery();
    });
    new ResizeObserver(() => renderTimeline()).observe(svg);
  }

  // ------------------------------------------------------------------ выгрузки
  const EXPORTS = {
    "scars-geojson": ["/api/v1/burn-scars", { layer: "scars", format: "geojson" }],
    "scars-shp": ["/api/v1/burn-scars", { layer: "scars", format: "shp" }],
    "scars-csv": ["/api/v1/burn-scars", { layer: "scars", format: "csv" }],
    "zones-geojson": ["/api/v1/burn-scars", { layer: "zones", format: "geojson" }],
    "zones-shp": ["/api/v1/burn-scars", { layer: "zones", format: "shp" }],
    "hs-geojson": ["/api/v1/hotspots", { format: "geojson" }],
    "hs-csv": ["/api/v1/hotspots", { format: "csv" }],
    "hs-shp": ["/api/v1/hotspots", { format: "shp" }],
    "rep-json": ["/api/v1/report", { format: "json" }],
    "rep-csv": ["/api/v1/report", { format: "csv" }],
    "rep-zip": ["/api/v1/report", { format: "zip" }],
  };

  async function download(key, btn) {
    if (!state.lastBody) return toast("Сначала постройте запрос.");
    const [url, params] = EXPORTS[key];
    btn.classList.add("busy");
    try {
      const r = await fetch(`${url}?${new URLSearchParams(params)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(state.lastBody),
      });
      if (!r.ok) {
        let msg = `HTTP ${r.status}`;
        try {
          msg = (await r.json()).message || msg;
        } catch (_) {
          /* пусто */
        }
        throw new Error(msg);
      }
      const blob = await r.blob();
      const cd = r.headers.get("Content-Disposition") || "";
      const m = /filename="([^"]+)"/.exec(cd);
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = m ? m[1] : `${key}.dat`;
      document.body.appendChild(a);
      a.click();
      setTimeout(() => {
        URL.revokeObjectURL(a.href);
        a.remove();
      }, 1500);
    } catch (e) {
      toast(`Выгрузка не удалась: ${e.message}`);
    } finally {
      btn.classList.remove("busy");
    }
  }

  // ------------------------------------------------------------------ запуск
  function bindUI() {
    $("#queryForm").addEventListener("submit", (e) => {
      e.preventDefault();
      if (state.mode === "bbox") readBboxInputs();
      runQuery();
    });
    $$("#territorySeg button").forEach((b) => (b.onclick = () => setMode(b.dataset.mode)));
    $$("#sourceSeg button").forEach(
      (b) =>
        (b.onclick = () => {
          if (b.disabled) return;
          state.source = b.dataset.source;
          $$("#sourceSeg button").forEach((x) => x.classList.toggle("on", x === b));
          closeScar();
          runQuery();
        })
    );
    $$("#basemapSeg button").forEach((b) => (b.onclick = () => setBasemap(b.dataset.basemap)));
    const lt = $("#layersToggle");
    const setLayersOpen = (open) => {
      lt.setAttribute("aria-expanded", String(open));
      $("#layersPanel").hidden = !open;
    };
    setLayersOpen(window.innerWidth >= 1500 && window.innerHeight >= 820);
    lt.onclick = () => setLayersOpen(lt.getAttribute("aria-expanded") !== "true");
    $$("#layersPanel input[data-layer]").forEach((i) => (i.onchange = () => setLayerVisible(i.dataset.layer, i.checked)));
    ["#dateFrom", "#dateTo"].forEach((s) =>
      $(s).addEventListener("change", () => {
        markDirty();
        renderTimeline();
      })
    );
    ["#bbMinLon", "#bbMinLat", "#bbMaxLon", "#bbMaxLat"].forEach((s) => $(s).addEventListener("change", readBboxInputs));
    $("#fileInput").addEventListener("change", onFile);
    $("#tlAll").onclick = () => {
      setDates(state.meta.date_min, state.meta.date_max);
      runQuery();
    };
    $$("#exports button[data-exp]").forEach((b) => (b.onclick = () => download(b.dataset.exp, b)));
    const ro = document.createElement("span");
    ro.id = "tlReadout";
    ro.style.cssText = "color:var(--ink-2);font-variant-numeric:tabular-nums;white-space:nowrap";
    $(".tl-spacer").before(ro);
    bindTimeline();
  }

  async function boot() {
    bindUI();
    enhanceSegs(document);
    try {
      const [meta, aoi, bs] = await Promise.all([fetchJSON("/api/v1/meta"), fetchJSON("/api/v1/aoi").catch(() => null), fetchJSON("/api/v1/chips?kind=bs")]);
      state.meta = meta;
      state.aoi = aoi;
      state.bsChips = bs;
      state.source = meta.default_source;
      $$("#sourceSeg button").forEach((b) => {
        const ok = !!meta.sources[b.dataset.source];
        b.disabled = !ok;
        if (!ok) b.title = "Нет в каталоге: соберите каталог с --pred-dir";
        b.classList.toggle("on", b.dataset.source === state.source);
      });
      if (!aoi) $('#territorySeg [data-mode="aoi"]').disabled = true;
      $("#dateFrom").min = $("#dateTo").min = meta.date_min;
      $("#dateFrom").max = $("#dateTo").max = meta.date_max;
      $("#dateFrom").value = meta.date_min;
      $("#dateTo").value = meta.date_max;
      const y0 = meta.date_min.slice(0, 4), y1 = meta.date_max.slice(0, 4);
      const fake = meta.catalog && meta.catalog.pred_dir && /fake/i.test(meta.catalog.pred_dir);
      $("#catalogStatus").classList.add("ok");
      $("#catalogStatus").innerHTML = `${fake ? '<span class="warn-badge" title="Каталог собран с синтетическими предсказаниями для отладки — пересоберите с --pred-dir реальной модели">модель: тестовые данные</span> ' : ""}Каталог: <b>${meta.n_bs_chips}</b> ${plural(meta.n_bs_chips, "чип", "чипа", "чипов")} Sentinel-2 · <b>${meta.n_af_scenes}</b> ${plural(meta.n_af_scenes, "сцена", "сцены", "сцен")} VIIRS · ${y0}–${y1}`;
      initMap();
      renderTimeline();
    } catch (e) {
      $("#catalogStatus").textContent = "Каталог недоступен";
      toast(`Не удалось загрузить каталог: ${e.message}`);
    }
  }

  // отладочный доступ (консоль браузера, автотесты интерфейса)
  window.firemon = { state, map: () => map, runQuery, selectScar };
  boot();
})();
