"""Тесты REST API сервиса (работают на собранном каталоге service/data/catalog)."""
from __future__ import annotations

import csv
import io
import json
import zipfile
from datetime import date

import numpy as np
import pytest
import shapely
from pyproj import Transformer
from shapely.geometry import shape

BBOX = [43.0, 47.0, 46.0, 50.0]


def _post(client, url, body, **params):
    r = client.post(url, json=body, params=params)
    assert r.status_code == 200, r.text
    return r


def _sum(values):
    return round(sum(values), 2)


# ---------------------------------------------------------------------------
# Сервис
# ---------------------------------------------------------------------------


def test_health_and_meta(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    m = client.get("/api/v1/meta").json()
    assert m["date_min"] <= m["date_max"]
    assert "reference" in m["sources"]
    assert m["n_bs_chips"] > 0 and m["n_af_scenes"] > 0
    assert [s["severity"] for s in m["severity"]] == [1, 2, 3]


def test_ui_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert client.get("/static/vendor/maplibre/maplibre-gl.js").status_code == 200
    assert client.get("/api/v1/aoi").json()["type"] == "FeatureCollection"


# ---------------------------------------------------------------------------
# Запросы: bbox, полигон, даты
# ---------------------------------------------------------------------------


def test_query_bbox(client):
    body = {"bbox": BBOX, "date_from": "2019-01-01", "date_to": "2024-12-31", "source": "reference"}
    d = _post(client, "/api/v1/query", body).json()
    assert d["query"]["spatial_filter"] == "bbox"
    for f in d["hotspots"]["features"]:
        lon, lat = f["geometry"]["coordinates"]
        assert BBOX[0] <= lon <= BBOX[2] and BBOX[1] <= lat <= BBOX[3]
    box = shapely.box(*BBOX).buffer(1e-6)
    for f in d["burn_scars"]["features"]:
        assert box.contains(shape(f["geometry"]))  # контуры обрезаны прямоугольником
    rep = d["report"]
    assert rep["summary"]["n_scars"] == len(d["burn_scars"]["features"])
    assert rep["summary"]["n_hotspots"] == len(d["hotspots"]["features"])


def test_query_polygon(client):
    tri = {"type": "Polygon", "coordinates": [[[42.0, 46.0], [47.5, 47.0], [44.0, 51.5], [42.0, 46.0]]]}
    body = {"geometry": tri, "source": "reference"}
    d = _post(client, "/api/v1/query", body).json()
    poly = shape(tri)
    assert d["query"]["spatial_filter"] == "polygon"
    assert len(d["hotspots"]["features"]) > 0
    for f in d["hotspots"]["features"]:
        assert poly.buffer(1e-9).contains(shapely.Point(f["geometry"]["coordinates"]))
    # Feature и FeatureCollection тоже принимаются
    fc = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": tri}]}
    d2 = _post(client, "/api/v1/query", {"geometry": fc, "source": "reference", "include": ["report"]}).json()
    assert d2["report"]["summary"] == d["report"]["summary"]


def test_date_filter(client):
    body = {"date_from": "2021-06-01", "date_to": "2021-08-31", "source": "reference"}
    d = _post(client, "/api/v1/query", body).json()
    for f in d["hotspots"]["features"]:
        assert "2021-06-01" <= f["properties"]["acq_date"] <= "2021-08-31"
    assert d["burn_scars"]["features"], "в период должны попасть гари"
    for f in d["burn_scars"]["features"]:
        p = f["properties"]
        assert p["date_pre"] <= "2021-08-31" and p["date_post"] >= "2021-06-01"
    # весь период содержит больше объектов, чем лето 2021
    full = _post(client, "/api/v1/query", {"source": "reference", "include": ["report"]}).json()["report"]["summary"]
    assert full["n_hotspots"] > d["report"]["summary"]["n_hotspots"]


def test_get_endpoints_bbox(client):
    r = client.get("/api/v1/hotspots", params={"bbox": ",".join(map(str, BBOX)), "source": "reference"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/geo+json")
    fc = r.json()
    rep = client.get("/api/v1/report", params={"bbox": ",".join(map(str, BBOX)), "source": "reference"}).json()
    assert len(fc["features"]) == rep["summary"]["n_hotspots"]


@pytest.mark.parametrize(
    "body,frag",
    [
        ({"date_from": "2022-01-01", "date_to": "2021-01-01"}, "позже"),
        ({"date_from": "2022-13-01"}, "date_from"),
        ({"bbox": [46, 47, 43, 50]}, "minLon < maxLon"),
        ({"bbox": [1, 2, 3]}, "bbox"),
        ({"bbox": [43, 47, 46, 50], "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}}, "либо"),
        ({"geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 1], [1, 0], [0, 1], [0, 0]]]}}, "некорректный"),
        ({"geometry": {"type": "Point", "coordinates": [44, 48]}}, "Polygon"),
        ({"geometry": {"type": "Polygon", "coordinates": [[[400, 0], [401, 0], [401, 1], [400, 0]]]}}, "EPSG:4326"),
        ({"source": "satellite"}, "source"),
    ],
)
def test_validation_errors(client, body, frag):
    r = client.post("/api/v1/query", json=body)
    assert r.status_code == 422, r.text
    assert frag in r.json()["message"], r.json()


def test_validation_errors_get(client):
    r = client.get("/api/v1/report", params={"bbox": "43,47,46"})
    assert r.status_code == 422 and "bbox" in r.json()["message"]
    r = client.get("/api/v1/report", params={"date_from": "2024-01-01", "date_to": "2023-01-01"})
    assert r.status_code == 422


def test_query_get_matches_post(client):
    """GET /api/v1/query даёт тот же результат, что и POST."""
    params = {"bbox": ",".join(map(str, BBOX)), "date_from": "2021-01-01", "date_to": "2021-12-31", "source": "reference"}
    g = client.get("/api/v1/query", params=params)
    assert g.status_code == 200, g.text
    p = _post(client, "/api/v1/query", {"bbox": BBOX, "date_from": "2021-01-01", "date_to": "2021-12-31", "source": "reference"})
    assert g.json()["report"]["summary"] == p.json()["report"]["summary"]
    assert len(g.json()["hotspots"]["features"]) == len(p.json()["hotspots"]["features"])


def test_huge_polygon_covers_whole_catalog(client):
    """Полигон «весь мир» нельзя перепроецировать в UTM напрямую — запрос обрезается охватом данных."""
    world = {"type": "Polygon", "coordinates": [[[-179, -89], [179, -89], [179, 89], [-179, 89], [-179, -89]]]}
    whole = _post(client, "/api/v1/report", {"source": "reference"}).json()["summary"]
    huge = _post(client, "/api/v1/report", {"geometry": world, "source": "reference"}).json()["summary"]
    assert huge["burned_area_ha"] == whole["burned_area_ha"] > 0
    assert huge["n_scars"] == whole["n_scars"] and huge["n_hotspots"] == whole["n_hotspots"]
    assert huge["n_scars_clipped"] == 0


# ---------------------------------------------------------------------------
# Справка: согласованность площадей
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source", ["reference", "model"])
@pytest.mark.parametrize("spatial", [None, BBOX])
def test_report_sums(client, cat, source, spatial):
    if source not in cat.sources:
        pytest.skip("нет источника model")
    body = {"source": source}
    if spatial:
        body["bbox"] = spatial
    rep = _post(client, "/api/v1/report", body).json()
    tot = rep["summary"]["burned_area_ha"]
    assert _sum(r["area_ha"] for r in rep["severity"]) == tot
    assert _sum(r["area_ha"] for r in rep["scars"]) == tot
    assert _sum(r["area_ha"] for r in rep["landcover"]) == tot
    for r in rep["scars"]:
        assert _sum([r["area_low_ha"], r["area_moderate_ha"], r["area_high_ha"]]) == r["area_ha"]
        assert r["area_ha"] <= r["area_total_ha"] + 1e-9
    if tot > 0:
        assert round(sum(r["share_pct"] for r in rep["severity"]), 2) == 100.0
    assert rep["summary"]["n_hotspots"] == rep["summary"]["n_hotspots_day"] + rep["summary"]["n_hotspots_night"]


def test_reference_total_equals_meta(client, cat):
    """Без пространственного фильтра эталонная площадь = сумма burn_area_ha из meta.csv."""
    rep = _post(client, "/api/v1/report", {"source": "reference", "compare": False}).json()
    meta_total = round(sum(c["burn_area_ha_meta"] for c in cat.bs_chips), 2)
    assert rep["summary"]["burned_area_ha"] == meta_total


def test_chip_area_equals_meta(client, cat):
    """Площадь всех контуров чипа (эталон) = burn_area_ha из meta.csv = (sev1+sev2+sev3) * 0.04 га."""
    chips = [c for c in cat.bs_chips if (c["burn_area_ha_meta"] or 0) > 0]
    chips = sorted(chips, key=lambda c: c["burn_area_ha_meta"])
    picks = [chips[0], chips[len(chips) // 3], chips[2 * len(chips) // 3], chips[-1]]
    for c in picks:
        poly = shapely.Polygon(c["footprint"]).buffer(0.002)
        body = {
            "geometry": shapely.geometry.mapping(poly),
            "date_from": c["date_pre"],
            "date_to": c["date_post"],
            "source": "reference",
        }
        rep = _post(client, "/api/v1/report", body, compare="false").json()
        rows = [r for r in rep["scars"] if r["chip_id"] == c["chip_id"]]
        assert rows and not any(r["clipped"] for r in rows)
        assert _sum(r["area_ha"] for r in rows) == round(c["burn_area_ha_meta"], 2)
        for s, key in ((1, "area_low_ha"), (2, "area_moderate_ha"), (3, "area_high_ha")):
            assert _sum(r[key] for r in rows) == round(c["sev_px_meta"][s - 1] * 0.04, 2)


def test_clipping_synthetic(client, cat):
    """Полигон запроса = левая половина чипа (граница по линии пикселей в UTM).

    Площадь после обрезки должна совпасть с числом горевших пикселей левой половины * 0.04 га.
    """
    c = max(cat.bs_chips, key=lambda c: c["burn_area_ha_meta"] or 0)
    mask = cat.sources["reference"].mask(c["chip_id"])
    h, w = mask.shape
    x0, y0, g = c["x0"], c["y0"], c["gsd"]
    xm = x0 + (w // 2) * g
    y1 = y0 - h * g
    tr = Transformer.from_crs(c["epsg"], 4326, always_xy=True)
    # прямоугольник в UTM с запасом по вертикали, стороны уплотнены (кривые в WGS84)
    ring = shapely.segmentize(shapely.box(x0 - 2000, y1 - 2000, xm, y0 + 2000), 20)
    lon, lat = tr.transform(*np.asarray(ring.exterior.coords).T)
    poly = shapely.Polygon(np.column_stack([lon, lat]))
    body = {
        "geometry": shapely.geometry.mapping(poly),
        "date_from": c["date_pre"],
        "date_to": c["date_post"],
        "source": "reference",
    }
    rep = _post(client, "/api/v1/report", body, compare="false").json()
    rows = [r for r in rep["scars"] if r["chip_id"] == c["chip_id"]]
    got = sum(r["area_ha"] for r in rows)
    left = mask[:, : w // 2]
    expected = (left > 0).sum() * g * g / 1e4
    assert any(r["clipped"] for r in rows)
    assert abs(got - expected) < 0.01 * len(rows) + 0.05, (got, expected)
    for s in (1, 2, 3):
        key = ("area_low_ha", "area_moderate_ha", "area_high_ha")[s - 1]
        exp_s = (left == s).sum() * g * g / 1e4
        assert abs(sum(r[key] for r in rows) - exp_s) < 0.01 * len(rows) + 0.05


# ---------------------------------------------------------------------------
# Выгрузки
# ---------------------------------------------------------------------------


def test_export_geojson(client):
    body = {"bbox": BBOX, "source": "reference"}
    r = client.post("/api/v1/burn-scars", json=body, params={"format": "geojson"})
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    fc = json.loads(r.content)
    assert fc["type"] == "FeatureCollection" and "CRS84" in fc["crs"]["properties"]["name"]
    rep = _post(client, "/api/v1/report", body, compare="false").json()
    assert len(fc["features"]) == rep["summary"]["n_scars"]
    assert _sum(f["properties"]["area_ha"] for f in fc["features"]) == rep["summary"]["burned_area_ha"]
    need = {"scar_id", "area_ha", "area_low_ha", "area_moderate_ha", "area_high_ha", "dominant_severity", "utm_epsg"}
    for f in fc["features"]:
        assert need <= set(f["properties"])
        g = shape(f["geometry"])
        assert g.is_valid and not g.is_empty
    # зоны степени: сумма площадей зон = сумма площадей контуров
    rz = client.post("/api/v1/burn-scars", json=body, params={"layer": "zones"})
    zones = rz.json()["features"]
    assert {f["properties"]["severity"] for f in zones} <= {1, 2, 3}
    assert _sum(f["properties"]["area_ha"] for f in zones) == rep["summary"]["burned_area_ha"]


def _read_shp(zbytes: bytes, base: str):
    import shapefile

    z = zipfile.ZipFile(io.BytesIO(zbytes))
    names = set(z.namelist())
    for ext in ("shp", "shx", "dbf", "prj", "cpg"):
        assert f"{base}.{ext}" in names, names
    rd = shapefile.Reader(
        shp=io.BytesIO(z.read(f"{base}.shp")),
        shx=io.BytesIO(z.read(f"{base}.shx")),
        dbf=io.BytesIO(z.read(f"{base}.dbf")),
        encoding="utf-8",
    )
    return rd, z.read(f"{base}.prj").decode(), z.read(f"{base}.cpg").decode()


def test_export_shapefile_wgs84(client):
    params = {"bbox": ",".join(map(str, BBOX)), "source": "reference", "format": "shp"}
    r = client.get("/api/v1/burn-scars", params=params)
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    rd, prj, cpg = _read_shp(r.content, "burn_scars")
    assert "GCS_WGS_1984" in prj and cpg == "UTF-8"
    fields = [f[0] for f in rd.fields[1:]]
    assert all(len(f) <= 10 for f in fields)
    assert {"scar_id", "dom_sev", "area_ha", "area_low", "area_mod", "area_high"} <= set(fields)
    rep = client.get("/api/v1/report", params={"bbox": params["bbox"], "source": "reference", "compare": "false"}).json()
    assert len(rd) == rep["summary"]["n_scars"]
    recs = rd.records()
    assert _sum(rec["area_ha"] for rec in recs) == rep["summary"]["burned_area_ha"]
    assert all(rec["dom_label"] in ("слабая", "средняя", "сильная") for rec in recs)  # кириллица в UTF-8


def test_export_shapefile_utm_area(client):
    """Shapefile в UTM: площадь полигона в файле = атрибуту area_ha (геометрия точная, попиксельная)."""
    r = client.get("/api/v1/burn-scars", params={"source": "reference", "format": "shp", "crs": "utm", "date_from": "2021-01-01", "date_to": "2021-12-31"})
    assert r.status_code == 200
    z = zipfile.ZipFile(io.BytesIO(r.content))
    bases = sorted({n.rsplit(".", 1)[0] for n in z.namelist() if n.endswith(".shp")})
    assert bases and all("_utm326" in b for b in bases)
    checked = 0
    for base in bases:
        rd, prj, _ = _read_shp(r.content, base)
        assert "UTM_Zone_3" in prj and "WGS_1984" in prj
        for sr in rd.iterShapeRecords():
            geo = shape(sr.shape.__geo_interface__)
            assert abs(geo.area / 1e4 - sr.record["area_ha"]) < 0.006
            checked += 1
    assert checked > 0


def test_export_zones_shp_and_csv(client):
    body = {"bbox": BBOX, "source": "reference"}
    r = client.post("/api/v1/burn-scars", json=body, params={"layer": "zones", "format": "shp"})
    rd, prj, _ = _read_shp(r.content, "severity_zones")
    assert {"zone_id", "severity", "sev_label", "area_ha"} <= {f[0] for f in rd.fields[1:]}
    r = client.post("/api/v1/burn-scars", json=body, params={"layer": "scars", "format": "csv"})
    rows = list(csv.DictReader(io.StringIO(r.content.decode("utf-8-sig"))))
    assert rows and "scar_id" in rows[0] and "area_ha" in rows[0]


@pytest.mark.parametrize("crs", ["4326", "utm"])
def test_export_geopackage(client, tmp_path, crs):
    """GeoPackage читается GDAL (pyogrio): два слоя, верная СК, число объектов как в справке."""
    pyogrio = pytest.importorskip("pyogrio")
    params = {"bbox": ",".join(map(str, BBOX)), "source": "reference"}
    r = client.get("/api/v1/burn-scars", params={**params, "format": "gpkg", "crs": crs})
    assert r.status_code == 200
    fn = tmp_path / "out.gpkg"
    fn.write_bytes(r.content)
    layers = {name: gt for name, gt in pyogrio.list_layers(fn)}
    rep = client.get("/api/v1/report", params={**params, "compare": "false"}).json()
    n_scars = 0
    for name in layers:
        info = pyogrio.read_info(fn, layer=name)
        if crs == "4326":
            assert info["crs"] == "EPSG:4326"
        else:
            assert info["crs"] in ("EPSG:32637", "EPSG:32638")
        if name.startswith("burn_scars"):
            n_scars += info["features"]
    assert any(n.startswith("severity_zones") for n in layers)
    assert n_scars == rep["summary"]["n_scars"]


def test_hotspot_exports(client):
    params = {"source": "reference", "date_from": "2020-01-01", "date_to": "2020-12-31"}
    gj = client.get("/api/v1/hotspots", params=params).json()
    r = client.get("/api/v1/hotspots", params={**params, "format": "csv"})
    rows = list(csv.DictReader(io.StringIO(r.content.decode("utf-8-sig"))))
    assert len(rows) == len(gj["features"]) > 0
    assert {"latitude", "longitude", "acq_datetime", "satellite", "daynight", "bright_ti4", "confidence"} <= set(rows[0])
    r = client.get("/api/v1/hotspots", params={**params, "format": "shp"})
    rd, prj, _ = _read_shp(r.content, "hotspots")
    assert len(rd) == len(rows) and "GCS_WGS_1984" in prj


def test_report_formats(client):
    params = {"bbox": ",".join(map(str, BBOX))}
    r = client.get("/api/v1/report", params={**params, "format": "csv"})
    rows = list(csv.reader(io.StringIO(r.content.decode("utf-8-sig"))))
    assert rows[0] == ["section", "metric", "value", "unit"]
    kv = {row[1]: row[2] for row in rows[1:]}
    j = client.get("/api/v1/report", params=params).json()
    assert float(kv["burned_area_ha"]) == j["summary"]["burned_area_ha"]
    r = client.get("/api/v1/report", params={**params, "format": "severity_csv"})
    sev = list(csv.DictReader(io.StringIO(r.content.decode("utf-8-sig"))))
    assert [s["severity"] for s in sev] == ["1", "2", "3", "total"]
    r = client.get("/api/v1/report", params={**params, "format": "zip"})
    names = set(zipfile.ZipFile(io.BytesIO(r.content)).namelist())
    assert {"report.json", "report_summary.csv", "report_scars.csv", "report_severity.csv"} <= names


def test_comparison(client, cat):
    if "model" not in cat.sources:
        pytest.skip("нет источника model")
    rep = _post(client, "/api/v1/report", {"source": "model"}).json()
    cmp_ = rep["comparison"]
    assert cmp_ is not None
    iou = cmp_["pixel_agreement"]["iou_burned"]
    assert 0 <= iou <= 1
    assert cmp_["model"]["burned_area_ha"] == rep["summary"]["burned_area_ha"]
    ref = _post(client, "/api/v1/report", {"source": "reference", "compare": False}).json()
    assert cmp_["reference"]["burned_area_ha"] == ref["summary"]["burned_area_ha"]
    # попиксельная матрица согласована с площадями эталона (без обрезки)
    cm = np.asarray(cmp_["pixel_agreement"]["confusion_matrix_px"])
    assert round(cm[1:, :].sum() * 0.04, 2) == ref["summary"]["burned_area_ha"]


def test_scar_detail_and_chips(client):
    rep = client.get("/api/v1/report", params={"source": "reference", "bbox": ",".join(map(str, BBOX))}).json()
    sid = rep["scars"][0]["scar_id"]
    d = client.get(f"/api/v1/scars/{sid}", params={"source": "reference"}).json()
    assert d["scar"]["scar_id"] == sid and len(d["chip"]["corners"]) == 4
    assert client.get("/api/v1/scars/NOPE-R001", params={"source": "reference"}).status_code == 404
    bs = client.get("/api/v1/chips", params={"kind": "bs"}).json()
    af = client.get("/api/v1/chips", params={"kind": "af"}).json()
    assert len(bs["features"]) > 0 and len(af["features"]) > 0
    ql = bs["features"][0]["properties"]["quicklook"]
    if ql:
        assert client.get("/catalog/" + ql["post"]).status_code == 200


def test_analyze_uploaded_mask(client):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    m = np.zeros((64, 64), np.uint8)
    m[10:20, 10:30] = 1  # 200 пикселей
    m[12:16, 12:16] = 3  # 16 из них — сильная
    m[40:44, 40:42] = 2  # отдельная гарь, 8 пикселей
    buf = io.BytesIO()
    with rasterio.MemoryFile() as mf:
        with mf.open(driver="GTiff", width=64, height=64, count=1, dtype="uint8", crs="EPSG:32638", transform=from_origin(500000, 5400000, 20, 20)) as dst:
            dst.write(m, 1)
        buf.write(mf.read())
    r = client.post("/api/v1/analyze/bs", files={"mask": ("mask.tif", buf.getvalue(), "image/tiff")})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["summary"]["n_scars"] == 2
    assert d["summary"]["burned_area_ha"] == round(208 * 0.04, 2)
    sev = {s["severity"]: s["area_ha"] for s in d["summary"]["severity"]}
    assert sev == {1: round(184 * 0.04, 2), 2: round(8 * 0.04, 2), 3: round(16 * 0.04, 2)}


def test_analyze_model_unavailable_is_503_or_ok(client):
    r = client.post("/api/v1/analyze/af", files={"viirs": ("v.tif", b"not a tiff", "image/tiff")})
    assert r.status_code in (422, 503)
