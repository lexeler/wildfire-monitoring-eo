"""Выгрузки: GeoJSON, ESRI Shapefile (zip), CSV."""
from __future__ import annotations

import csv
import io
import json
import zipfile
from collections import defaultdict

import numpy as np
import shapely
from pyproj import CRS
from shapely.geometry.polygon import orient

from . import common as C
from .catalog import Catalog, Query, ScarHit, SourceStore, ha, utm_to_ll
from .report import dominant, scar_row, shares_pct

CRS84 = {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}
WGS84_PRJ = (
    'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],'
    'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]'
)
LL_DIGITS = 7  # ~1 см


def prj_wkt(epsg: int) -> str:
    if epsg == 4326:
        return WGS84_PRJ
    return CRS.from_epsg(epsg).to_wkt("WKT1_ESRI")


# ---------------------------------------------------------------------------
# Термоточки
# ---------------------------------------------------------------------------

HOTSPOT_FIELDS = [
    # (ключ в GeoJSON/CSV, имя поля DBF <= 10 символов, тип, размер, знаки)
    ("hotspot_id", "hs_id", "C", 32, 0),
    ("acq_datetime", "acq_dt", "C", 20, 0),
    ("acq_date", "acq_date", "C", 10, 0),
    ("acq_time", "acq_time", "C", 4, 0),
    ("satellite", "satellite", "C", 10, 0),
    ("daynight", "daynight", "C", 1, 0),
    ("latitude", "latitude", "N", 12, 6),
    ("longitude", "longitude", "N", 12, 6),
    ("bright_ti4", "bright_ti4", "N", 8, 2),
    ("bright_ti5", "bright_ti5", "N", 8, 2),
    ("delta_t", "delta_t", "N", 8, 2),
    ("confidence", "confidence", "N", 6, 4),
    ("landcover_code", "lc_code", "N", 4, 0),
    ("landcover", "landcover", "C", 100, 0),
    ("solar_zenith", "sol_zen", "N", 7, 2),
    ("sensor_zenith", "sen_zen", "N", 7, 2),
    ("chip_id", "chip_id", "C", 24, 0),
    ("utm_epsg", "utm_epsg", "N", 6, 0),
    ("source", "source", "C", 10, 0),
]


def hotspot_props(st: SourceStore, i: int) -> dict:
    h = st.hs
    t = h["t"][i]
    i4, i5 = h["i4"][i], h["i5"][i]
    return {
        "hotspot_id": h["id"][i],
        "acq_datetime": t,
        "acq_date": t[:10],
        "acq_time": t[11:13] + t[14:16],
        "satellite": h["sat"][i],
        "daynight": h["dn"][i],
        "latitude": h["lat"][i],
        "longitude": h["lon"][i],
        "bright_ti4": i4,
        "bright_ti5": i5,
        "delta_t": None if i4 is None or i5 is None else round(i4 - i5, 2),
        "confidence": h["conf"][i],
        "landcover_code": h["lc"][i],
        "landcover": C.landcover_name(h["lc"][i]),
        "solar_zenith": h["sza"][i],
        "sensor_zenith": h["vza"][i],
        "chip_id": h["chip"][i],
        "utm_epsg": h["epsg"][i],
        "source": st.name,
    }


def hotspots_fc(st: SourceStore, idx: np.ndarray, name="hotspots") -> dict:
    feats = []
    for i in idx.tolist():
        p = hotspot_props(st, i)
        feats.append(
            {
                "type": "Feature",
                "id": p["hotspot_id"],
                "geometry": {"type": "Point", "coordinates": [p["longitude"], p["latitude"]]},
                "properties": p,
            }
        )
    return {"type": "FeatureCollection", "name": name, "crs": CRS84, "features": feats}


# ---------------------------------------------------------------------------
# Контуры гарей и зоны степени
# ---------------------------------------------------------------------------

SCAR_FIELDS = [
    ("scar_id", "scar_id", "C", 32, 0),
    ("chip_id", "chip_id", "C", 24, 0),
    ("fire_event_id", "event_id", "C", 16, 0),
    ("date_pre", "date_pre", "C", 10, 0),
    ("date_post", "date_post", "C", 10, 0),
    ("area_ha", "area_ha", "N", 14, 2),
    ("area_low_ha", "area_low", "N", 14, 2),
    ("area_moderate_ha", "area_mod", "N", 14, 2),
    ("area_high_ha", "area_high", "N", 14, 2),
    ("share_low_pct", "shr_low", "N", 7, 2),
    ("share_moderate_pct", "shr_mod", "N", 7, 2),
    ("share_high_pct", "shr_high", "N", 7, 2),
    ("dominant_severity", "dom_sev", "N", 1, 0),
    ("dominant_label", "dom_label", "C", 20, 0),
    ("clipped", "clipped", "N", 1, 0),
    ("area_total_ha", "area_full", "N", 14, 2),
    ("n_pixels", "n_pixels", "N", 10, 0),
    ("pixel_size_m", "pix_m", "N", 6, 1),
    ("utm_epsg", "utm_epsg", "N", 6, 0),
    ("center_lon", "cen_lon", "N", 12, 6),
    ("center_lat", "cen_lat", "N", 12, 6),
    ("confidence", "confidence", "N", 6, 4),
    ("source", "source", "C", 10, 0),
]

ZONE_FIELDS = [
    ("zone_id", "zone_id", "C", 36, 0),
    ("scar_id", "scar_id", "C", 32, 0),
    ("severity", "severity", "N", 1, 0),
    ("severity_label", "sev_label", "C", 20, 0),
    ("area_ha", "area_ha", "N", 14, 2),
    ("share_of_scar_pct", "shr_scar", "N", 7, 2),
    ("n_pixels", "n_pixels", "N", 10, 0),
    ("chip_id", "chip_id", "C", 24, 0),
    ("fire_event_id", "event_id", "C", 16, 0),
    ("date_pre", "date_pre", "C", 10, 0),
    ("date_post", "date_post", "C", 10, 0),
    ("clipped", "clipped", "N", 1, 0),
    ("utm_epsg", "utm_epsg", "N", 6, 0),
    ("source", "source", "C", 10, 0),
]


def scar_props(cat: Catalog, st: SourceStore, h: ScarHit) -> dict:
    r = scar_row(cat, st, h)
    return {k: r[k] for k, *_ in SCAR_FIELDS}


def zone_items(cat: Catalog, st: SourceStore, q: Query, h: ScarHit, exact: bool = True):
    """(properties, геометрия UTM) для зон степени контура с учётом обрезки."""
    s = st.scars[h.idx]
    chip = cat.bs_chips[st.scar_chip[h.idx]]
    sh = shares_pct(h.sev_units)
    for zi in st.scar_zones[h.idx]:
        z = st.zones[zi]
        sev = z["severity"]
        if h.clipped:
            g = (h.zone_geoms or {}).get(sev)
            if g is None or h.sev_units[sev - 1] <= 0:
                continue
        else:
            g = st.zone_exact[zi]
        if not exact:
            g = shapely.simplify(g, st.scar_gsd[h.idx] * 1.0, preserve_topology=True)
        props = {
            "zone_id": f"{s['scar_id']}-Z{sev}",
            "scar_id": s["scar_id"],
            "severity": sev,
            "severity_label": C.SEVERITY[sev]["label"],
            "area_ha": ha(h.sev_units[sev - 1]),
            "share_of_scar_pct": sh[sev - 1],
            "n_pixels": int(z["n_px"]),
            "chip_id": chip["chip_id"],
            "fire_event_id": chip["fire_event_id"],
            "date_pre": chip["date_pre"],
            "date_post": chip["date_post"],
            "clipped": bool(h.clipped),
            "utm_epsg": int(st.scar_epsg[h.idx]),
            "source": st.name,
        }
        yield props, g


def layer_items(cat: Catalog, st: SourceStore, q: Query, hits: list[ScarHit], layer: str, exact: bool = True):
    """Список (properties, геометрия UTM, epsg) для слоя scars|zones."""
    items = []
    for h in hits:
        epsg = int(st.scar_epsg[h.idx])
        if layer == "zones":
            for p, g in zone_items(cat, st, q, h, exact):
                items.append((p, g, epsg))
        else:
            g = st.clipped_scar_geom(h, q, exact=exact)
            if g is None:
                continue
            items.append((scar_props(cat, st, h), g, epsg))
    return items


def to_ll_items(items):
    """Перевод геометрии UTM -> WGS84 пакетно по зонам."""
    by_epsg = defaultdict(list)
    for k, (_, g, e) in enumerate(items):
        by_epsg[e].append(k)
    out = [None] * len(items)
    for e, ks in by_epsg.items():
        ll = utm_to_ll(np.asarray([items[k][1] for k in ks], dtype=object), e, LL_DIGITS)
        for k, g in zip(ks, ll):
            out[k] = g
    return out


def polygons_fc(items, name: str) -> dict:
    lls = to_ll_items(items)
    if lls:  # RFC 7946: внешние кольца против часовой стрелки, дырки — по часовой
        lls = list(shapely.orient_polygons(np.asarray(lls, dtype=object), exterior_cw=False))
    feats = []
    for (p, _, _), g in zip(items, lls):
        fid = p.get("zone_id") or p.get("scar_id")
        feats.append({"type": "Feature", "id": fid, "geometry": shapely.geometry.mapping(g), "properties": p})
    return {"type": "FeatureCollection", "name": name, "crs": CRS84, "features": feats}


# ---------------------------------------------------------------------------
# Shapefile
# ---------------------------------------------------------------------------


def _dbf_value(v, ftype):
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    return v


def _poly_parts(g) -> list:
    """Кольца для shapefile: внешние — по часовой стрелке, дырки — против."""
    parts = []
    for p in shapely.get_parts(g):
        if p.is_empty:
            continue
        p = orient(p, sign=-1.0)
        parts.append([list(c) for c in p.exterior.coords])
        for r in p.interiors:
            parts.append([list(c) for c in r.coords])
    return parts


def _write_shp(zf: zipfile.ZipFile, base: str, shape_type: int, fields, rows, geoms, epsg: int):
    import shapefile

    shp, shx, dbf = io.BytesIO(), io.BytesIO(), io.BytesIO()
    w = shapefile.Writer(shp=shp, shx=shx, dbf=dbf, shapeType=shape_type, encoding="utf-8")
    for _, fname, ftype, size, dec in fields:
        w.field(fname, ftype, size, dec)
    for props, g in zip(rows, geoms):
        if shape_type == shapefile.POINT:
            w.point(*g)
        else:
            parts = _poly_parts(g)
            if not parts:
                w.null()
            else:
                w.poly(parts)
        w.record(*[_dbf_value(props.get(k), t) for k, _, t, *_ in fields])
    w.close()
    zf.writestr(f"{base}.shp", shp.getvalue())
    zf.writestr(f"{base}.shx", shx.getvalue())
    zf.writestr(f"{base}.dbf", dbf.getvalue())
    zf.writestr(f"{base}.prj", prj_wkt(epsg))
    zf.writestr(f"{base}.cpg", "UTF-8")


def _fields_readme(fields, title: str) -> str:
    lines = [f"{title}", "Поле DBF   <- атрибут GeoJSON/CSV"]
    for key, fname, *_ in fields:
        lines.append(f"  {fname:<10} <- {key}")
    return "\n".join(lines)


def polygons_shapefile_zip(items, base: str, fields, crs: str = "4326") -> bytes:
    import shapefile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if crs == "utm":
            by = defaultdict(list)
            for p, g, e in items:
                by[e].append((p, g))
            for e, lst in sorted(by.items()):
                _write_shp(zf, f"{base}_utm{e}", shapefile.POLYGON, fields, [p for p, _ in lst], [g for _, g in lst], e)
            note = "Система координат: UTM, отдельный слой на каждую зону (EPSG в имени файла)."
        else:
            lls = to_ll_items(items)
            _write_shp(zf, base, shapefile.POLYGON, fields, [p for p, _, _ in items], lls, 4326)
            note = "Система координат: WGS84 (EPSG:4326). Площади (area_*) посчитаны в UTM зоны чипа."
        zf.writestr("README.txt", note + "\nКодировка атрибутов: UTF-8 (.cpg).\n\n" + _fields_readme(fields, base))
    return buf.getvalue()


def hotspots_shapefile_zip(st: SourceStore, idx: np.ndarray, base="hotspots") -> bytes:
    import shapefile

    rows = [hotspot_props(st, i) for i in idx.tolist()]
    pts = [(r["longitude"], r["latitude"]) for r in rows]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        _write_shp(zf, base, shapefile.POINT, HOTSPOT_FIELDS, rows, pts, 4326)
        zf.writestr("README.txt", "Система координат: WGS84 (EPSG:4326).\nКодировка: UTF-8.\n\n" + _fields_readme(HOTSPOT_FIELDS, base))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# GeoPackage (OGC GeoPackage 1.2, пишется напрямую через sqlite3 — без GDAL)
# ---------------------------------------------------------------------------

_GPKG_TYPES = {"C": "TEXT", "N": "DOUBLE"}


def _gpkg_blob(g, srs_id: int) -> bytes:
    import struct

    if g is None or g.is_empty:
        return b"GP" + bytes([0, 0x11]) + struct.pack("<i", srs_id) + shapely.to_wkb(g, byte_order=1)
    minx, miny, maxx, maxy = g.bounds
    head = b"GP" + bytes([0, 0x03]) + struct.pack("<i", srs_id) + struct.pack("<4d", minx, maxx, miny, maxy)
    return head + shapely.to_wkb(g, byte_order=1, flavor="iso")


def gpkg_bytes(layers: list[tuple]) -> bytes:
    """layers: [(имя таблицы, epsg, поля [(ключ, тип C|N)], rows [dict], geoms, тип геометрии)]."""
    import sqlite3

    con = sqlite3.connect(":memory:")
    cur = con.cursor()
    cur.execute("PRAGMA application_id = 1196444487")  # 'GPKG'
    cur.execute("PRAGMA user_version = 10200")
    cur.executescript(
        """
        CREATE TABLE gpkg_spatial_ref_sys (srs_name TEXT NOT NULL, srs_id INTEGER PRIMARY KEY,
          organization TEXT NOT NULL, organization_coordsys_id INTEGER NOT NULL, definition TEXT NOT NULL, description TEXT);
        CREATE TABLE gpkg_contents (table_name TEXT NOT NULL PRIMARY KEY, data_type TEXT NOT NULL, identifier TEXT UNIQUE,
          description TEXT DEFAULT '', last_change DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
          min_x DOUBLE, min_y DOUBLE, max_x DOUBLE, max_y DOUBLE, srs_id INTEGER,
          CONSTRAINT fk_gc_r_srs_id FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys(srs_id));
        CREATE TABLE gpkg_geometry_columns (table_name TEXT NOT NULL, column_name TEXT NOT NULL, geometry_type_name TEXT NOT NULL,
          srs_id INTEGER NOT NULL, z TINYINT NOT NULL, m TINYINT NOT NULL,
          CONSTRAINT pk_geom_cols PRIMARY KEY (table_name, column_name),
          CONSTRAINT fk_gc_tn FOREIGN KEY (table_name) REFERENCES gpkg_contents(table_name),
          CONSTRAINT fk_gc_srs FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys (srs_id));
        """
    )
    cur.executemany(
        "INSERT INTO gpkg_spatial_ref_sys VALUES (?,?,?,?,?,?)",
        [
            ("Undefined cartesian SRS", -1, "NONE", -1, "undefined", "undefined cartesian coordinate reference system"),
            ("Undefined geographic SRS", 0, "NONE", 0, "undefined", "undefined geographic coordinate reference system"),
        ],
    )
    srs_done = set()
    for name, epsg, fields, rows, geoms, gtype in layers:
        if epsg not in srs_done:
            crs = CRS.from_epsg(epsg)
            cur.execute(
                "INSERT INTO gpkg_spatial_ref_sys VALUES (?,?,?,?,?,?)",
                (crs.name, epsg, "EPSG", epsg, crs.to_wkt("WKT1_GDAL"), crs.name),
            )
            srs_done.add(epsg)
        cols = ", ".join(f'"{k}" {_GPKG_TYPES[t]}' for k, t in fields)
        cur.execute(f'CREATE TABLE "{name}" (fid INTEGER PRIMARY KEY AUTOINCREMENT, geom {gtype}, {cols})')
        b = shapely.bounds(np.asarray(geoms, dtype=object)) if len(geoms) else np.full((1, 4), np.nan)
        env = [float(np.nanmin(b[:, 0])), float(np.nanmin(b[:, 1])), float(np.nanmax(b[:, 2])), float(np.nanmax(b[:, 3]))] if len(geoms) else [None] * 4
        cur.execute(
            "INSERT INTO gpkg_contents (table_name, data_type, identifier, description, min_x, min_y, max_x, max_y, srs_id) VALUES (?,?,?,?,?,?,?,?,?)",
            (name, "features", name, "", *env, epsg),
        )
        cur.execute("INSERT INTO gpkg_geometry_columns VALUES (?,?,?,?,0,0)", (name, "geom", gtype, epsg))
        ph = ", ".join("?" for _ in range(len(fields) + 1))
        colnames = ", ".join(f'"{k}"' for k, _ in fields)
        cur.executemany(
            f'INSERT INTO "{name}" (geom, {colnames}) VALUES ({ph})',
            [
                (_gpkg_blob(g, epsg), *[_dbf_value(r.get(k), t) for k, t in fields])
                for r, g in zip(rows, geoms)
            ],
        )
    con.commit()
    data = con.serialize()
    con.close()
    return bytes(data)


def polygons_gpkg(scar_items, zone_items_, crs: str = "4326") -> bytes:
    """Оба слоя (контуры и зоны степени) в одном GeoPackage: WGS84 или по слою на зону UTM."""
    layers = []
    for base, items, fields in (("burn_scars", scar_items, SCAR_FIELDS), ("severity_zones", zone_items_, ZONE_FIELDS)):
        f2 = [(k, "N" if t == "N" else "C") for k, _, t, *_ in fields]
        if crs == "utm":
            by = defaultdict(list)
            for p, g, e in items:
                by[e].append((p, g))
            for e, lst in sorted(by.items()):
                layers.append((f"{base}_utm{e}", e, f2, [p for p, _ in lst], [C.to_multipolygon(g) for _, g in lst], "MULTIPOLYGON"))
        else:
            lls = to_ll_items(items)
            layers.append((base, 4326, f2, [p for p, _, _ in items], [C.to_multipolygon(g) for g in lls], "MULTIPOLYGON"))
    return gpkg_bytes(layers)


def hotspots_gpkg(st: SourceStore, idx: np.ndarray) -> bytes:
    rows = [hotspot_props(st, i) for i in idx.tolist()]
    pts = [shapely.Point(r["longitude"], r["latitude"]) for r in rows]
    f2 = [(k, "N" if t == "N" else "C") for k, _, t, *_ in HOTSPOT_FIELDS]
    return gpkg_bytes([("hotspots", 4326, f2, rows, pts, "POINT")])


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        return repr(round(v, 6)) if v == v else ""
    if isinstance(v, (list, tuple)):
        return " ".join(_fmt(x) for x in v)
    return str(v)


def csv_bytes(header: list[str], rows: list[list]) -> bytes:
    """CSV (RFC 4180): разделитель «,», десятичная точка, UTF-8 с BOM (для корректной кириллицы в Excel)."""
    s = io.StringIO()
    w = csv.writer(s, lineterminator="\r\n")
    w.writerow(header)
    for r in rows:
        w.writerow([_fmt(v) for v in r])
    return ("﻿" + s.getvalue()).encode("utf-8")


def dicts_csv(rows: list[dict], columns: list[str]) -> bytes:
    return csv_bytes(columns, [[r.get(c) for c in columns] for r in rows])


def hotspots_csv(st: SourceStore, idx: np.ndarray) -> bytes:
    cols = [k for k, *_ in HOTSPOT_FIELDS]
    return dicts_csv([hotspot_props(st, i) for i in idx.tolist()], cols)


def items_csv(items, fields) -> bytes:
    cols = [k for k, *_ in fields]
    return dicts_csv([p for p, _, _ in items], cols)


SCAR_CSV_COLUMNS = [k for k, *_ in SCAR_FIELDS]


def report_summary_rows(rep: dict) -> list[list]:
    q, s = rep["query"], rep["summary"]
    rows = [
        ["query", "spatial_filter", q["spatial_filter"], ""],
        ["query", "bbox", q["bbox"], "deg"],
        ["query", "area_km2", q["area_km2"], "km2"],
        ["query", "date_from", q["date_from"], ""],
        ["query", "date_to", q["date_to"], ""],
        ["query", "source", q["source"], ""],
        ["query", "clip", q["clip"], ""],
        ["summary", "burned_area_ha", s["burned_area_ha"], "ha"],
    ]
    for r in rep["severity"]:
        rows.append(["severity", f"area_{r['key']}_ha", r["area_ha"], "ha"])
    for r in rep["severity"]:
        rows.append(["severity", f"share_{r['key']}_pct", r["share_pct"], "%"])
    rows += [
        ["summary", "n_scars", s["n_scars"], "count"],
        ["summary", "n_scars_clipped", s["n_scars_clipped"], "count"],
        ["summary", "burned_area_unclipped_ha", s["burned_area_unclipped_ha"], "ha"],
        ["summary", "burned_share_of_query_pct", s["burned_share_of_query_pct"], "%"],
        ["summary", "n_hotspots", s["n_hotspots"], "count"],
        ["summary", "n_hotspots_day", s["n_hotspots_day"], "count"],
        ["summary", "n_hotspots_night", s["n_hotspots_night"], "count"],
        ["summary", "n_bs_chips", s["n_bs_chips"], "count"],
        ["summary", "n_af_scenes", s["n_af_scenes"], "count"],
    ]
    cmp_ = rep.get("comparison")
    if cmp_:
        for src in ("model", "reference"):
            rows.append(["comparison", f"{src}_burned_area_ha", cmp_[src]["burned_area_ha"], "ha"])
        pa = cmp_["pixel_agreement"]
        for k in ("iou_burned", "precision", "recall", "f1", "severity_accuracy_on_common"):
            rows.append(["comparison", k, pa[k], ""])
        ha_ = cmp_["hotspot_agreement"]
        rows.append(["comparison", "hotspots_f1", ha_["f1"], ""])
    rows += [
        ["meta", "generated_at", rep["generated_at"], ""],
        ["meta", "catalog_generated_at", rep["service"]["catalog_generated_at"], ""],
    ]
    return rows


def report_csv(rep: dict) -> bytes:
    return csv_bytes(["section", "metric", "value", "unit"], report_summary_rows(rep))


def severity_csv(rep: dict) -> bytes:
    rows = [[r["severity"], r["key"], r["label"], r["area_ha"], r["share_pct"], r["n_zones"], r["n_scars_dominant"]] for r in rep["severity"]]
    rows.append(["total", "total", "итого", rep["summary"]["burned_area_ha"], 100.0 if rep["summary"]["burned_area_ha"] else 0.0, sum(r["n_zones"] for r in rep["severity"]), rep["summary"]["n_scars"]])
    return csv_bytes(["severity", "key", "label", "area_ha", "share_pct", "n_zones", "n_scars_dominant"], rows)


def scars_csv(rep: dict) -> bytes:
    return dicts_csv(rep["scars"], SCAR_CSV_COLUMNS)


def landcover_csv(rep: dict) -> bytes:
    cols = ["code", "name", "area_ha", "share_pct", "area_low_ha", "area_moderate_ha", "area_high_ha"]
    return dicts_csv(rep["landcover"], cols)


def hotspots_by_date_csv(rep: dict) -> bytes:
    return dicts_csv(rep["hotspots"]["by_date"], ["date", "count", "day", "night"])


REPORT_README = """Аналитическая справка — пакет выгрузки
======================================
report.json            полная справка (машиночитаемая)
report_summary.csv     сводные показатели: section, metric, value, unit
report_severity.csv    площадь гарей по классам степени поражения (+ строка total)
report_scars.csv       контуры гарей: площади по классам, доли, даты, EPSG зоны UTM
report_landcover.csv   площадь гарей по типам покрова ESA WorldCover
report_hotspots_by_date.csv  число термоточек по датам (день/ночь)

CSV: RFC 4180, разделитель «,», десятичная точка, UTF-8 с BOM.
Площади — в гектарах, посчитаны в проекции UTM зоны чипа (пиксель 20 м = 0.04 га).
Сумма area_ha по контурам = итог burned_area_ha; сумма по классам = итог.
"""


def report_zip(rep: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("report.json", json.dumps(rep, ensure_ascii=False, indent=2))
        zf.writestr("report_summary.csv", report_csv(rep))
        zf.writestr("report_severity.csv", severity_csv(rep))
        zf.writestr("report_scars.csv", scars_csv(rep))
        zf.writestr("report_landcover.csv", landcover_csv(rep))
        zf.writestr("report_hotspots_by_date.csv", hotspots_by_date_csv(rep))
        zf.writestr("README.txt", REPORT_README)
    return buf.getvalue()
