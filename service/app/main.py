"""REST API и веб-интерфейс сервиса мониторинга природных пожаров.

Запуск (из корня репозитория):
    uvicorn service.app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
import time
from datetime import date
from pathlib import Path
from typing import Literal, Optional

import orjson
import shapely
from fastapi import Body, FastAPI, File, HTTPException, Query as Q, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import common as C
from . import exporters as X
from .catalog import Catalog, Query
from .report import SERVICE_VERSION, build_report
from .schemas import FullQueryRequest, Include, QueryRequest, QueryResponse, Report

SERVICE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("FIREMON_DATA_DIR", SERVICE_DIR / "data"))
STATIC_DIR = SERVICE_DIR / "static"

_catalog: Catalog | None = None
_lock = threading.Lock()


def get_catalog() -> Catalog:
    global _catalog
    if _catalog is None:
        with _lock:
            if _catalog is None:
                _catalog = Catalog(DATA_DIR)
    return _catalog


TAGS = [
    {"name": "Сервис", "description": "Состояние сервиса и метаданные каталога"},
    {"name": "Запрос", "description": "Пространственно-временной запрос: термоточки, контуры гарей, справка"},
    {"name": "Выгрузка", "description": "Экспорт в GeoJSON / Shapefile / GeoPackage / CSV (GET — по bbox или geometry, POST — по полигону в теле)"},
    {"name": "Справочные слои", "description": "Граница территории, покрытие снимками, карточка контура"},
    {"name": "Анализ", "description": "Необязательная обработка нового чипа (инференс модели по GeoTIFF)"},
]

@asynccontextmanager
async def lifespan(_app):
    # каталог загружается при старте, чтобы первый запрос не ждал чтения данных (~1 с)
    if os.environ.get("FIREMON_LAZY_LOAD") != "1":
        get_catalog()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Мониторинг природных пожаров — API",
    version=SERVICE_VERSION,
    description=(
        "Информационно-аналитический сервис команды LitWin (КосмоХакатон 2026, кейс «Мониторинг природных пожаров»).\n\n"
        "Принимает пространственно-временной запрос — полигон (GeoJSON, EPSG:4326) **или** прямоугольник "
        "`bbox=minLon,minLat,maxLon,maxLat` и интервал дат — и возвращает термоточки (VIIRS 375 м), контуры "
        "гарей и зоны степени поражения (Sentinel-2 20 м) и аналитическую справку с площадями в гектарах.\n\n"
        "* Площади считаются в проекции UTM чипа (EPSG:32637/32638): пиксель 20x20 м = 0.04 га; при обрезке "
        "полигоном запроса — площадь пересечения точной попиксельной геометрии в UTM.\n"
        "* Геометрия в ответах — WGS84 (EPSG:4326).\n"
        "* `source=model` — результаты модели, `source=reference` — эталонная разметка."
    ),
    openapi_tags=TAGS,
)
app.add_middleware(GZipMiddleware, minimum_size=2048, compresslevel=5)


def jresp(obj, status_code: int = 200, headers: dict | None = None, media_type: str = "application/json") -> Response:
    return Response(
        orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS),
        status_code=status_code,
        media_type=media_type,
        headers=headers,
    )


# Типовые ошибки pydantic -> понятное сообщение на русском (ключ — тип ошибки или его префикс)
_RU_ERRORS = {
    "date_": "ожидается дата в формате ГГГГ-ММ-ДД (например, 2021-04-01)",
    "float_": "ожидается число",
    "int_": "ожидается целое число",
    "bool_": "ожидается true или false",
    "dict_type": "ожидается объект JSON (для geometry — GeoJSON Polygon / MultiPolygon / Feature / FeatureCollection)",
    "list_type": "ожидается список",
    "string_type": "ожидается строка",
    "extra_forbidden": "неизвестный параметр",
}


def _msg(e: dict) -> str:
    m = str(e.get("msg", ""))
    for pfx in ("Value error, ", "Assertion failed, "):
        if m.startswith(pfx):
            m = m[len(pfx) :]
    typ = str(e.get("type", ""))
    raw_loc = [x for x in e.get("loc", ()) if x not in ("body", "query", "path")]
    if typ == "json_invalid":
        return "тело запроса — некорректный JSON" + (f" (ошибка около символа {raw_loc[0]})" if raw_loc else "")
    if typ == "missing":
        m = "обязательный параметр не задан" if raw_loc else "нужно тело запроса в формате JSON (можно пустой объект {})"
    elif typ == "literal_error":
        m = "допустимые значения: " + str((e.get("ctx") or {}).get("expected", "")).replace(" or ", ", ").replace("'", "")
    else:
        m = next((ru for key, ru in _RU_ERRORS.items() if typ == key or (key.endswith("_") and typ.startswith(key))), m)
    loc = ".".join(str(x) for x in raw_loc)
    return f"{loc}: {m}" if loc and not m.startswith(loc.split(".")[-1]) else m


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError):
    errs = exc.errors()
    return jresp(
        {
            "message": "Некорректные параметры запроса: " + "; ".join(_msg(e) for e in errs[:5]),
            "detail": [{"loc": list(e.get("loc", ())), "msg": _msg(e), "type": e.get("type")} for e in errs],
        },
        status_code=422,
    )


def _bad(msg: str, code: int = 422):
    raise HTTPException(status_code=code, detail=msg)


_HTTP_RU = {404: "адрес не найден; список эндпоинтов — в /docs", 405: "метод не поддерживается для этого адреса; см. /docs"}


@app.exception_handler(StarletteHTTPException)  # базовый класс: сюда же попадают 404/405 маршрутизатора
async def _http_handler(request: Request, exc: StarletteHTTPException):
    detail = exc.detail
    if detail in ("Not Found", "Method Not Allowed"):
        detail = _HTTP_RU.get(exc.status_code, detail)
    return jresp({"message": detail, "detail": detail}, status_code=exc.status_code, headers=getattr(exc, "headers", None))


# ---------------------------------------------------------------------------
# Разбор запроса
# ---------------------------------------------------------------------------


BBOX_DESC = "Прямоугольник minLon,minLat,maxLon,maxLat (EPSG:4326)"
GEOM_DESC = "Полигон GeoJSON (или WKT) в EPSG:4326 — альтернатива bbox"


def to_query(req: QueryRequest) -> Query:
    cat = get_catalog()
    src = req.source or cat.default_source
    if src not in cat.sources:
        _bad(f"источник «{src}» отсутствует в каталоге; доступны: {', '.join(cat.sources)}")
    df = req.date_from or date.fromisoformat(cat.date_min)
    dt = req.date_to or date.fromisoformat(cat.date_max)
    if df > dt:
        _bad(f"date_from ({df}) позже date_to ({dt})")
    geom = req._geom
    spatial = "bbox" if req.bbox is not None else ("polygon" if geom is not None else "none")
    bbox = [round(v, 6) for v in shapely.bounds(geom).tolist()] if geom is not None else None
    return Query(
        geom=geom, spatial=spatial, date_from=df, date_to=dt, source=src, clip=req.clip, bbox=bbox, data_bounds=cat.data_bounds
    )


def query_from_params(bbox, geometry, date_from, date_to, source, clip=True) -> Query:
    from .schemas import parse_bbox, parse_geometry

    try:
        if bbox and geometry:
            raise ValueError("укажите либо geometry, либо bbox, но не оба сразу")
        bb = parse_bbox(bbox) if bbox else None
        req = QueryRequest(bbox=bb, geometry=None, date_from=date_from, date_to=date_to, source=source, clip=clip)
        if geometry:
            req._geom = parse_geometry(geometry)
    except ValidationError as e:
        _bad("; ".join(_msg(x) for x in e.errors()))
    except ValueError as e:
        _bad(str(e))
    return to_query(req)


def _fname(q: Query, stem: str, ext: str) -> str:
    return f"{stem}_{q.source}_{q.date_from.isoformat()}_{q.date_to.isoformat()}.{ext}"


def _attach(name: str) -> dict:
    return {"Content-Disposition": f'attachment; filename="{name}"'}


# ---------------------------------------------------------------------------
# Сервис
# ---------------------------------------------------------------------------


@app.get("/api/health", tags=["Сервис"], summary="Проверка работоспособности")
def health():
    t = time.time()
    cat = get_catalog()
    return {
        "status": "ok",
        "version": SERVICE_VERSION,
        "catalog_generated_at": cat.meta.get("generated_at"),
        "sources": list(cat.sources),
        "load_ms": round((time.time() - t) * 1000, 1),
    }


@app.get("/api/v1/meta", tags=["Сервис"], summary="Метаданные каталога: охват, период, источники, классы")
def meta():
    cat = get_catalog()
    srcs = {}
    for name, st in cat.sources.items():
        srcs[name] = {
            "label": C.SOURCE_LABELS[name],
            "n_hotspots": st.hs_n,
            "n_scars": len(st.scars),
            "n_zones": len(st.zones),
            "burned_area_ha": round(int(st.scar_sev_units.sum()) / 100, 2) if len(st.scars) else 0.0,
        }
    years = sorted({d[:4] for d in (cat.date_min, cat.date_max)})
    y0, y1 = int(cat.date_min[:4]), int(cat.date_max[:4])
    return {
        "service": "Мониторинг природных пожаров (LitWin)",
        "version": SERVICE_VERSION,
        "catalog": {k: cat.meta.get(k) for k in ("catalog_version", "generated_at", "pred_dir", "display_simplify_m")},
        "extent": cat.extent,
        "aoi_bbox": list(cat.aoi.bounds) if cat.aoi is not None else None,
        "date_min": cat.date_min,
        "date_max": cat.date_max,
        "years": list(range(y0, y1 + 1)) if years else [],
        "default_source": cat.default_source,
        "sources": srcs,
        "n_bs_chips": len(cat.bs_chips),
        "n_af_scenes": len(cat.af_chips),
        "severity": [{"severity": s, **v} for s, v in C.SEVERITY.items()],
        "landcover": [{"code": k, "name": v} for k, v in C.LANDCOVER.items()],
        "satellites": [{"code": k, "label": v} for k, v in C.SATELLITES.items()],
        "pixel_size_m": {"hotspots": C.AF_GSD, "burn_scars": C.BS_GSD},
        "checks": cat.meta.get("checks"),
        "timeline": cat.timeline(),
    }


# ---------------------------------------------------------------------------
# Основной запрос
# ---------------------------------------------------------------------------


def _scars_display_fc(cat, st, q, hits) -> dict:
    from .report import scar_row

    feats = []
    for h in hits:
        g = st.scar_display_ll(h, q)
        if g is None:
            continue
        p = scar_row(cat, st, h)
        feats.append({"type": "Feature", "id": p["scar_id"], "geometry": g, "properties": p})
    return {"type": "FeatureCollection", "name": "burn_scars", "crs": X.CRS84, "features": feats}


@app.post(
    "/api/v1/query",
    tags=["Запрос"],
    summary="Пространственно-временной запрос (термоточки + контуры + зоны + справка)",
    response_model=QueryResponse,
    responses={422: {"description": "Некорректные параметры (даты, полигон, bbox)"}},
)
def query(req: FullQueryRequest = Body(...)):
    return _query_response(to_query(req), req.include, req.detail, req.compare)


@app.get(
    "/api/v1/query",
    tags=["Запрос"],
    summary="То же, что POST /api/v1/query, но параметрами строки запроса (bbox или geometry)",
    response_model=QueryResponse,
    responses={422: {"description": "Некорректные параметры (даты, полигон, bbox)"}},
)
def query_get(
    bbox: Optional[str] = Q(None, description=BBOX_DESC, examples=["43,47,46,50"]),
    geometry: Optional[str] = Q(None, description=GEOM_DESC),
    date_from: Optional[date] = Q(None, description="YYYY-MM-DD"),
    date_to: Optional[date] = Q(None, description="YYYY-MM-DD"),
    source: Optional[Literal["model", "reference"]] = Q(None),
    include: list[Include] = Q(["hotspots", "burn_scars", "severity_zones", "report"], description="Какие блоки вернуть"),
    detail: Literal["display", "exact"] = Q("display", description="Геометрия гарей: display — упрощённая, exact — точная"),
    compare: bool = Q(True, description="Сравнение модели с эталоном в справке"),
    clip: bool = Q(True, description="Обрезать контуры границей запроса"),
):
    return _query_response(query_from_params(bbox, geometry, date_from, date_to, source, clip), include, detail, compare)


def _query_response(q: Query, include, detail: str, compare: bool):
    t0 = time.time()
    cat = get_catalog()
    st = cat.sources[q.source]
    hs_idx = st.hotspot_index(q)
    hits = st.scar_hits(q)
    out: dict = {"query": None}
    inc = set(include)
    if "hotspots" in inc:
        out["hotspots"] = X.hotspots_fc(st, hs_idx)
    if "burn_scars" in inc:
        if detail == "exact":
            out["burn_scars"] = X.polygons_fc(X.layer_items(cat, st, q, hits, "scars", True), "burn_scars")
        else:
            out["burn_scars"] = _scars_display_fc(cat, st, q, hits)
    if "severity_zones" in inc:
        out["severity_zones"] = X.polygons_fc(
            X.layer_items(cat, st, q, hits, "zones", exact=detail == "exact"), "severity_zones"
        )
    rep = build_report(cat, q, st, hits, hs_idx, compare=compare) if "report" in inc else None
    if rep is not None:
        out["report"] = rep
        out["query"] = rep["query"]
    else:
        out["query"] = {
            "spatial_filter": q.spatial,
            "bbox": q.bbox,
            "date_from": q.date_from.isoformat(),
            "date_to": q.date_to.isoformat(),
            "source": q.source,
            "clip": q.clip,
        }
    out["timing_ms"] = round((time.time() - t0) * 1000, 1)
    return jresp(out)


# ---------------------------------------------------------------------------
# Выгрузки
# ---------------------------------------------------------------------------

def _hotspots_response(q: Query, fmt: str):
    cat = get_catalog()
    st = cat.sources[q.source]
    idx = st.hotspot_index(q)
    if fmt == "csv":
        return Response(X.hotspots_csv(st, idx), media_type="text/csv; charset=utf-8", headers=_attach(_fname(q, "hotspots", "csv")))
    if fmt == "shp":
        return Response(X.hotspots_shapefile_zip(st, idx), media_type="application/zip", headers=_attach(_fname(q, "hotspots_shp", "zip")))
    if fmt == "gpkg":
        return Response(X.hotspots_gpkg(st, idx), media_type="application/geopackage+sqlite3", headers=_attach(_fname(q, "hotspots", "gpkg")))
    return jresp(X.hotspots_fc(st, idx), media_type="application/geo+json", headers=_attach(_fname(q, "hotspots", "geojson")))


@app.get("/api/v1/hotspots", tags=["Выгрузка"], summary="Термоточки за период в области (GeoJSON / CSV / Shapefile / GeoPackage)")
def hotspots_get(
    bbox: Optional[str] = Q(None, description=BBOX_DESC, examples=["43,47,46,50"]),
    geometry: Optional[str] = Q(None, description=GEOM_DESC),
    date_from: Optional[date] = Q(None, description="YYYY-MM-DD"),
    date_to: Optional[date] = Q(None, description="YYYY-MM-DD"),
    source: Optional[Literal["model", "reference"]] = Q(None),
    format: Literal["geojson", "csv", "shp", "gpkg"] = Q("geojson", description="Формат выгрузки"),
):
    return _hotspots_response(query_from_params(bbox, geometry, date_from, date_to, source), format)


@app.post("/api/v1/hotspots", tags=["Выгрузка"], summary="Термоточки по полигону (тело — как у /api/v1/query)")
def hotspots_post(req: QueryRequest, format: Literal["geojson", "csv", "shp", "gpkg"] = Q("geojson")):
    return _hotspots_response(to_query(req), format)


def _scars_response(q: Query, layer: str, fmt: str, crs: str, detail: str):
    cat = get_catalog()
    st = cat.sources[q.source]
    hits = st.scar_hits(q)
    exact = detail == "exact"
    items = X.layer_items(cat, st, q, hits, layer, exact)
    fields = X.ZONE_FIELDS if layer == "zones" else X.SCAR_FIELDS
    stem = "severity_zones" if layer == "zones" else "burn_scars"
    if fmt == "csv":
        return Response(X.items_csv(items, fields), media_type="text/csv; charset=utf-8", headers=_attach(_fname(q, stem, "csv")))
    if fmt == "gpkg":
        other = X.layer_items(cat, st, q, hits, "scars" if layer == "zones" else "zones", exact)
        sc, zn = (other, items) if layer == "zones" else (items, other)
        suffix = "_utm" if crs == "utm" else ""
        return Response(
            X.polygons_gpkg(sc, zn, crs),
            media_type="application/geopackage+sqlite3",
            headers=_attach(_fname(q, f"burn_scars{suffix}", "gpkg")),
        )
    if fmt == "shp":
        data = X.polygons_shapefile_zip(items, stem, fields, crs)
        suffix = "shp_utm" if crs == "utm" else "shp"
        return Response(data, media_type="application/zip", headers=_attach(_fname(q, f"{stem}_{suffix}", "zip")))
    return jresp(X.polygons_fc(items, stem), media_type="application/geo+json", headers=_attach(_fname(q, stem, "geojson")))


@app.get(
    "/api/v1/burn-scars",
    tags=["Выгрузка"],
    summary="Контуры гарей / зоны степени поражения (GeoJSON / Shapefile / GeoPackage / CSV)",
    description=(
        "layer=scars — контуры гарей (связные области), layer=zones — зоны степени внутри контуров. "
        "Геометрия по умолчанию точная (попиксельная), WGS84. Shapefile — zip (.shp/.shx/.dbf/.prj/.cpg, UTF-8); "
        "GeoPackage — оба слоя (контуры и зоны) в одном файле; crs=utm — отдельный слой на каждую зону UTM в её проекции."
    ),
)
def burn_scars_get(
    bbox: Optional[str] = Q(None, description=BBOX_DESC, examples=["43,47,46,50"]),
    geometry: Optional[str] = Q(None, description=GEOM_DESC),
    date_from: Optional[date] = Q(None),
    date_to: Optional[date] = Q(None),
    source: Optional[Literal["model", "reference"]] = Q(None),
    layer: Literal["scars", "zones"] = Q("scars"),
    format: Literal["geojson", "shp", "gpkg", "csv"] = Q("geojson"),
    crs: Literal["4326", "utm"] = Q("4326", description="Система координат Shapefile/GeoPackage: 4326 или utm (слой на каждую зону UTM)"),
    detail: Literal["exact", "display"] = Q("exact"),
    clip: bool = Q(True, description="Обрезать контуры границей запроса"),
):
    return _scars_response(query_from_params(bbox, geometry, date_from, date_to, source, clip), layer, format, crs, detail)


@app.post("/api/v1/burn-scars", tags=["Выгрузка"], summary="Контуры гарей / зоны по полигону (тело — как у /api/v1/query)")
def burn_scars_post(
    req: QueryRequest,
    layer: Literal["scars", "zones"] = Q("scars"),
    format: Literal["geojson", "shp", "gpkg", "csv"] = Q("geojson"),
    crs: Literal["4326", "utm"] = Q("4326"),
    detail: Literal["exact", "display"] = Q("exact"),
):
    return _scars_response(to_query(req), layer, format, crs, detail)


ReportFormat = Literal["json", "csv", "severity_csv", "scars_csv", "landcover_csv", "hotspots_csv", "zip"]


def _report_response(q: Query, fmt: str, compare: bool):
    cat = get_catalog()
    st = cat.sources[q.source]
    rep = build_report(cat, q, st, st.scar_hits(q), st.hotspot_index(q), compare=compare)
    if fmt == "json":
        return jresp(rep, headers={"Content-Disposition": f'inline; filename="{_fname(q, "report", "json")}"'})
    if fmt == "zip":
        return Response(X.report_zip(rep), media_type="application/zip", headers=_attach(_fname(q, "report", "zip")))
    fn = {
        "csv": (X.report_csv, "report_summary"),
        "severity_csv": (X.severity_csv, "report_severity"),
        "scars_csv": (X.scars_csv, "report_scars"),
        "landcover_csv": (X.landcover_csv, "report_landcover"),
        "hotspots_csv": (X.hotspots_by_date_csv, "report_hotspots_by_date"),
    }[fmt]
    return Response(fn[0](rep), media_type="text/csv; charset=utf-8", headers=_attach(_fname(q, fn[1], "csv")))


@app.get(
    "/api/v1/report",
    tags=["Запрос"],
    summary="Аналитическая справка (JSON / CSV / ZIP)",
    responses={200: {"model": Report, "description": "Справка (format=json)"}},
    description=(
        "Общая площадь гарей, распределение по классам степени, таблица контуров, термоточки, типы покрова, "
        "сравнение модели с эталоном. format=csv — сводка (section,metric,value,unit); severity_csv — по классам; "
        "scars_csv — по контурам; zip — всё вместе + report.json."
    ),
)
def report_get(
    bbox: Optional[str] = Q(None, description=BBOX_DESC, examples=["43,47,46,50"]),
    geometry: Optional[str] = Q(None, description=GEOM_DESC),
    date_from: Optional[date] = Q(None),
    date_to: Optional[date] = Q(None),
    source: Optional[Literal["model", "reference"]] = Q(None),
    format: ReportFormat = Q("json"),
    compare: bool = Q(True, description="Сравнение модели с эталоном"),
    clip: bool = Q(True),
):
    return _report_response(query_from_params(bbox, geometry, date_from, date_to, source, clip), format, compare)


@app.post("/api/v1/report", tags=["Запрос"], summary="Аналитическая справка по полигону", responses={200: {"model": Report}})
def report_post(req: QueryRequest, format: ReportFormat = Q("json"), compare: bool = Q(True)):
    return _report_response(to_query(req), format, compare)


# ---------------------------------------------------------------------------
# Справочные слои
# ---------------------------------------------------------------------------


@app.get("/api/v1/aoi", tags=["Справочные слои"], summary="Граница территории мониторинга (GeoJSON)")
def aoi():
    cat = get_catalog()
    if cat.aoi_fc is None:
        _bad("граница территории не найдена", 404)
    return jresp(cat.aoi_fc, media_type="application/geo+json")


@app.get("/api/v1/chips", tags=["Справочные слои"], summary="Покрытие снимками: чипы Sentinel-2 (bs) и сцены VIIRS (af)")
def chips(kind: Literal["bs", "af"] = Q("bs")):
    cat = get_catalog()
    feats = []
    if kind == "bs":
        for c in cat.bs_chips:
            props = {k: c[k] for k in ("chip_id", "fire_event_id", "date_pre", "date_post", "epsg", "valid_frac", "cloud_frac", "burn_area_ha_meta")}
            props.update({"quicklook": c.get("quicklook"), "overlays": c.get("overlays"), "corners": c["corners"]})
            feats.append({"type": "Feature", "id": c["chip_id"], "geometry": {"type": "Polygon", "coordinates": [c["footprint"]]}, "properties": props})
    else:
        groups: dict = {}
        for i, c in enumerate(cat.af_chips):
            key = (c["epsg"], *c["bounds_utm"])
            groups.setdefault(key, []).append(i)
        for key, idx in groups.items():
            cs = [cat.af_chips[i] for i in idx]
            times = sorted(c["acq_datetime"] for c in cs)
            geom = shapely.geometry.mapping(cat.af_footprint[idx[0]])
            feats.append(
                {
                    "type": "Feature",
                    "id": f"T{key[0]}_{int(key[1])}_{int(key[2])}",
                    "geometry": geom,
                    "properties": {
                        "tile_id": f"T{key[0]}_{int(key[1])}_{int(key[2])}",
                        "epsg": key[0],
                        "n_scenes": len(cs),
                        "first": times[0],
                        "last": times[-1],
                        "scenes": [{"chip_id": c["chip_id"], "acq_datetime": c["acq_datetime"], "satellite": c["satellite"]} for c in sorted(cs, key=lambda c: c["acq_datetime"])],
                    },
                }
            )
    return jresp({"type": "FeatureCollection", "name": f"chips_{kind}", "crs": X.CRS84, "features": feats}, media_type="application/geo+json")


@app.get("/api/v1/scars/{scar_id}", tags=["Справочные слои"], summary="Карточка контура гари: атрибуты, зоны, превью снимков")
def scar_detail(scar_id: str, source: Optional[Literal["model", "reference"]] = Q(None)):
    cat = get_catalog()
    src = source or ("model" if "-M" in scar_id and "model" in cat.sources else "reference" if "-R" in scar_id else cat.default_source)
    st = cat.sources.get(src)
    if st is None:
        _bad(f"источник «{src}» недоступен", 404)
    idx = next((i for i, s in enumerate(st.scars) if s["scar_id"] == scar_id), None)
    if idx is None:
        _bad(f"контур {scar_id} не найден в источнике {src}", 404)
    from .catalog import ScarHit
    from .report import scar_row

    h = ScarHit(idx, False, st.scar_sev_units[idx].tolist())
    chip = cat.bs_chips[st.scar_chip[idx]]
    lc = st.scars[idx]["lc_px"]
    px_ha = chip["gsd"] ** 2 / 1e4
    return jresp(
        {
            "scar": scar_row(cat, st, h),
            "geometry": st.disp_ll[idx],
            "landcover": [
                {"code": int(c), "name": C.landcover_name(c), "area_ha": round(sum(v) * px_ha, 2), "by_severity_ha": [round(x * px_ha, 2) for x in v]}
                for c, v in sorted(lc.items(), key=lambda kv: -sum(kv[1]))
            ],
            "chip": {
                k: chip.get(k)
                for k in ("chip_id", "fire_event_id", "date_pre", "date_post", "s1_date_pre", "s1_date_post", "epsg", "corners", "quicklook", "overlays", "valid_frac", "cloud_frac")
            },
        }
    )


# ---------------------------------------------------------------------------
# Необязательный анализ нового чипа
# ---------------------------------------------------------------------------

from .analyze import router as analyze_router  # noqa: E402

app.include_router(analyze_router)


# ---------------------------------------------------------------------------
# Веб-интерфейс и статика
# ---------------------------------------------------------------------------

if (DATA_DIR / "catalog").exists():
    app.mount("/catalog", StaticFiles(directory=DATA_DIR / "catalog"), name="catalog")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")
