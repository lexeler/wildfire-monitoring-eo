"""Модели запросов/ответов REST API (pydantic) и разбор пространственного фильтра."""
from __future__ import annotations

from datetime import date
from typing import Any, Literal, Optional

import shapely
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

MAX_VERTICES = 20000
Source = Literal["model", "reference"]


class GeometryError(ValueError):
    pass


def parse_bbox(v) -> list[float]:
    if isinstance(v, str):
        try:
            v = [float(x) for x in v.replace(";", ",").split(",") if x.strip() != ""]
        except ValueError:
            raise GeometryError("bbox: ожидаются 4 числа через запятую: minLon,minLat,maxLon,maxLat")
    if v is None or len(v) != 4:
        raise GeometryError("bbox: ожидаются 4 числа: minLon,minLat,maxLon,maxLat")
    x0, y0, x1, y1 = (float(a) for a in v)
    if not (-180 <= x0 <= 180 and -180 <= x1 <= 180 and -90 <= y0 <= 90 and -90 <= y1 <= 90):
        raise GeometryError("bbox: долгота должна быть в [-180, 180], широта — в [-90, 90] (EPSG:4326)")
    if not (x0 < x1 and y0 < y1):
        raise GeometryError("bbox: должно выполняться minLon < maxLon и minLat < maxLat")
    return [x0, y0, x1, y1]


def parse_geometry(obj: Any):
    """GeoJSON Polygon/MultiPolygon, Feature или FeatureCollection (EPSG:4326) -> shapely."""
    if isinstance(obj, str):
        import json

        try:
            obj = json.loads(obj)
        except json.JSONDecodeError:
            try:
                g = shapely.from_wkt(obj)
            except Exception:
                raise GeometryError("geometry: ожидается GeoJSON (или WKT) полигона в EPSG:4326")
            obj = shapely.geometry.mapping(g)
    if not isinstance(obj, dict) or "type" not in obj:
        raise GeometryError("geometry: ожидается объект GeoJSON с полем type")
    t = obj["type"]
    if t == "FeatureCollection":
        feats = obj.get("features")
        if not isinstance(feats, list):
            raise GeometryError("geometry: у FeatureCollection нет списка features")
        # элементы без геометрии (null, пустые Feature) пропускаем, а не падаем на них
        geoms = [parse_geometry(f) for f in feats if isinstance(f, dict) and f.get("geometry")]
        if not geoms:
            raise GeometryError("geometry: FeatureCollection не содержит полигонов")
        return shapely.union_all(geoms)
    if t == "Feature":
        if not obj.get("geometry"):
            raise GeometryError("geometry: у Feature нет геометрии")
        return parse_geometry(obj["geometry"])
    if t not in ("Polygon", "MultiPolygon"):
        raise GeometryError(f"geometry: поддерживаются только Polygon и MultiPolygon, получен {t}")
    try:
        g = shapely.geometry.shape(obj)
    except Exception as e:  # noqa: BLE001
        raise GeometryError(f"geometry: не удалось разобрать координаты ({e})")
    if g.is_empty:
        raise GeometryError("geometry: пустой полигон")
    n = shapely.get_num_coordinates(g)
    if n > MAX_VERTICES:
        raise GeometryError(f"geometry: слишком много вершин ({n} > {MAX_VERTICES})")
    minx, miny, maxx, maxy = g.bounds
    if minx < -180 or maxx > 180 or miny < -90 or maxy > 90:
        raise GeometryError("geometry: координаты вне диапазона EPSG:4326 (порядок — долгота, широта)")
    if not g.is_valid:
        from shapely.validation import explain_validity

        why = explain_validity(g)
        import re

        m = re.match(r"(Ring )?Self-intersection\[([-\d.eE]+) ([-\d.eE]+)\]", why)
        if m:
            why = f"самопересечение границы около точки {float(m.group(2)):.4f}, {float(m.group(3)):.4f}"
        raise GeometryError(f"geometry: некорректный полигон — {why}; нарисуйте контур без самопересечений")
    if g.area <= 0:
        raise GeometryError("geometry: полигон нулевой площади")
    return g


class QueryRequest(BaseModel):
    """Пространственно-временной запрос. Указывается либо geometry, либо bbox (либо ничего — весь каталог)."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "bbox": [43.0, 47.0, 46.0, 50.0],
                    "date_from": "2019-01-01",
                    "date_to": "2024-12-31",
                    "source": "model",
                },
                {
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[44.0, 48.0], [45.5, 48.0], [45.5, 49.2], [44.0, 49.2], [44.0, 48.0]]],
                    },
                    "date_from": "2021-04-01",
                    "date_to": "2021-10-31",
                    "source": "reference",
                },
            ]
        }
    )

    geometry: Optional[dict[str, Any]] = Field(
        None, description="Полигон запроса: GeoJSON Polygon/MultiPolygon (или Feature/FeatureCollection) в EPSG:4326"
    )
    bbox: Optional[list[float]] = Field(None, description="Прямоугольник [minLon, minLat, maxLon, maxLat], EPSG:4326")
    date_from: Optional[date] = Field(None, description="Начало периода (YYYY-MM-DD, включительно). По умолчанию — начало данных")
    date_to: Optional[date] = Field(None, description="Конец периода (YYYY-MM-DD, включительно). По умолчанию — конец данных")
    source: Optional[Source] = Field(None, description="Источник: model — результаты модели (по умолчанию), reference — эталонная разметка")
    clip: bool = Field(True, description="Обрезать контуры гарей границей полигона запроса (площади — только внутри запроса)")

    _geom: Any = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _check(self):
        if self.geometry is not None and self.bbox is not None:
            raise ValueError("укажите либо geometry, либо bbox, но не оба сразу")
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError(f"date_from ({self.date_from}) позже date_to ({self.date_to})")
        if self.bbox is not None:
            self.bbox = parse_bbox(self.bbox)
            self._geom = shapely.box(*self.bbox)
        elif self.geometry is not None:
            self._geom = parse_geometry(self.geometry)
        return self


Include = Literal["hotspots", "burn_scars", "severity_zones", "report"]


class FullQueryRequest(QueryRequest):
    include: list[Include] = Field(
        default_factory=lambda: ["hotspots", "burn_scars", "severity_zones", "report"],
        description="Какие блоки вернуть",
    )
    detail: Literal["display", "exact"] = Field(
        "display", description="Геометрия гарей: display — упрощённая для карты (допуск 20 м), exact — точная попиксельная"
    )
    compare: bool = Field(True, description="Добавить в справку сравнение модели с эталоном (если доступны оба)")


# ---------------------------------------------------------------------------
# Модели ответа (для документации OpenAPI)
# ---------------------------------------------------------------------------


class SeverityRow(BaseModel):
    severity: int = Field(description="Класс степени поражения: 1 — слабая, 2 — средняя, 3 — сильная")
    key: str
    label: str
    color: str
    area_ha: float = Field(description="Площадь, га")
    share_pct: float = Field(description="Доля от общей площади гарей, %")
    n_zones: int
    n_scars_dominant: int


class ScarRow(BaseModel):
    scar_id: str
    chip_id: str
    fire_event_id: Optional[str]
    date_pre: str
    date_post: str
    area_ha: float = Field(description="Площадь контура внутри запроса, га")
    area_low_ha: float
    area_moderate_ha: float
    area_high_ha: float
    share_low_pct: float
    share_moderate_pct: float
    share_high_pct: float
    dominant_severity: int
    dominant_label: Optional[str]
    clipped: bool = Field(description="Контур обрезан границей запроса")
    area_total_ha: float = Field(description="Полная площадь контура (без обрезки), га")
    n_pixels: int
    pixel_size_m: float
    utm_epsg: int
    center_lon: float
    center_lat: float
    bbox: list[float]
    confidence: Optional[float] = Field(description="Средняя вероятность гари по модели (для эталона — null)")
    source: str


class Summary(BaseModel):
    burned_area_ha: float = Field(description="Общая площадь гарей внутри запроса, га")
    n_scars: int
    n_scars_clipped: int
    burned_area_unclipped_ha: float
    burned_share_of_query_pct: Optional[float]
    n_hotspots: int
    n_hotspots_day: int
    n_hotspots_night: int
    n_bs_chips: int
    n_af_scenes: int


class Report(BaseModel):
    title: str
    generated_at: str
    service: dict
    query: dict
    summary: Summary
    severity: list[SeverityRow]
    burned_by_month: list[dict]
    landcover: list[dict]
    landcover_estimated: bool
    hotspots: dict
    scars: list[ScarRow]
    comparison: Optional[dict]
    method: dict


class QueryResponse(BaseModel):
    query: dict
    hotspots: Optional[dict] = Field(None, description="GeoJSON FeatureCollection термоточек (Point, EPSG:4326)")
    burn_scars: Optional[dict] = Field(None, description="GeoJSON FeatureCollection контуров гарей (EPSG:4326)")
    severity_zones: Optional[dict] = Field(None, description="GeoJSON FeatureCollection зон степени поражения")
    report: Optional[Report] = None
    timing_ms: Optional[float] = None
