"""Загрузка каталога и пространственно-временные запросы.

Каталог строится заранее (``service/build_catalog.py``) и целиком держится в
памяти: термоточки — numpy-массивы, контуры гарей — массивы shapely-геометрий
в «родной» проекции UTM каждого чипа с пространственным индексом STRtree.

Площади
-------
Растры гарей заданы на сетке UTM (EPSG:32637/32638) с пикселем 20x20 м, поэтому
площадь контура без обрезки = число пикселей * 400 м² (ровно так же посчитан
``burn_area_ha`` в meta.csv). Если полигон запроса пересекает контур, точная
(попиксельная) геометрия контура и зон степени пересекается с полигоном
запроса, переведённым в ту же зону UTM (стороны полигона предварительно
уплотняются, чтобы прямые в WGS84 корректно легли в UTM), и площадь
пересечения считается планиметрически в UTM.

Все площади ведутся в целых «сотых гектара» (1 = 100 м²): суммы по классам,
контурам и итог сходятся без ошибок округления.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import shapely
from pyproj import Transformer

from . import common as C

M2_PER_UNIT = 100.0  # 1 единица площади = 0.01 га = 100 м²


def ha(units) -> float:
    """Целые сотые гектара -> гектары (float с 2 знаками)."""
    return round(int(units) / 100.0, 2)


@lru_cache(maxsize=16)
def _tr_to_ll(epsg: int) -> Transformer:
    return Transformer.from_crs(int(epsg), 4326, always_xy=True)


@lru_cache(maxsize=16)
def _tr_from_ll(epsg: int) -> Transformer:
    return Transformer.from_crs(4326, int(epsg), always_xy=True)


def utm_to_ll(geoms, epsg: int, ndigits: int | None = None):
    tr = _tr_to_ll(epsg)

    def f(c):
        if len(c) == 0:
            return c
        x, y = tr.transform(c[:, 0], c[:, 1])
        out = np.column_stack([x, y])
        return np.round(out, ndigits) if ndigits is not None else out

    return shapely.transform(geoms, f)


def ll_to_utm(geoms, epsg: int):
    tr = _tr_from_ll(epsg)

    def f(c):
        if len(c) == 0:
            return c
        x, y = tr.transform(c[:, 0], c[:, 1])
        return np.column_stack([x, y])

    return shapely.transform(geoms, f)


def polygonal(g):
    """Оставляет только площадные части (после пересечения могут появиться линии/точки)."""
    if g is None or g.is_empty:
        return None
    t = shapely.get_type_id(g)
    if t in (3, 6):
        return g
    if t == 7:
        parts = [p for p in shapely.get_parts(g) if shapely.get_type_id(p) in (3, 6) and not p.is_empty]
        if not parts:
            return None
        return C.to_multipolygon(shapely.union_all(parts))
    return None


def apportion(total: int, weights) -> list[int]:
    """Разбиение целого total пропорционально весам методом наибольших остатков."""
    w = np.asarray(weights, dtype=np.float64)
    if total <= 0 or w.sum() <= 0:
        return [0] * len(w)
    raw = w / w.sum() * total
    base = np.floor(raw).astype(np.int64)
    rest = int(total - base.sum())
    if rest > 0:
        order = np.argsort(-(raw - base), kind="stable")[:rest]
        base[order] += 1
    return base.tolist()


# ---------------------------------------------------------------------------
# Запрос
# ---------------------------------------------------------------------------


@dataclass
class Query:
    geom: object | None  # shapely (Multi)Polygon в WGS84 или None — без пространственного фильтра
    spatial: str  # "polygon" | "bbox" | "none"
    date_from: date
    date_to: date
    source: str
    clip: bool = True
    bbox: list[float] | None = None
    # охват данных каталога (WGS84, с запасом): полигон запроса обрезается по нему
    # перед перепроецированием в UTM — см. utm()
    data_bounds: tuple[float, float, float, float] | None = None
    _utm: dict = field(default_factory=dict, repr=False)
    _prepared: bool = False

    def prepare(self):
        if self.geom is not None and not self._prepared:
            shapely.prepare(self.geom)
            self._prepared = True
        return self

    def utm(self, epsg: int):
        """Полигон запроса в заданной зоне UTM (стороны уплотнены до ~1 км).

        Полигон запроса может быть сколь угодно большим (вплоть до всего мира), а
        перепроецирование далёких от зоны UTM координат «заворачивает» геометрию и
        даёт бессмысленный результат. Поэтому запрос сначала обрезается по охвату
        данных каталога: все чипы лежат внутри него, так что на ответ это не влияет.
        """
        if epsg not in self._utm:
            src = self.geom
            if self.data_bounds is not None:
                box = shapely.box(*self.data_bounds)
                if not shapely.covers(box, src):
                    src = shapely.intersection(src, box)
            if src.is_empty:
                g = shapely.Polygon()
            else:
                dense = shapely.segmentize(src, 0.01)
                g = shapely.make_valid(ll_to_utm(dense, epsg))
                g = polygonal(g) or g
            shapely.prepare(g)
            self._utm[epsg] = g
        return self._utm[epsg]

    def area_km2(self) -> float | None:
        """Площадь полигона запроса на эллипсоиде WGS84, км²."""
        if self.geom is None:
            return None
        from pyproj import Geod

        a, _ = Geod(ellps="WGS84").geometry_area_perimeter(shapely.segmentize(self.geom, 0.05))
        return round(abs(a) / 1e6, 3)


@dataclass
class ScarHit:
    idx: int
    clipped: bool
    sev_units: list[int]  # площади по классам 1..3 в сотых га (внутри запроса)
    zone_geoms: dict | None = None  # {severity: обрезанная точная геометрия UTM} (только при обрезке)

    @property
    def units(self) -> int:
        return int(sum(self.sev_units))


# ---------------------------------------------------------------------------
# Данные одного источника (модель / эталон)
# ---------------------------------------------------------------------------


class SourceStore:
    def __init__(self, cat_dir: Path, name: str, bs_chips: list[dict], af_chips: list[dict]):
        self.name = name
        d = cat_dir / name
        self.dir = d
        self.bs_chips = bs_chips
        # ---------- термоточки ----------
        with open(d / "hotspots.json", encoding="utf-8") as f:
            hs = json.load(f)
        self.hs = hs
        self.hs_n = len(hs["id"])
        self.hs_lon = np.asarray(hs["lon"], np.float64)
        self.hs_lat = np.asarray(hs["lat"], np.float64)
        self.hs_t = np.asarray([t.rstrip("Z") for t in hs["t"]], dtype="datetime64[s]")
        self.hs_date = self.hs_t.astype("datetime64[D]")
        self.hs_dn = np.asarray(hs["dn"])
        self.hs_sat = np.asarray(hs["sat"])
        self.hs_lc = np.asarray(hs["lc"], np.int64)
        self.hs_key = np.asarray([f"{c}|{r}|{k}" for c, r, k in zip(hs["chip"], hs["row"], hs["col"])])
        # ---------- гари ----------
        with open(d / "scars.json", encoding="utf-8") as f:
            sj = json.load(f)
        self.scars = sj["scars"]
        self.zones = sj["zones"]
        n = len(self.scars)
        z = np.load(d / "geometry.npz")
        enc = lambda p: {k: z[f"{p}_{k}"] for k in ("d", "o0", "o1", "o2")}  # noqa: E731
        sc = z["scar_chip"]
        zc = z["zone_chip"]
        chip_x0 = np.asarray([c["x0"] for c in bs_chips])
        chip_y0 = np.asarray([c["y0"] for c in bs_chips])
        chip_gsd = np.asarray([c["gsd"] for c in bs_chips])
        self.scar_exact = C.decode_geoms(enc("scar"), chip_x0[sc], chip_y0[sc], chip_gsd[sc])
        self.scar_disp = C.decode_geoms(enc("disp"), chip_x0[sc], chip_y0[sc], chip_gsd[sc])
        self.zone_exact = C.decode_geoms(enc("zone"), chip_x0[zc], chip_y0[zc], chip_gsd[zc])
        assert len(self.scar_exact) == n and len(self.zone_exact) == len(self.zones)
        self.scar_chip = sc.astype(np.int64)
        self.scar_epsg = np.asarray([bs_chips[c]["epsg"] for c in sc], np.int64)
        self.scar_gsd = chip_gsd[sc] if n else np.zeros(0)
        self.scar_date_pre = np.asarray([bs_chips[c]["date_pre"] for c in sc], dtype="datetime64[D]")
        self.scar_date_post = np.asarray([bs_chips[c]["date_post"] for c in sc], dtype="datetime64[D]")
        self.scar_zones: list[list[int]] = [[] for _ in range(n)]
        for zi, zz in enumerate(self.zones):
            self.scar_zones[zz["scar"]].append(zi)
        # площади без обрезки, в сотых га: n_px * gsd² / 100 м²
        px_units = (self.scar_gsd**2) / M2_PER_UNIT
        self.scar_sev_units = np.rint(np.asarray([s["sev_px"] for s in self.scars], np.float64).reshape(n, 3) * px_units[:, None]).astype(np.int64)
        shapely.prepare(self.scar_exact)
        self.trees: dict[int, tuple[shapely.STRtree, np.ndarray]] = {}
        for epsg in np.unique(self.scar_epsg):
            idx = np.nonzero(self.scar_epsg == epsg)[0]
            self.trees[int(epsg)] = (shapely.STRtree(self.scar_exact[idx]), idx)
        # упрощённая геометрия в WGS84 для веб-карты (без обрезки) — готовые GeoJSON-словари
        self.disp_ll: list[dict | None] = [None] * n
        for epsg in np.unique(self.scar_epsg):
            idx = np.nonzero(self.scar_epsg == epsg)[0]
            ll = utm_to_ll(self.scar_disp[idx], int(epsg), 6)
            for i, g in zip(idx, ll):
                self.disp_ll[i] = shapely.geometry.mapping(g)
        self.scar_bounds = shapely.bounds(self.scar_exact) if n else np.zeros((0, 4))
        self.masks = np.load(d / "masks.npz")
        self._mask_lock = threading.Lock()

    def mask(self, chip_id: str) -> np.ndarray | None:
        with self._mask_lock:
            if chip_id not in self.masks.files:
                return None
            return self.masks[chip_id]

    # ------------------------------------------------------------------
    def hotspot_index(self, q: Query) -> np.ndarray:
        sel = (self.hs_date >= np.datetime64(q.date_from)) & (self.hs_date <= np.datetime64(q.date_to))
        if q.geom is not None:
            q.prepare()
            cand = np.nonzero(sel)[0]
            inside = shapely.intersects_xy(q.geom, self.hs_lon[cand], self.hs_lat[cand])
            sel = np.zeros(self.hs_n, bool)
            sel[cand[inside]] = True
        idx = np.nonzero(sel)[0]
        # хронологический порядок
        return idx[np.argsort(self.hs_t[idx], kind="stable")]

    def time_mask(self, q: Query) -> np.ndarray:
        """Окно наблюдения гари [date_pre, date_post] пересекается с периодом запроса."""
        return (self.scar_date_pre <= np.datetime64(q.date_to)) & (self.scar_date_post >= np.datetime64(q.date_from))

    def scar_hits(self, q: Query) -> list[ScarHit]:
        tsel = self.time_mask(q)
        hits: list[ScarHit] = []
        if q.geom is None:
            for i in np.nonzero(tsel)[0]:
                hits.append(ScarHit(int(i), False, self.scar_sev_units[i].tolist()))
            return hits
        for epsg, (tree, idx_map) in self.trees.items():
            qg = q.utm(epsg)
            cand = idx_map[tree.query(qg, predicate="intersects")]
            cand = np.sort(cand[tsel[cand]])
            if len(cand) == 0:
                continue
            b = self.scar_bounds[cand]
            inside = shapely.contains(qg, shapely.box(b[:, 0], b[:, 1], b[:, 2], b[:, 3]))
            rest = cand[~inside]
            if len(rest):
                inside2 = shapely.contains(qg, self.scar_exact[rest])
                inside[~inside] = inside2
            for i, ins in zip(cand.tolist(), inside.tolist()):
                if ins or not q.clip:
                    hits.append(ScarHit(i, False, self.scar_sev_units[i].tolist()))
                    continue
                sev_units = [0, 0, 0]
                zg = {}
                for zi in self.scar_zones[i]:
                    s = self.zones[zi]["severity"]
                    g = polygonal(shapely.intersection(self.zone_exact[zi], qg))
                    if g is None:
                        continue
                    u = int(round(g.area / M2_PER_UNIT))
                    if g.area > 0:
                        zg[s] = g
                        sev_units[s - 1] = u
                if sum(sev_units) > 0:
                    hits.append(ScarHit(i, True, sev_units, zg))
        hits.sort(key=lambda h: h.idx)
        return hits

    # ------------------------------------------------------------------
    def clipped_scar_geom(self, hit: ScarHit, q: Query, exact: bool = True):
        """Геометрия контура (UTM) с учётом обрезки полигоном запроса."""
        g = self.scar_exact[hit.idx] if exact else self.scar_disp[hit.idx]
        if not hit.clipped:
            return g
        return polygonal(shapely.intersection(g, q.utm(int(self.scar_epsg[hit.idx]))))

    def scar_display_ll(self, hit: ScarHit, q: Query) -> dict | None:
        if not hit.clipped:
            return self.disp_ll[hit.idx]
        g = self.clipped_scar_geom(hit, q, exact=False)
        if g is None:  # упрощённый контур мог «не попасть», берём точный
            g = self.clipped_scar_geom(hit, q, exact=True)
        if g is None:
            return None
        return shapely.geometry.mapping(utm_to_ll(g, int(self.scar_epsg[hit.idx]), 6))


# ---------------------------------------------------------------------------
# Каталог
# ---------------------------------------------------------------------------


class Catalog:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.cat_dir = self.data_dir / "catalog"
        if not (self.cat_dir / "meta.json").exists():
            raise FileNotFoundError(
                f"Каталог не найден: {self.cat_dir}. Соберите его: python service/build_catalog.py --data-dir <train> --out service/data"
            )
        with open(self.cat_dir / "meta.json", encoding="utf-8") as f:
            self.meta = json.load(f)
        with open(self.cat_dir / "chips.json", encoding="utf-8") as f:
            chips = json.load(f)
        self.bs_chips: list[dict] = chips["bs"]
        self.af_chips: list[dict] = chips["af"]
        self.bs_chip_index = {c["chip_id"]: i for i, c in enumerate(self.bs_chips)}
        self.af_chip_index = {c["chip_id"]: i for i, c in enumerate(self.af_chips)}
        self.aoi_fc = None
        self.aoi = None
        aoi_path = self.data_dir / "aoi.geojson"
        if aoi_path.exists():
            with open(aoi_path, encoding="utf-8") as f:
                self.aoi_fc = json.load(f)
            feat = next((ft for ft in self.aoi_fc["features"] if ft.get("id") == "aoi"), self.aoi_fc["features"][0])
            self.aoi = shapely.geometry.shape(feat["geometry"])
        self.sources: dict[str, SourceStore] = {}
        for s in self.meta["sources"]:
            self.sources[s] = SourceStore(self.cat_dir, s, self.bs_chips, self.af_chips)
        self.default_source = "model" if "model" in self.sources else "reference"
        # полигоны чипов BS (WGS84) — для сравнения растров
        self.bs_footprint = np.asarray([shapely.Polygon(c["footprint"]) for c in self.bs_chips], dtype=object)
        self.af_date = np.asarray([c["acq_datetime"][:10] for c in self.af_chips], dtype="datetime64[D]")
        self.af_footprint = []
        for c in self.af_chips:
            x0, y0, x1, y1 = c["bounds_utm"]
            box = shapely.segmentize(shapely.box(x0, y0, x1, y1), 4000)
            self.af_footprint.append(utm_to_ll(box, c["epsg"], 6))
        self.af_footprint = np.asarray(self.af_footprint, dtype=object)
        # диапазон дат
        dates = [d for c in self.bs_chips for d in (c["date_pre"], c["date_post"]) if d]
        dates += [c["acq_datetime"][:10] for c in self.af_chips]
        self.date_min = min(dates)
        self.date_max = max(dates)
        allb = [shapely.bounds(g) for g in list(self.bs_footprint) + list(self.af_footprint)]
        allb = np.asarray(allb)
        self.extent = [float(allb[:, 0].min()), float(allb[:, 1].min()), float(allb[:, 2].max()), float(allb[:, 3].max())]
        # охват данных с запасом 1° — рамка обрезки полигона запроса перед переходом в UTM
        self.data_bounds = (
            max(-180.0, self.extent[0] - 1.0),
            max(-90.0, self.extent[1] - 1.0),
            min(180.0, self.extent[2] + 1.0),
            min(90.0, self.extent[3] + 1.0),
        )
        # попиксельные матрицы ошибок эталон/модель для целых чипов (кэш)
        self._chip_conf: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    def timeline(self) -> dict:
        """Помесячные ряды для ленты времени: число термоточек и площадь гарей (по дате снимка «после»)."""
        m0 = np.datetime64(self.date_min[:7], "M")
        m1 = np.datetime64(self.date_max[:7], "M")
        months = np.arange(m0, m1 + np.timedelta64(1, "M"))
        n = len(months)
        out = {"months": [str(m) for m in months], "sources": {}}
        for name, st in self.sources.items():
            hi = (st.hs_t.astype("datetime64[M]") - m0).astype(np.int64)
            hs = np.bincount(hi[(hi >= 0) & (hi < n)], minlength=n)
            si = (st.scar_date_post.astype("datetime64[M]") - m0).astype(np.int64)
            ok = (si >= 0) & (si < n)
            w = st.scar_sev_units.sum(axis=1) if len(st.scars) else np.zeros(0)
            bu = np.bincount(si[ok], weights=w[ok], minlength=n) if ok.any() else np.zeros(n)
            out["sources"][name] = {"hotspots": hs.astype(int).tolist(), "burned_ha": [round(v / 100.0, 2) for v in bu.tolist()]}
        return out

    # ------------------------------------------------------------------
    def chips_in_query(self, q: Query, kind: str) -> np.ndarray:
        if kind == "bs":
            d0 = np.asarray([c["date_pre"] for c in self.bs_chips], dtype="datetime64[D]")
            d1 = np.asarray([c["date_post"] for c in self.bs_chips], dtype="datetime64[D]")
            sel = (d0 <= np.datetime64(q.date_to)) & (d1 >= np.datetime64(q.date_from))
            fp = self.bs_footprint
        else:
            sel = (self.af_date >= np.datetime64(q.date_from)) & (self.af_date <= np.datetime64(q.date_to))
            fp = self.af_footprint
        if q.geom is not None:
            q.prepare()
            sel &= shapely.intersects(q.geom, fp)
        return np.nonzero(sel)[0]

    def confusion(self, q: Query) -> tuple[np.ndarray, int]:
        """Попиксельная матрица ошибок 4x4 (строки — эталон, столбцы — модель) внутри запроса.

        Пиксель учитывается, если его центр лежит внутри полигона запроса.
        """
        ref, mod = self.sources.get("reference"), self.sources.get("model")
        cm = np.zeros((4, 4), np.int64)
        n_chips = 0
        if ref is None or mod is None:
            return cm, 0
        for ci in self.chips_in_query(q, "bs"):
            chip = self.bs_chips[ci]
            cid = chip["chip_id"]
            full = q.geom is None or shapely.contains(q.geom, self.bs_footprint[ci])
            if full and cid in self._chip_conf:
                cm += self._chip_conf[cid]
                n_chips += 1
                continue
            a, b = ref.mask(cid), mod.mask(cid)
            if a is None or b is None:
                continue
            a = a.astype(np.int64)
            b = b.astype(np.int64)
            if full:
                c = np.bincount((a * 4 + b).ravel(), minlength=16).reshape(4, 4)
                self._chip_conf[cid] = c
                cm += c
                n_chips += 1
                continue
            qg = q.utm(chip["epsg"])
            gsd = chip["gsd"]
            h, w = a.shape
            xs = chip["x0"] + (np.arange(w) + 0.5) * gsd
            ys = chip["y0"] - (np.arange(h) + 0.5) * gsd
            xx, yy = np.meshgrid(xs, ys)
            inside = shapely.intersects_xy(qg, xx.ravel(), yy.ravel()).reshape(h, w)
            if not inside.any():
                continue
            cm += np.bincount((a[inside] * 4 + b[inside]).ravel(), minlength=16).reshape(4, 4)
            n_chips += 1
        return cm, n_chips
