#!/usr/bin/env python
"""Сборка каталога сервиса из обучающей выборки и (необязательно) предсказаний модели.

Пример:
    python service/build_catalog.py --data-dir /path/to/train \
        --pred-dir /path/to/oof_preds --out service/data

Результат — каталог ``<out>/catalog`` (заменяется целиком), который сервис
загружает при старте. Формат предсказаний (``--pred-dir``):
    <pred_dir>/af/<chip_id>.npz  : mask uint8 (H, W) 0/1,   prob float16 (H, W)
    <pred_dir>/bs/<chip_id>.npz  : mask uint8 (H, W) 0..3,  prob float16 (H, W)
Без ``--pred-dir`` в каталог попадает только эталонная разметка.

Зависимости сборщика: numpy, pandas, rasterio, scipy, shapely, pyproj, pillow.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from service.app import common as C  # noqa: E402
from service.app.vectorize import extract_hotspots, vectorize_severity  # noqa: E402

QL_SIZE = 320  # сторона превью Sentinel-2, пикселей (исходно 512 px по 20 м)
QL_QUALITY = 76
DISPLAY_SIMPLIFY_PX = 1.0  # допуск упрощения контуров для карты: 1 пиксель = 20 м


# ---------------------------------------------------------------------------
# вспомогательные функции
# ---------------------------------------------------------------------------


def _read(path):
    import rasterio

    with rasterio.open(path) as src:
        return src.read(), src.transform, src.crs


def _nz(v):
    """NaN/пустые значения meta.csv -> None."""
    if v is None:
        return None
    if isinstance(v, float) and not np.isfinite(v):
        return None
    if isinstance(v, str) and v.strip().lower() in ("", "nan", "none"):
        return None
    return v


def _iso_utc(s: str) -> str:
    dt = datetime.fromisoformat(str(s))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _corners_ll(epsg: int, x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    """Углы чипа (TL, TR, BR, BL) в WGS84 — для наложения растров в MapLibre."""
    from pyproj import Transformer

    tr = Transformer.from_crs(int(epsg), 4326, always_xy=True)
    xs = [x0, x1, x1, x0]
    ys = [y1, y1, y0, y0]
    lon, lat = tr.transform(xs, ys)
    return [[round(float(a), 7), round(float(b), 7)] for a, b in zip(lon, lat)]


def _footprint_ll(epsg: int, x0, y0, x1, y1, n=16) -> list[list[float]]:
    """Контур чипа в WGS84 (стороны уплотнены — в WGS84 они слегка изогнуты)."""
    from pyproj import Transformer

    tr = Transformer.from_crs(int(epsg), 4326, always_xy=True)
    t = np.linspace(0, 1, n, endpoint=False)
    xs = np.concatenate([x0 + (x1 - x0) * t, np.full(n, x1), x1 - (x1 - x0) * t, np.full(n, x0), [x0]])
    ys = np.concatenate([np.full(n, y1), y1 - (y1 - y0) * t, np.full(n, y0), y0 + (y1 - y0) * t, [y1]])
    lon, lat = tr.transform(xs, ys)
    return [[round(float(a), 6), round(float(b), 6)] for a, b in zip(lon, lat)]


def _grid(meta_row, transform) -> tuple[float, float, float]:
    gsd = float(transform.a)
    if abs(transform.b) > 1e-9 or abs(transform.d) > 1e-9 or abs(transform.e + gsd) > 1e-6:
        raise ValueError(f"{meta_row['chip_id']}: ожидается сетка «север вверх» с квадратным пикселем")
    return float(transform.c), float(transform.f), gsd


def _load_pred(pred_dir: Path | None, kind: str, cid: str):
    if pred_dir is None:
        return None
    p = pred_dir / kind / f"{cid}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    mask = np.asarray(z["mask"]).astype(np.uint8)
    prob = np.asarray(z["prob"]).astype(np.float32) if "prob" in z.files else None
    return mask, prob


# ---------------------------------------------------------------------------
# обработка чипов (в отдельных процессах)
# ---------------------------------------------------------------------------


def process_af(task) -> dict:
    row, data_dir, pred_dir = task
    from pyproj import Transformer

    cid = row["chip_id"]
    viirs, tr, crs = _read(data_dir / "af" / "viirs" / f"{cid}_VIIRS_I1-I5.tif")
    aux, _, _ = _read(data_dir / "af" / "aux" / f"{cid}_AUX.tif")
    ref, _, _ = _read(data_dir / "af" / "masks" / f"{cid}_MASK.tif")
    x0, y0, gsd = _grid(row, tr)
    epsg = int(crs.to_epsg())
    to_ll = Transformer.from_crs(epsg, 4326, always_xy=True)
    out = {"chip_id": cid, "epsg": epsg, "x0": x0, "y0": y0, "gsd": gsd, "h": viirs.shape[1], "w": viirs.shape[2], "sources": {}}
    masks = {"reference": (ref[0].astype(np.uint8), None)}
    pred = _load_pred(pred_dir, "af", cid)
    if pred is not None:
        if pred[0].shape != ref[0].shape:
            raise ValueError(f"{cid}: размер предсказания {pred[0].shape} != {ref[0].shape}")
        masks["model"] = pred
    for src, (m, prob) in masks.items():
        hs = extract_hotspots(m, viirs, aux, prob)
        x = x0 + (hs["col"] + 0.5) * gsd
        y = y0 - (hs["row"] + 0.5) * gsd
        lon, lat = to_ll.transform(x, y)
        hs["lon"] = np.asarray(lon)
        hs["lat"] = np.asarray(lat)
        out["sources"][src] = hs
    out["valid_frac"] = float(np.nanmean(viirs[7] > 0)) if viirs.shape[0] > 7 else None
    return out


def _quicklook(pre: np.ndarray, post: np.ndarray, out_dir: Path, cid: str) -> dict:
    """Превью Sentinel-2 в псевдоцветах SWIR2-NIR-Red (B12, B8A, B4), общий контраст для pre/post."""
    from PIL import Image

    idx = [C_S2["B12"], C_S2["B8A"], C_S2["B4"]]
    a = pre[idx].astype(np.float32)
    b = post[idx].astype(np.float32)
    res = {}
    lo, hi = [], []
    for i in range(3):
        v = np.concatenate([a[i][a[i] > 0], b[i][b[i] > 0]])
        if v.size == 0:
            lo.append(0.0)
            hi.append(1.0)
        else:
            lo.append(float(np.percentile(v, 1.5)))
            hi.append(float(np.percentile(v, 98.5)))
    for name, arr in (("pre", a), ("post", b)):
        rgb = np.zeros(arr.shape[1:] + (3,), np.uint8)
        for i in range(3):
            s = np.clip((arr[i] - lo[i]) / max(hi[i] - lo[i], 1e-6), 0, 1) ** (1 / 1.15)
            rgb[..., i] = (s * 255 + 0.5).astype(np.uint8)
        im = Image.fromarray(rgb, "RGB")
        if QL_SIZE and QL_SIZE < max(im.size):
            im = im.resize((QL_SIZE, QL_SIZE), Image.LANCZOS)
        fn = f"{cid}_{name}.jpg"
        im.save(out_dir / fn, "JPEG", quality=QL_QUALITY, optimize=True, progressive=True)
        res[name] = f"quicklooks/{fn}"
    return res


C_S2 = {"B2": 0, "B3": 1, "B4": 2, "B5": 3, "B6": 4, "B7": 5, "B8A": 6, "B11": 7, "B12": 8, "SCL": 9}


def _overlay_png(mask: np.ndarray, path: Path) -> None:
    from PIL import Image

    im = Image.fromarray(np.clip(mask, 0, 3).astype(np.uint8), "P")
    pal = [0, 0, 0]
    for s in (1, 2, 3):
        h = C.SEVERITY[s]["color"].lstrip("#")
        pal += [int(h[i : i + 2], 16) for i in (0, 2, 4)]
    im.putpalette(pal + [0, 0, 0] * (256 - 4))
    im.save(path, "PNG", optimize=True, transparency=0)


def process_bs(task) -> dict:
    row, data_dir, pred_dir, tmp_out, make_ql = task
    cid = row["chip_id"]
    ref, tr, crs = _read(data_dir / "bs" / "masks" / f"{cid}_MASK.tif")
    aux, _, _ = _read(data_dir / "bs" / "aux" / f"{cid}_AUX.tif")
    x0, y0, gsd = _grid(row, tr)
    epsg = int(crs.to_epsg())
    h, w = ref.shape[1:]
    lc = aux[2] if aux.shape[0] >= 3 else None
    out = {"chip_id": cid, "epsg": epsg, "x0": x0, "y0": y0, "gsd": gsd, "h": h, "w": w, "sources": {}}
    masks = {"reference": (ref[0].astype(np.uint8), None)}
    pred = _load_pred(pred_dir, "bs", cid)
    if pred is not None:
        if pred[0].shape != ref[0].shape:
            raise ValueError(f"{cid}: размер предсказания {pred[0].shape} != {ref[0].shape}")
        masks["model"] = pred
    for src, (m, prob) in masks.items():
        m = m.copy()
        m[m > 3] = 0
        v = vectorize_severity(m, prob, lc, simplify_px=DISPLAY_SIMPLIFY_PX)
        # проверка: площадь каждого точного полигона == число пикселей
        for s, g in zip(v["scars"], v["scar_geoms"]):
            if not g.is_valid or abs(g.area - s["n_px"]) > 1e-6:
                raise AssertionError(f"{cid}/{src}: некорректная геометрия контура {s['k']}")
        for z, g in zip(v["zones"], v["zone_geoms"]):
            if not g.is_valid or abs(g.area - z["n_px"]) > 1e-6:
                raise AssertionError(f"{cid}/{src}: некорректная геометрия зоны {z}")
        # bbox и центроид каждого контура в WGS84
        out["sources"][src] = {
            "scars": v["scars"],
            "zones": v["zones"],
            "enc_scar": C.encode_pixel_geoms(v["scar_geoms"]),
            "enc_disp": C.encode_pixel_geoms(v["scar_disp"]),
            "enc_zone": C.encode_pixel_geoms(v["zone_geoms"]),
            "mask": m,
            "sev_px": [int((m == s).sum()) for s in (1, 2, 3)],
        }
        _overlay_png(m, tmp_out / src / "overlays" / f"{cid}.png")
    if make_ql:
        pre, _, _ = _read(data_dir / "bs" / "sentinel2_pre" / f"{cid}_Sentinel-2_pre.tif")
        post, _, _ = _read(data_dir / "bs" / "sentinel2_post" / f"{cid}_Sentinel-2_post.tif")
        out["quicklook"] = _quicklook(pre, post, tmp_out / "quicklooks", cid)
    return out


# ---------------------------------------------------------------------------
# сборка
# ---------------------------------------------------------------------------


def _dump(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))


def _r(x, nd):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def build(args) -> None:
    import pandas as pd
    import shapely
    from pyproj import Transformer

    t_start = time.time()
    data_dir = Path(args.data_dir).expanduser().resolve()
    pred_dir = Path(args.pred_dir).expanduser().resolve() if args.pred_dir else None
    out_root = Path(args.out).expanduser().resolve()
    final = out_root / "catalog"
    tmp = out_root / "catalog.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    sources = ["reference"] + (["model"] if pred_dir else [])
    for s in sources:
        (tmp / s / "overlays").mkdir(parents=True, exist_ok=True)
    (tmp / "quicklooks").mkdir(parents=True, exist_ok=True)

    af_meta = pd.read_csv(data_dir / "af" / "meta.csv")
    bs_meta = pd.read_csv(data_dir / "bs" / "meta.csv")
    if args.limit:
        af_meta = af_meta.head(args.limit)
        bs_meta = bs_meta.head(args.limit)
    af_rows = af_meta.to_dict("records")
    bs_rows = bs_meta.to_dict("records")
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    print(f"[build] AF: {len(af_rows)} чипов, BS: {len(bs_rows)} чипов, источники: {sources}, процессов: {workers}")

    with ProcessPoolExecutor(workers) as ex:
        af_res = list(ex.map(process_af, [(r, data_dir, pred_dir) for r in af_rows], chunksize=4))
        print(f"[build] AF готово за {time.time() - t_start:.1f} с")
        bs_res = list(
            ex.map(process_bs, [(r, data_dir, pred_dir, tmp, not args.no_quicklooks) for r in bs_rows], chunksize=2)
        )
        print(f"[build] BS готово за {time.time() - t_start:.1f} с")

    checks: dict = {"af": [], "bs": []}

    # ---------------- термоточки ----------------
    af_by_id = {r["chip_id"]: r for r in af_rows}
    af_chips = []
    for res in af_res:
        r = af_by_id[res["chip_id"]]
        af_chips.append(
            {
                "chip_id": res["chip_id"],
                "epsg": res["epsg"],
                "acq_datetime": _iso_utc(r["acq_datetime"]),
                "satellite": r["satellite"],
                "bounds_utm": [res["x0"], res["y0"] - res["gsd"] * res["h"], res["x0"] + res["gsd"] * res["w"], res["y0"]],
                "n_fire_px_meta": None if _nz(r.get("n_fire_px")) is None else int(r["n_fire_px"]),
                "has_model": "model" in res["sources"],
            }
        )
        n_ref = len(res["sources"]["reference"]["row"])
        if _nz(r.get("n_fire_px")) is not None and int(r["n_fire_px"]) != n_ref:
            checks["af"].append(f"{res['chip_id']}: n_fire_px={r['n_fire_px']} != {n_ref}")

    for src in sources:
        cols = {k: [] for k in ("id", "chip", "row", "col", "lon", "lat", "t", "sat", "dn", "i4", "i5", "sza", "vza", "lc", "conf", "epsg")}
        for res, chip in zip(af_res, af_chips):
            hs = res["sources"].get(src)
            if hs is None:
                continue
            n = len(hs["row"])
            cid = res["chip_id"]
            cols["id"] += [f"{cid}-r{rr:03d}c{cc:03d}" for rr, cc in zip(hs["row"].tolist(), hs["col"].tolist())]
            cols["chip"] += [cid] * n
            cols["row"] += hs["row"].tolist()
            cols["col"] += hs["col"].tolist()
            cols["lon"] += [round(float(v), 6) for v in hs["lon"]]
            cols["lat"] += [round(float(v), 6) for v in hs["lat"]]
            cols["t"] += [chip["acq_datetime"]] * n
            cols["sat"] += [chip["satellite"]] * n
            cols["dn"] += hs["daynight"].tolist()
            cols["i4"] += [_r(v, 2) for v in hs["i4"]]
            cols["i5"] += [_r(v, 2) for v in hs["i5"]]
            cols["sza"] += [_r(v, 2) for v in hs["sza"]]
            cols["vza"] += [_r(v, 2) for v in hs["vza"]]
            cols["lc"] += hs["lc"].astype(int).tolist()
            cols["conf"] += [round(float(v), 4) for v in hs["conf"]]
            cols["epsg"] += [res["epsg"]] * n
        _dump(cols, tmp / src / "hotspots.json")
        print(f"[build] {src}: термоточек {len(cols['id'])}")

    # ---------------- гари ----------------
    bs_by_id = {r["chip_id"]: r for r in bs_rows}
    bs_chips = []
    for res in bs_res:
        r = bs_by_id[res["chip_id"]]
        x0, y0, gsd, h, w, epsg = res["x0"], res["y0"], res["gsd"], res["h"], res["w"], res["epsg"]
        chip = {
            "chip_id": res["chip_id"],
            "fire_event_id": _nz(r.get("fire_event_id")),
            "epsg": epsg,
            "x0": x0,
            "y0": y0,
            "gsd": gsd,
            "width": w,
            "height": h,
            "bounds_utm": [x0, y0 - gsd * h, x0 + gsd * w, y0],
            "date_pre": _nz(r.get("date_pre")),
            "date_post": _nz(r.get("date_post")),
            "s1_date_pre": _nz(r.get("s1_date_pre")),
            "s1_date_post": _nz(r.get("s1_date_post")),
            "valid_frac": _nz(r.get("valid_frac")),
            "cloud_frac": _nz(r.get("cloud_frac")),
            "burn_area_ha_meta": _nz(r.get("burn_area_ha")),
            "sev_px_meta": [None if _nz(r.get(f"sev{s}_px")) is None else int(r[f"sev{s}_px"]) for s in (1, 2, 3)],
            "corners": _corners_ll(epsg, x0, y0 - gsd * h, x0 + gsd * w, y0),
            "footprint": _footprint_ll(epsg, x0, y0 - gsd * h, x0 + gsd * w, y0),
            "quicklook": res.get("quicklook"),
            "overlays": {s: f"{s}/overlays/{res['chip_id']}.png" for s in res["sources"]},
            "sources": sorted(res["sources"]),
        }
        bs_chips.append(chip)
        ref = res["sources"]["reference"]
        ref_ha = sum(s["n_px"] for s in ref["scars"]) * gsd * gsd / 1e4
        if chip["burn_area_ha_meta"] is not None and abs(ref_ha - float(chip["burn_area_ha_meta"])) > 0.011:
            checks["bs"].append(f"{chip['chip_id']}: burn_area_ha={chip['burn_area_ha_meta']} != {ref_ha:.2f}")
        for s in range(3):
            mv = chip["sev_px_meta"][s]
            if mv is not None and mv != ref["sev_px"][s]:
                checks["bs"].append(f"{chip['chip_id']}: sev{s + 1}_px={mv} != {ref['sev_px'][s]}")

    for src in sources:
        scars, zones, enc_s, enc_d, enc_z, chip_idx_s, chip_idx_z = [], [], [], [], [], [], []
        masks = {}
        for ci, (res, chip) in enumerate(zip(bs_res, bs_chips)):
            sd = res["sources"].get(src)
            if sd is None:
                continue
            masks[chip["chip_id"]] = sd["mask"]
            prefix = "R" if src == "reference" else "M"
            to_ll = Transformer.from_crs(chip["epsg"], 4326, always_xy=True)
            # центроиды и bbox контуров (по упрощённой геометрии достаточно для навигации)
            disp = C.decode_geoms(
                sd["enc_disp"],
                np.full(len(sd["scars"]), chip["x0"]),
                np.full(len(sd["scars"]), chip["y0"]),
                np.full(len(sd["scars"]), chip["gsd"]),
            )
            for s, g in zip(sd["scars"], disp):
                sid = f"{chip['chip_id']}-{prefix}{s['k']:03d}"
                b = shapely.bounds(g)
                cx, cy = shapely.get_coordinates(shapely.point_on_surface(g))[0]
                bx, by = to_ll.transform([b[0], b[2], b[0], b[2]], [b[1], b[3], b[3], b[1]])
                clon, clat = to_ll.transform(cx, cy)
                scars.append(
                    {
                        "scar_id": sid,
                        "chip": ci,
                        "k": s["k"],
                        "n_px": s["n_px"],
                        "sev_px": s["sev_px"],
                        "prob_mean": None if s["prob_mean"] is None else round(s["prob_mean"], 4),
                        "lc_px": s["lc_px"],
                        "bbox_ll": [round(min(bx), 6), round(min(by), 6), round(max(bx), 6), round(max(by), 6)],
                        "center_ll": [round(float(clon), 6), round(float(clat), 6)],
                    }
                )
            base = len(scars) - len(sd["scars"])
            for z in sd["zones"]:
                zones.append({"scar": base + z["scar_k"] - 1, "severity": z["severity"], "n_px": z["n_px"]})
            enc_s.append(sd["enc_scar"])
            enc_d.append(sd["enc_disp"])
            enc_z.append(sd["enc_zone"])
            chip_idx_s += [ci] * len(sd["scars"])
            chip_idx_z += [ci] * len(sd["zones"])
        es, ed, ez = C.concat_encoded(enc_s), C.concat_encoded(enc_d), C.concat_encoded(enc_z)
        np.savez_compressed(
            tmp / src / "geometry.npz",
            **{f"scar_{k}": v for k, v in es.items()},
            **{f"disp_{k}": v for k, v in ed.items()},
            **{f"zone_{k}": v for k, v in ez.items()},
            scar_chip=np.asarray(chip_idx_s, np.int32),
            zone_chip=np.asarray(chip_idx_z, np.int32),
        )
        np.savez_compressed(tmp / src / "masks.npz", **masks)
        _dump({"scars": scars, "zones": zones}, tmp / src / "scars.json")
        tot = sum(s["n_px"] for s in scars) * C.BS_GSD**2 / 1e4
        print(f"[build] {src}: контуров {len(scars)}, зон {len(zones)}, площадь гарей {tot:.2f} га")

    # ---------------- контроль по территории мониторинга ----------------
    aoi_path = Path(args.aoi) if args.aoi else out_root / "aoi.geojson"
    outside = []
    if aoi_path.exists():
        with open(aoi_path, encoding="utf-8") as f:
            aoi = json.load(f)
        feat = next((ft for ft in aoi["features"] if ft.get("id") == "aoi"), aoi["features"][0])
        aoi_geom = shapely.geometry.shape(feat["geometry"])
        for ch in bs_chips:
            if not aoi_geom.contains(shapely.Polygon(ch["footprint"])):
                outside.append(ch["chip_id"])
    print(f"[build] BS-чипов вне/на границе AOI: {len(outside)}")

    missing = {}
    if pred_dir is not None:
        missing = {
            "af": [r["chip_id"] for r in af_res if "model" not in r["sources"]],
            "bs": [r["chip_id"] for r in bs_res if "model" not in r["sources"]],
        }
        if missing["af"] or missing["bs"]:
            print(f"[build] ВНИМАНИЕ: нет предсказаний модели для AF {len(missing['af'])}, BS {len(missing['bs'])} чипов")
    meta = {
        "catalog_version": C.CATALOG_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_dir": data_dir.name,
        "pred_dir": pred_dir.name if pred_dir else None,
        "sources": sources,
        "display_simplify_m": DISPLAY_SIMPLIFY_PX * C.BS_GSD,
        "quicklook": {"bands": ["B12", "B8A", "B4"], "size_px": QL_SIZE, "enabled": not args.no_quicklooks},
        "checks": {
            "af_mismatch": checks["af"],
            "bs_mismatch": checks["bs"],
            "bs_outside_aoi": outside,
            "model_missing": missing,
        },
        "build_seconds": round(time.time() - t_start, 1),
    }
    _dump({"af": af_chips, "bs": bs_chips}, tmp / "chips.json")
    _dump(meta, tmp / "meta.json")

    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)
    size = sum(p.stat().st_size for p in final.rglob("*") if p.is_file()) / 1e6
    print(f"[build] каталог: {final} ({size:.1f} МБ) за {time.time() - t_start:.1f} с")
    if checks["af"] or checks["bs"]:
        print(f"[build] ВНИМАНИЕ: расхождения с meta.csv: AF {len(checks['af'])}, BS {len(checks['bs'])}")
        for m in (checks["af"] + checks["bs"])[:20]:
            print("   ", m)
    else:
        print("[build] контроль: число пикселей и площади совпадают с meta.csv")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Сборка каталога сервиса мониторинга пожаров")
    ap.add_argument("--data-dir", required=True, help="каталог train (af/, bs/ с meta.csv)")
    ap.add_argument("--pred-dir", default=None, help="каталог предсказаний модели (af/*.npz, bs/*.npz)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "data"), help="куда писать (по умолчанию service/data)")
    ap.add_argument("--aoi", default=None, help="GeoJSON границы территории (по умолчанию <out>/aoi.geojson)")
    ap.add_argument("--workers", type=int, default=0, help="число процессов")
    ap.add_argument("--no-quicklooks", action="store_true", help="не строить превью Sentinel-2")
    ap.add_argument("--limit", type=int, default=0, help="взять первые N чипов каждого типа (отладка)")
    build(ap.parse_args(argv))


if __name__ == "__main__":
    main()
