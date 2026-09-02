"""Проверка submission.csv по формальным требованиям постановки и (опционально)
подсчёт метрики по эталонным маскам.

    python scripts/validate_submission.py --submission submission.csv --data-dir /path/to/test
    python scripts/validate_submission.py --submission sub.csv --data-dir /path/to/pseudo_test --gt-dir /path/to/masks_root

Проверки: набор пар (chip_id, class_id) совпадает с шаблоном, нет пропусков/NaN,
RLE декодируется, серии упорядочены и не касаются, не выходят за чип, маски
разных классов одного чипа не пересекаются.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from firemon.metric import ScoreAccumulator  # noqa: E402
from firemon.rle import rle_decode  # noqa: E402


def chip_shape(data_dir: Path, cid: str) -> tuple[int, int]:
    af = data_dir / "af" / "viirs" / f"{cid}_VIIRS_I1-I5.tif"
    p = af if af.exists() else data_dir / "bs" / "sentinel2_post" / f"{cid}_Sentinel-2_post.tif"
    with rasterio.open(p) as s:
        return s.height, s.width


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--gt-dir", default=None, help="корень с af/masks и bs/masks для подсчёта метрики")
    a = ap.parse_args()
    data_dir = Path(a.data_dir)
    sub = read_rows(Path(a.submission))
    tmpl = read_rows(data_dir / "sample_submission.csv")
    errors: list[str] = []

    if not sub or list(sub[0].keys()) != ["chip_id", "class_id", "rle"]:
        errors.append(f"bad header: {list(sub[0].keys()) if sub else 'empty'}")
    pairs = [(r["chip_id"], r["class_id"]) for r in sub]
    tpairs = {(r["chip_id"], r["class_id"]) for r in tmpl}
    if len(pairs) != len(set(pairs)):
        errors.append("duplicate (chip_id, class_id) pairs")
    if set(pairs) != tpairs:
        errors.append(f"pair set mismatch: missing {len(tpairs - set(pairs))}, extra {len(set(pairs) - tpairs)}")
    if len(sub) != len(tmpl):
        errors.append(f"row count {len(sub)} != template {len(tmpl)}")

    shapes: dict[str, tuple[int, int]] = {}
    per_chip: dict[str, dict[int, np.ndarray]] = {}
    for r in sub:
        cid, cls, rle = r["chip_id"], r["class_id"], r["rle"]
        if rle is None or cid in (None, "") or cls in (None, ""):
            errors.append(f"{cid},{cls}: missing value")
            continue
        if rle.strip().lower() in {"nan", "none", "null"}:
            errors.append(f"{cid},{cls}: NaN-like rle")
            continue
        if cid not in shapes:
            shapes[cid] = chip_shape(data_dir, cid)
        try:
            m = rle_decode(rle, shapes[cid])
        except ValueError as e:
            errors.append(f"{cid},{cls}: {e}")
            continue
        per_chip.setdefault(cid, {})[int(cls)] = m
    for cid, d in per_chip.items():
        if len(d) > 1 and np.sum([m.astype(np.uint8) for m in d.values()], 0).max() > 1:
            errors.append(f"{cid}: class masks overlap")

    # сырое содержимое: rle должен быть в кавычках (как в примере постановки)
    raw = Path(a.submission).read_text(encoding="utf-8").splitlines()[1:]
    unquoted = sum(1 for line in raw if not line.rstrip().endswith('"'))
    if unquoted:
        errors.append(f"{unquoted} rows without quoted rle")

    report = {"rows": len(sub), "chips": len(per_chip), "errors": len(errors)}
    if a.gt_dir:
        acc = ScoreAccumulator()
        gt = Path(a.gt_dir)
        for cid, d in per_chip.items():
            if cid.startswith("AF"):
                with rasterio.open(gt / "af" / "masks" / f"{cid}_MASK.tif") as s:
                    acc.add_af(d.get(1, np.zeros(shapes[cid], bool)).astype(np.uint8), s.read(1))
            else:
                pred = np.zeros(shapes[cid], np.uint8)
                for k, m in d.items():
                    pred[m] = k
                with rasterio.open(gt / "bs" / "masks" / f"{cid}_MASK.tif") as s:
                    acc.add_bs(pred, s.read(1))
        report["metric"] = {k: round(v, 5) for k, v in acc.result().items()}
    print(json.dumps(report, indent=1, ensure_ascii=False))
    for e in errors[:30]:
        print("ERROR:", e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
