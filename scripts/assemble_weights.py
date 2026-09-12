"""Сборка каталога весов из обученных моделей (то же делает train.py в конце).

python scripts/assemble_weights.py --af-dir runs/af --gbdt runs/bs_gbdt_s1.txt \
    --cnn ctx_s2_s42=runs/full_ctx_s2_e150_s42/f-1/model.pt ... --out weights

Сети переводятся в fp16 (вдвое меньше файл), bs_config.json строится по
configs/bs_ensemble.json (веса бленда, порог, TTA).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import to_fp16  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--af-dir", required=True, help="каталог с af_config.json и af_lgbm_s*.txt")
    ap.add_argument("--gbdt", required=True, help="файл модели GBDT области (bs_gbdt_s1.txt)")
    ap.add_argument("--cnn", nargs="+", required=True, help="имя=путь к model.pt для каждой сети ансамбля")
    ap.add_argument("--out", default=str(REPO / "weights"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    af = Path(a.af_dir)
    cfg_af = json.loads((af / "af_config.json").read_text())
    for f in cfg_af["model_files"] + ["af_config.json"] + (["af_cv_report.json"] if (af / "af_cv_report.json").exists() else []):
        shutil.copy(af / f, out / f)
    g = Path(a.gbdt)
    shutil.copy(g, out / "bs_gbdt_s1.txt")
    if g.with_suffix(".json").exists():
        shutil.copy(g.with_suffix(".json"), out / "bs_gbdt_s1.json")
    ens = json.loads((REPO / "configs" / "bs_ensemble.json").read_text())
    paths = dict(x.split("=", 1) for x in a.cnn)
    specs = []
    for m in ens["cnn"]:
        fn = f"bs_{m['name']}.pt"
        to_fp16(Path(paths[m["name"]]), out / fn)
        specs.append({"file": fn, "group": m["name"]})
    bs_cfg = {"cnn": specs, "threshold": ens["blend"]["threshold"], "gate": True, "cloud": True, "tta": ens.get("tta", 4),
              "gbdt": {"block": ens["gbdt"]["block"], "s1": ["bs_gbdt_s1.txt"], "sparse": bool(ens["gbdt"].get("sparse", False))},
              "blend_sources": ens["blend"]["sources"], "blend_weights": ens["blend"]["weights"], "final": "blend"}
    bs_cfg["severity_mode"] = ens.get("severity_mode", "legacy")
    bs_cfg["min_area_px"] = int(ens.get("min_area_px", 0))
    (out / "bs_config.json").write_text(json.dumps(bs_cfg, indent=1, ensure_ascii=False))
    total = sum(p.stat().st_size for p in out.iterdir()) / 2**20
    print(f"weights in {out}: {sorted(p.name for p in out.iterdir())}, {total:.1f} MB")


if __name__ == "__main__":
    main()
