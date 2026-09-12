#!/usr/bin/env python
"""Обучение решения команды LitWin (оба модуля) одной командой.

    python train.py --data-dir /path/to/train                # финальные веса в ./weights
    python train.py --data-dir /path/to/train --only bs      # только модуль гарей
    python train.py --data-dir /path/to/train --cv           # + 5-fold CV для отчёта

Состав ансамбля модуля гарей (сети, GBDT, веса бленда, порог) —
configs/bs_ensemble.json; веса и порог подобраны на OOF скриптом
scripts/bs_blend_search.py (с --cv он перезапускается на свежих OOF). Фолды — configs/folds_{af,bs}.csv (фиксированы,
пересоздаются scripts/make_folds.py). Все случайные начальные значения
закреплены.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO)


def to_fp16(src: Path, dst: Path) -> None:
    """Веса в fp16 — вдвое меньше файл, точность инференса не меняется
    (инференс всё равно идёт в bfloat16/float16 autocast)."""
    ck = torch.load(src, map_location="cpu", weights_only=False)
    ck["state_dict"] = {k: (v.half() if v.is_floating_point() else v) for k, v in ck["state_dict"].items()}
    ck.pop("args", None)
    torch.save(ck, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="каталог train (af/, bs/ с meta.csv)")
    ap.add_argument("--out", default=str(REPO / "weights"))
    ap.add_argument("--work-dir", default=str(REPO / "runs"), help="промежуточные чекпойнты и OOF")
    ap.add_argument("--only", choices=["af", "bs"], default=None)
    ap.add_argument("--cv", action="store_true", help="дополнительно 5-fold CV модуля гарей")
    a = ap.parse_args()
    out, work = Path(a.out), Path(a.work_dir)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    if a.only in (None, "af"):
        run([py, "-m", "firemon.af.train", "--data-dir", a.data_dir, "--out", str(out), "--oof-dir", str(work / "af_oof")])

    if a.only in (None, "bs"):
        cfg = json.loads((REPO / "configs" / "bs_ensemble.json").read_text())
        cnn_specs = []
        for m in cfg["cnn"]:
            args = [f"--{k}={v}" for k, v in m["args"].items() if not isinstance(v, bool)]
            args += [f"--{k}" for k, v in m["args"].items() if v is True]
            if a.cv:
                for f in range(5):
                    run([py, "-m", "firemon.bs.train", "--data-dir", a.data_dir, "--fold", str(f),
                         "--out", str(work / "cv" / m["name"] / f"f{f}"), *args])
            run([py, "-m", "firemon.bs.train", "--data-dir", a.data_dir, "--fold", "-1",
                 "--out", str(work / "full" / m["name"]), *args])
            fn = f"bs_{m['name']}.pt"
            to_fp16(work / "full" / m["name"] / "model.pt", out / fn)
            cnn_specs.append({"file": fn, "group": m["name"]})
        g = cfg["gbdt"]
        run([py, "scripts/bs_gbdt_cv.py", "--data-dir", a.data_dir, "--out", str(work / "cv" / "gbdt_s1"),
             "--block", str(g["block"]), "--rounds", str(g["rounds"]), "--save-final", str(out), "--name", "bs_gbdt_s1.txt",
             *(["--sparse"] if g.get("sparse") else [])])
        blend = cfg["blend"]
        bs_cfg = {"cnn": cnn_specs, "threshold": blend["threshold"], "gate": True, "cloud": True, "tta": cfg.get("tta", 4),
                  "gbdt": {"block": g["block"], "s1": ["bs_gbdt_s1.txt"], "sparse": bool(g.get("sparse", False))},
                  "blend_sources": blend["sources"], "blend_weights": blend["weights"], "final": "blend"}
        bs_cfg["severity_mode"] = cfg.get("severity_mode", "legacy")
        bs_cfg["min_area_px"] = int(cfg.get("min_area_px", 0))
        (out / "bs_config.json").write_text(json.dumps(bs_cfg, indent=1))
        if a.cv:  # воспроизведение подбора весов бленда по OOF
            runs = [str(work / "cv" / ("gbdt_s1" if s_ == "gbdt_s1" else s_)) for s_ in blend["sources"]]
            run([py, "scripts/bs_blend_search.py", "--data-dir", a.data_dir, "--runs", *runs,
                 "--configs", ",".join(str(w) for w in blend["weights"]),
                 "--severity-mode", cfg.get("severity_mode", "legacy")])
    print("done; weights in", out)


if __name__ == "__main__":
    main()
