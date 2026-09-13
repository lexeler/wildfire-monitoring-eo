"""Пересчёт OOF-вероятностей готовых фолдовых моделей с другим числом TTA
(оценка компромисса «качество — время инференса»).

python scripts/bs_oof_repredict.py --data-dir data/train --run runs/NAME --tta 4 --out runs/NAME_tta4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from firemon.bs.model import build_model  # noqa: E402
from firemon.bs.train import batch_features, load_arrays, predict_probs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tta", type=int, default=4)
    ap.add_argument("--folds-file", default=str(Path(__file__).resolve().parents[1] / "configs" / "folds_bs.csv"))
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = pd.read_csv(Path(a.data_dir) / "bs" / "meta.csv")
    folds = pd.read_csv(a.folds_file).set_index("chip_id").loc[meta.chip_id, "fold"].values
    arr = load_arrays(a.data_dir, list(meta.chip_id))
    for f in sorted(set(folds)):
        ck = torch.load(Path(a.run) / f"f{f}" / "model.pt", map_location="cpu", weights_only=False)
        m = build_model(ck["arch"], ck["encoder"], pretrained=False, classes=ck.get("classes", 4), ctx=int(ck.get("ctx", 0)),
                        coord=bool(ck.get("coord", False)), dec_attn=ck.get("dec_attn"), hot=bool(ck.get("hot", False)))
        m.load_state_dict(ck["state_dict"])
        m.scale = ck.get("scale", 1)
        m.eval().to(dev)
        ctx, hot = int(ck.get("ctx", 0)), bool(ck.get("hot", False))
        out = Path(a.out) / f"f{f}" / "oof"
        out.mkdir(parents=True, exist_ok=True)
        va = np.where(folds == f)[0]
        for b in range(0, len(va), 4):
            idx = va[b : b + 4]
            X, _, _, _ = batch_features(arr, idx, dev, ctx, hot)
            p = predict_probs(m, X, a.tta).cpu().numpy()
            for i, j in enumerate(idx):
                np.save(out / f"{meta.chip_id[j]}.npy", (p[i] * 255).round().astype(np.uint8))
        print("fold", f, "done", flush=True)


if __name__ == "__main__":
    main()
