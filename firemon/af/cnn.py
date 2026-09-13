"""Сверточная модель активного горения поверх контекстных карт признаков.

Вход — те же 117 карт признаков, что у бустинга (features.compute_features),
нормированные по статистике train; выход — вероятность горения на пиксель.
Используется в ансамбле с LightGBM: сеть видит пространственную форму
кластера горящих пикселей, бустинг — точные пороговые соотношения.

ОТВЕРГНУТЫЙ ЭКСПЕРИМЕНТ (журнал, строка A8): одна сеть F1 0.899, бленд с
бустингом не лучше бустинга (0.962). inference.py и train.py модуль не вызывают;
оставлен для воспроизводимости отчёта.

python -m firemon.af.cnn --data-dir data/train --fold 0 --out runs/af_cnn
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F

from ..io import load_af, parallel_map
from ..utils import seed_everything
from .features import compute_features

REPO = Path(__file__).resolve().parents[2]


def build(n_in: int, encoder: str = "resnet18", pretrained: bool = False) -> torch.nn.Module:
    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                    in_channels=n_in, classes=1, encoder_depth=4, decoder_channels=(128, 64, 32, 16))


def normalize(X: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.nan_to_num(((X - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)).clamp(-8, 8))


def dihedral(x, k, f):
    x = torch.rot90(x, k, dims=(-2, -1))
    return torch.flip(x, dims=(-1,)) if f else x


def inv_dihedral(x, k, f):
    if f:
        x = torch.flip(x, dims=(-1,))
    return torch.rot90(x, -k, dims=(-2, -1))


@torch.no_grad()
def predict(model, Xn: torch.Tensor, tta: int = 4) -> torch.Tensor:
    combos = [(0, False), (0, True), (2, False), (2, True), (1, False), (3, False), (1, True), (3, True)][:tta]
    acc = 0
    for k, f in combos:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=Xn.is_cuda):
            out = model(dihedral(Xn, k, f)).float()
        acc = acc + inv_dihedral(torch.sigmoid(out), k, f)
    return (acc / len(combos))[:, 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--folds-file", default=str(REPO / "configs" / "folds_af.csv"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    seed_everything(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(Path(a.data_dir) / "af" / "meta.csv")
    folds = pd.read_csv(a.folds_file).set_index("chip_id").loc[meta.chip_id, "fold"].values
    tr = np.where(folds != a.fold)[0] if a.fold >= 0 else np.arange(len(meta))
    va = np.where(folds == a.fold)[0] if a.fold >= 0 else np.array([], int)
    t0 = time.time()
    chips = parallel_map(lambda c: load_af(a.data_dir, c, with_mask=True), meta.chip_id)
    V = np.stack([c.viirs for c in chips])
    A = np.stack([c.aux for c in chips])
    M = torch.from_numpy((np.stack([c.mask for c in chips]) == 1).astype(np.float32))
    feats, cands = [], []
    for s in range(0, len(V), 64):
        X, C, names = compute_features(V[s : s + 64], A[s : s + 64], device=str(dev))
        feats.append(X.float().cpu())
        cands.append(C.cpu())
    X = torch.nan_to_num(torch.cat(feats), nan=0.0, posinf=0.0, neginf=0.0)
    C = torch.cat(cands)
    Xtr = X[tr].float()
    mean = Xtr.mean((0, 2, 3))
    std = Xtr.std((0, 2, 3)).clamp_min(1e-3)
    mean, std = torch.nan_to_num(mean), torch.nan_to_num(std, nan=1.0).clamp_min(1e-3)
    del Xtr
    print(f"features {X.shape} in {time.time() - t0:.0f}s", flush=True)

    model = build(X.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = int(np.ceil(len(tr) / a.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps, pct_start=0.05)
    rng = np.random.default_rng(a.seed)
    m_dev, s_dev = mean.to(dev), std.to(dev)
    for ep in range(a.epochs):
        model.train()
        perm = rng.permutation(tr)
        ls = []
        for b in range(steps):
            idx = np.sort(perm[b * a.bs : (b + 1) * a.bs])
            xb = normalize(X[idx].to(dev).float(), m_dev, s_dev)
            yb = M[idx].to(dev)[:, None]
            k, f = int(rng.integers(4)), bool(rng.integers(2))
            xb, yb = dihedral(xb, k, f).contiguous(), dihedral(yb, k, f).contiguous()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                z = model(xb).float()
            bce = F.binary_cross_entropy_with_logits(z, yb, pos_weight=torch.tensor(20.0, device=dev))
            p = torch.sigmoid(z)
            dice = 1 - (2 * (p * yb).sum() + 1) / (p.sum() + yb.sum() + 1)
            loss = bce + dice
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            ls.append(loss.item())
        if ep % 10 == 0 or ep == a.epochs - 1:
            print(f"ep {ep} loss {np.mean(ls):.4f} t {time.time() - t0:.0f}s", flush=True)
    torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std, "n_in": X.shape[1], "names": names}, out / "model.pt")
    if len(va):
        model.eval()
        (out / "oof").mkdir(exist_ok=True)
        for s in range(0, len(va), 32):
            idx = va[s : s + 32]
            p = predict(model, normalize(X[idx].to(dev).float(), m_dev, s_dev)).cpu().numpy()
            for i, j in enumerate(idx):
                np.save(out / "oof" / f"{meta.chip_id[j]}.npy", p[i].astype(np.float16))
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
