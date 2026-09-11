"""Обучение сети картирования гарей (одна модель на фолд или на всём train).

Пример:
  python -m firemon.bs.train --data-dir data/train --fold 0 --out runs/bs_r34
  python -m firemon.bs.train --data-dir data/train --fold -1 --out weights/bs   # весь train
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..io import load_bs, parallel_map
from ..metric import ScoreAccumulator
from ..utils import seed_everything
from .features import compute_features
from .model import build_model, init_stem_from_bands
from .rules import SCL_EXCLUDE

REPO = Path(__file__).resolve().parents[2]
# группы входных каналов для --chdrop (порядок каналов — features.compute_features):
# 44..63 SCL «до/после», 64..72 Sentinel-1, 73..82 рельеф и покров, 83..93 дальний контекст
CHANNEL_GROUPS = ((44, 64), (64, 73), (73, 83), (83, 94))


def load_arrays(data_dir: str, ids: list[str]) -> dict[str, np.ndarray]:
    chips = parallel_map(lambda c: load_bs(data_dir, c, with_mask=True), ids)
    return {
        "s2_pre": np.stack([c.s2_pre for c in chips]),
        "s2_post": np.stack([c.s2_post for c in chips]),
        "s1_pre": np.stack([c.s1_pre for c in chips]),
        "s1_post": np.stack([c.s1_post for c in chips]),
        "aux": np.stack([c.aux for c in chips]),
        "mask": np.stack([c.mask for c in chips]).astype(np.int64),
    }


def exclusion(scl_pre: torch.Tensor, scl_post: torch.Tensor) -> torch.Tensor:
    """Пиксели, которые в эталоне всегда «не горело» (см. rules.SCL_EXCLUDE)."""
    ex = torch.tensor(SCL_EXCLUDE, device=scl_pre.device)
    return torch.isin(scl_pre.long(), ex) | torch.isin(scl_post.long(), ex)


def batch_features(arr: dict, idx, device, ctx: int = 0, hot: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    t = {k: torch.from_numpy(arr[k][idx]).to(device, non_blocking=True) for k in ("s2_pre", "s2_post", "s1_pre", "s1_post", "aux")}
    with torch.no_grad():
        X, rule = compute_features(t["s2_pre"], t["s2_post"], t["s1_pre"].int(), t["s1_post"].int(), t["aux"].int(), ctx=ctx, hot=hot)
        excl = exclusion(t["s2_pre"][:, 9], t["s2_post"][:, 9])
    y = torch.from_numpy(arr["mask"][idx]).to(device)
    return X, rule, y, excl


def dihedral(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    x = torch.rot90(x, k, dims=(-2, -1))
    return torch.flip(x, dims=(-1,)) if flip else x


def inv_dihedral(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    if flip:
        x = torch.flip(x, dims=(-1,))
    return torch.rot90(x, -k, dims=(-2, -1))


def lovasz_hinge_flat(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Lovász hinge (Berman et al., 2018) — выпуклая суррогатная функция IoU по
    пулу пикселей: z — логиты, y — {0,1}, оба одномерные."""
    if z.numel() == 0:
        return z.sum() * 0.0
    signs = 2.0 * y - 1.0
    errors = 1.0 - z * signs
    errors_sorted, perm = torch.sort(errors, descending=True)
    gt = y[perm]
    gts = gt.sum()
    inter = gts - gt.cumsum(0)
    union = gts + (1.0 - gt).cumsum(0)
    jac = 1.0 - inter / union
    jac[1:] = jac[1:] - jac[:-1]
    return torch.dot(F.relu(errors_sorted), jac)


def region_loss(logits: torch.Tensor, y: torch.Tensor, rule: torch.Tensor, excl: torch.Tensor, dice_w: float,
                lovasz_w: float = 0.0, soft: torch.Tensor | None = None, soft_alpha: float = 1.0,
                smooth: float = 0.0, boot: float = 1.0, gce: float = 0.0) -> torch.Tensor:
    """Сеть предсказывает «область пожара»; эталон = область ∩ (dNBR выше
    порога) ∖ облака, поэтому учим только там, где правило допускает гарь.
    soft — OOF-вероятности ансамбля-учителя: цель = alpha·эталон + (1-alpha)·soft
    (граница области в эталоне во многом случайна — дуги буфера вокруг
    термоточек; мягкая цель не даёт сети заучивать этот шум)."""
    yb = (y > 0).float()
    if soft is not None:
        yb = soft_alpha * yb + (1.0 - soft_alpha) * soft
    w = (((rule > 0) | (y > 0)) & ~excl).float()
    z = logits[:, 0].float()
    if boot < 1.0:  # soft bootstrapping (Reed et al., 2015): цель = boot·эталон + (1 - boot)·собственный прогноз сети
        yb = boot * yb + (1.0 - boot) * torch.sigmoid(z).detach()
    if smooth > 0:  # сглаживание меток: граница области в эталоне случайна
        yb = yb * (1.0 - smooth) + 0.5 * smooth
    if gce > 0:  # обобщённая кросс-энтропия (Zhang & Sabuncu, 2018): L = (1 - p_t^q) / q, устойчива к шуму меток
        pt = torch.where(yb > 0.5, torch.sigmoid(z), torch.sigmoid(-z)).clamp_min(1e-6)
        pix = (1.0 - pt.pow(gce)) / gce
    else:
        pix = F.binary_cross_entropy_with_logits(z, yb, reduction="none")
    bce = (pix * w).sum() / w.sum().clamp_min(1.0)
    p = torch.sigmoid(z) * w
    inter = (p * yb).sum()
    union = p.sum() + (yb * w).sum() - inter
    loss = bce + dice_w * (1 - (inter + 1) / (union + 1))
    if lovasz_w > 0:
        sel = w > 0
        loss = loss + lovasz_w * lovasz_hinge_flat(z[sel], (y > 0).float()[sel])
    return loss


def chip_fraction_loss(aux_logit: torch.Tensor, y: torch.Tensor, rule: torch.Tensor, excl: torch.Tensor) -> torch.Tensor:
    """Вспомогательная цель уровня чипа: доля гари среди допустимых пикселей
    (заставляет энкодер оценивать «масштаб события» по всему чипу)."""
    w = (((rule > 0) | (y > 0)) & ~excl).float()
    frac = ((y > 0).float() * w).sum((1, 2)) / w.sum((1, 2)).clamp_min(1.0)
    return F.binary_cross_entropy_with_logits(aux_logit[:, 0].float(), frac)


def loss_fn(logits: torch.Tensor, y: torch.Tensor, dice_w: float) -> torch.Tensor:
    ce = F.cross_entropy(logits, y)
    p = logits.float().softmax(1)
    pb = 1 - p[:, 0]
    yb = (y > 0).float()
    inter = (pb * yb).sum()
    dice = 1 - (2 * inter + 1) / (pb.sum() + yb.sum() + 1)
    return ce + dice_w * dice


def forward_scaled(model, X: torch.Tensor, scale: int = 1) -> torch.Tensor:
    """scale > 1: сеть работает на входе, уменьшенном в scale раз (усреднение
    блоками), логиты билинейно возвращаются к исходному размеру."""
    if scale == 1:
        return model(X)
    out = model(F.avg_pool2d(X, scale))
    return F.interpolate(out.float(), size=X.shape[-2:], mode="bilinear", align_corners=False)


@torch.no_grad()
def predict_probs(model, X: torch.Tensor, tta: int = 1) -> torch.Tensor:
    """softmax (N, 4, H, W) для 4-классовой сети или sigmoid (N, 1, H, W) для
    сети области; усреднение по диэдральным TTA.

    Все варианты TTA идут через сеть одним батчем на уже уменьшенном входе:
    вход усредняется блоками один раз, а число запусков ядер GPU не растёт с
    числом TTA (на быстрых GPU инференс упирается именно в запуск ядер)."""
    combos = [(0, False), (0, True), (2, False), (2, True), (1, False), (3, False), (1, True), (3, True)][: max(1, tta)]
    scale = getattr(model, "scale", 1)
    size = X.shape[-2:]
    xs = F.avg_pool2d(X, scale) if scale > 1 else X
    n = len(xs)
    square = size[0] == size[1]
    # повороты на 90° меняют форму неквадратного чипа — такие варианты идут отдельным батчем
    groups = [combos] if square else [[c for c in combos if c[0] % 2 == 0], [c for c in combos if c[0] % 2 == 1]]
    acc = None
    for group in filter(None, groups):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=X.is_cuda):
            out = model(torch.cat([dihedral(xs, k, f) for k, f in group]))
        for i, (k, f) in enumerate(group):
            o = out[i * n : (i + 1) * n].float()
            if scale > 1:
                o = F.interpolate(o, size=(size[0], size[1]) if k % 2 == 0 else (size[1], size[0]), mode="bilinear", align_corners=False)
            o = o.sigmoid() if o.shape[1] == 1 else o.softmax(1)
            p = inv_dihedral(o, k, f)
            acc = p if acc is None else acc + p
    return acc / len(combos)


def evaluate(probs: np.ndarray, rules: np.ndarray, masks: np.ndarray, excl: np.ndarray) -> dict:
    """Сравнение способов сборки итоговой маски на OOF (гейт по правилу и
    исключение облаков включены)."""
    res = {}
    region = probs.shape[1] == 1
    pb = probs[:, 0] if region else 1 - probs[:, 0]
    for th in (0.3, 0.4, 0.5, 0.6):
        for mode in (("rule",) if region else ("rule", "cnn")):
            acc = ScoreAccumulator()
            for i in range(len(masks)):
                burned = (pb[i] > th) & (rules[i] > 0) & ~excl[i]
                if mode == "rule":
                    sev = np.clip(rules[i], 1, 3)
                else:
                    sev = probs[i, 1:].argmax(0) + 1
                pred = np.where(burned, sev, 0)
                acc.add_bs(pred, masks[i])
            r = acc.result()
            res[f"{mode}@{th}"] = {k: round(r[k], 4) for k in ("iou_burn", "miou_sev", "iou_sev1", "iou_sev2", "iou_sev3")}
            res[f"{mode}@{th}"]["bs_part"] = round(0.35 * r["iou_burn"] + 0.30 * r["miou_sev"], 4)
    if region:
        return res
    acc = ScoreAccumulator()
    for i in range(len(masks)):
        acc.add_bs(probs[i].argmax(0), masks[i])
    r = acc.result()
    res["argmax"] = {"iou_burn": round(r["iou_burn"], 4), "miou_sev": round(r["miou_sev"], 4), "bs_part": round(0.35 * r["iou_burn"] + 0.30 * r["miou_sev"], 4)}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--fold", type=int, default=0, help="-1 = обучение на всём train")
    ap.add_argument("--folds-file", default=str(REPO / "configs" / "folds_bs.csv"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="unet")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--crop", type=int, default=384)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dice", type=float, default=0.5)
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--target", default="sev4", choices=["sev4", "region"])
    ap.add_argument("--ctx", type=int, nargs="?", const=1, default=0,
                    help="уровень контекстных каналов: 1 — дальний контекст (--ctx), 2 — + статистики чипа")
    ap.add_argument("--scale", type=int, default=1, help="уменьшение входа сети (1 = 20 м, 2 = 40 м)")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA весов (например 0.998); 0 — выключено")
    ap.add_argument("--drop-fold", type=int, default=-1, help="дополнительно исключить фолд из обучения (вложенная CV)")
    ap.add_argument("--soft-dir", default=None, help="каталог OOF-вероятностей учителя (<chip>.npy, uint8) для мягкой цели")
    ap.add_argument("--soft-alpha", type=float, default=0.5, help="доля эталона в мягкой цели")
    ap.add_argument("--stem-mode", default=None, choices=["rgb_pair", "swir_pair"],
                    help="инициализация первого слоя ImageNet-ядрами по спектральным каналам (model.init_stem_from_bands)")
    ap.add_argument("--smooth", type=float, default=0.0, help="сглаживание меток области (0.1 -> цели 0.05 / 0.95)")
    ap.add_argument("--boot", type=float, default=1.0, help="soft bootstrapping: доля эталона в цели после --boot-start эпох")
    ap.add_argument("--boot-start", type=int, default=50)
    ap.add_argument("--gce", type=float, default=0.0, help="q обобщённой кросс-энтропии вместо BCE (0 — выключено)")
    ap.add_argument("--chdrop", type=float, default=0.0, help="вероятность обнулить группу входных каналов (S1 / SCL / покров+рельеф / контекст) в батче")
    ap.add_argument("--noise", type=float, default=0.0, help="СКО гауссова шума, добавляемого к входу при обучении")
    ap.add_argument("--no-aug", action="store_true", help="без диэдральной аугментации и TTA (проверка направленных эффектов)")
    ap.add_argument("--lovasz", type=float, default=0.0, help="вес Lovász hinge в лоссе области")
    ap.add_argument("--aux-frac", type=float, default=0.0, help="вес вспомогательной цели «доля гари в чипе» (голова на энкодере)")
    ap.add_argument("--hot", action="store_true", help="каналы «видимый огонь» (features.hot_features)")
    ap.add_argument("--coord", action="store_true", help="каналы положения в чипе (model.CoordInput)")
    ap.add_argument("--dec-attn", default=None, help="внимание в декодере: scse")
    ap.add_argument("--drop-path", type=float, default=0.0, help="stochastic depth энкодера timm")
    ap.add_argument("--cache", action="store_true",
                    help="кэш уменьшенных признаков на GPU (только весь чип и scale > 1): быстрее в разы, результат тот же")
    a = ap.parse_args()

    seed_everything(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(Path(a.data_dir) / "bs" / "meta.csv")
    folds = pd.read_csv(a.folds_file).set_index("chip_id").loc[meta.chip_id, "fold"].values
    tr_idx = np.where(folds != a.fold)[0] if a.fold >= 0 else np.arange(len(meta))
    if a.drop_fold >= 0:  # вложенная проверка дистилляции: учитель, не видевший ещё и этот фолд
        tr_idx = tr_idx[folds[tr_idx] != a.drop_fold]
    va_idx = np.where(folds == a.fold)[0] if a.fold >= 0 else np.array([], int)

    t0 = time.time()
    arr = load_arrays(a.data_dir, list(meta.chip_id))
    print(f"loaded {len(meta)} chips in {time.time() - t0:.1f}s", flush=True)

    model = build_model(a.arch, a.encoder, pretrained=not a.no_pretrained, classes=1 if a.target == "region" else 4, ctx=a.ctx,
                        coord=a.coord, dec_attn=a.dec_attn, drop_path=a.drop_path, aux_head=a.aux_frac > 0, hot=a.hot).to(dev)
    if a.stem_mode and not a.no_pretrained:
        init_stem_from_bands(model, a.stem_mode)
        print("stem initialization:", a.stem_mode, flush=True)
    model.scale = a.scale
    # EMA весов: усреднение по траектории обучения сглаживает шум последних шагов (мало данных)
    ema = {k: v.detach().clone().float() for k, v in model.state_dict().items() if v.dtype.is_floating_point} if a.ema > 0 else None

    def ema_update(step: int) -> None:
        if ema is None:
            return
        d = min(a.ema, (1 + step) / (10 + step))
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in ema:
                    ema[k].mul_(d).add_(v.detach().float(), alpha=1 - d)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps_per_epoch = int(np.ceil(len(tr_idx) / a.bs))
    total = a.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.05, anneal_strategy="cos")
    rng = np.random.default_rng(a.seed)

    cache = None
    if a.cache and a.scale > 1 and a.crop >= arr["mask"].shape[-1]:
        cache = {"X": [], "rule": [], "y": [], "excl": []}
        for b in range(0, len(meta), 8):
            idx = np.arange(b, min(b + 8, len(meta)))
            X, rule, y, excl = batch_features(arr, idx, dev, a.ctx, a.hot)
            cache["X"].append(F.avg_pool2d(X, a.scale).half())
            cache["rule"].append(rule.to(torch.uint8))
            cache["y"].append(y.to(torch.uint8))
            cache["excl"].append(excl)
        cache = {k: torch.cat(v) for k, v in cache.items()}
        if a.soft_dir:
            in_train = set(meta.chip_id[tr_idx])  # мягкая цель нужна только обучающим чипам
            soft = [np.load(Path(a.soft_dir) / f"{c}.npy") if c in in_train else np.zeros(arr["mask"].shape[-2:], np.uint8) for c in meta.chip_id]
            cache["soft"] = torch.from_numpy(np.stack([x[0] if x.ndim == 3 else x for x in soft])).to(dev)
        print(f"feature cache {tuple(cache['X'].shape)} in {time.time() - t0:.0f}s", flush=True)

    step = 0
    for ep in range(a.epochs):
        model.train()
        perm = rng.permutation(tr_idx)
        losses = []
        for b in range(steps_per_epoch):
            idx = np.sort(perm[b * a.bs : (b + 1) * a.bs])
            if cache is not None:
                ti = torch.from_numpy(idx).to(dev)
                Xs, rule, y, excl = cache["X"][ti].float(), cache["rule"][ti].long(), cache["y"][ti].long(), cache["excl"][ti]
                k, f = (0, False) if a.no_aug else (int(rng.integers(4)), bool(rng.integers(2)))
                Xs, y, rule, excl = (dihedral(t, k, f).contiguous() for t in (Xs, y, rule, excl))
                soft = dihedral(cache["soft"][ti].float() / 255.0, k, f).contiguous() if "soft" in cache else None
                if a.chdrop > 0:
                    for lo, hi in CHANNEL_GROUPS:
                        if rng.random() < a.chdrop:
                            Xs[:, lo:hi] = 0
                if a.noise > 0:
                    Xs = Xs + a.noise * torch.randn_like(Xs)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    logits = model(Xs)
                aux_logit = None
                if isinstance(logits, tuple):
                    logits, aux_logit = logits
                logits = F.interpolate(logits.float(), size=y.shape[-2:], mode="bilinear", align_corners=False)
                loss = (region_loss(logits, y, rule, excl, a.dice, a.lovasz, soft, a.soft_alpha, a.smooth,
                                    a.boot if ep >= a.boot_start else 1.0, a.gce)
                        if a.target == "region" else loss_fn(logits, y, a.dice))
                if aux_logit is not None:
                    loss = loss + a.aux_frac * chip_fraction_loss(aux_logit, y, rule, excl)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
                sched.step()
                step += 1
                ema_update(step)
                losses.append(loss.item())
                continue
            X, rule, y, excl = batch_features(arr, idx, dev, a.ctx, a.hot)
            H = X.shape[-1]
            if a.crop < H:
                oy, ox = rng.integers(0, H - a.crop + 1, 2)
                sl = (..., slice(oy, oy + a.crop), slice(ox, ox + a.crop))
                X, y, rule, excl = X[sl], y[sl], rule[sl], excl[sl]
            k, f = int(rng.integers(4)), bool(rng.integers(2))
            X, y, rule, excl = (dihedral(t, k, f).contiguous() for t in (X, y, rule, excl))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logits = forward_scaled(model, X, a.scale)
            if a.target == "region":
                loss = region_loss(logits, y, rule, excl, a.dice)
            else:
                loss = loss_fn(logits.float(), y, a.dice)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            step += 1
            ema_update(step)
            losses.append(loss.item())
        if ep % 10 == 0 or ep == a.epochs - 1:
            print(f"ep {ep} loss {np.mean(losses):.4f} t {time.time() - t0:.0f}s", flush=True)

    if ema is not None:
        model.load_state_dict({**model.state_dict(), **ema})
    if a.aux_frac > 0:  # голова нужна только в обучении: в чекпойнт и инференс не попадает
        (model.net if a.coord else model).classification_head = None
    torch.save({"state_dict": model.state_dict(), "arch": a.arch, "encoder": a.encoder, "classes": 1 if a.target == "region" else 4, "ctx": a.ctx, "scale": a.scale,
                "coord": a.coord, "dec_attn": a.dec_attn, "hot": a.hot, "args": vars(a)}, out / "model.pt")

    if len(va_idx):
        model.eval()
        probs, rules, excls = [], [], []
        for b in range(0, len(va_idx), 4):
            idx = va_idx[b : b + 4]
            X, rule, _, excl = batch_features(arr, idx, dev, a.ctx, a.hot)
            probs.append(predict_probs(model, X, 1 if a.no_aug else a.tta).cpu().numpy())
            rules.append(rule.cpu().numpy())
            excls.append(excl.cpu().numpy())
        probs = np.concatenate(probs)
        rules = np.concatenate(rules).astype(np.uint8)
        excls = np.concatenate(excls)
        oof_dir = out / "oof"
        oof_dir.mkdir(exist_ok=True)
        for i, j in enumerate(va_idx):
            np.save(oof_dir / f"{meta.chip_id[j]}.npy", (probs[i] * 255).round().astype(np.uint8))
        res = evaluate(probs, rules, arr["mask"][va_idx], excls)
        (out / "val.json").write_text(json.dumps(res, indent=1))
        best = max(res.items(), key=lambda kv: kv[1]["bs_part"])
        print("VAL best", best, flush=True)
        if "argmax" in res:
            print("VAL argmax", res["argmax"], flush=True)
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
