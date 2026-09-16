"""Рисунки результатов для отчёта: абляции, примеры OOF-прогнозов BS и AF.

python scripts/make_result_figures.py --data-dir data/train --results reports/results.json \
    --bs-runs runs/A runs/B --bs-weights 1 0.5 --bs-threshold 0.4 --af-oof runs/af_oof --out reports/figures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from firemon.bs.rules import excluded, output_rule, sieve_small  # noqa: E402
from firemon.io import load_af, load_bs  # noqa: E402
from firemon.metric import ScoreAccumulator  # noqa: E402
from scripts.bs_oof_analysis import load_oof  # noqa: E402

SEV = ListedColormap(["#f3f1ec", "#f6c667", "#f08a3c", "#b3261e"])
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False})


def fig_ablation(res: dict, out: Path) -> None:
    rows = res["bs_ablation"]
    names = [r["name"] for r in rows]
    vals = [r["bs_part"] for r in rows]
    fig, ax = plt.subplots(figsize=(11.5, 0.42 * len(rows) + 1.4))
    colors = ["#9aa5ad" if r.get("kind") == "baseline" else ("#b3261e" if r.get("final") else "#f08a3c") for r in rows]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, vals, color=colors, height=0.66)
    for yi, v in zip(y, vals):
        ax.text(v + 0.004, yi, f"{v:.3f}", va="center", fontsize=10)
    ax.set_yticks(y, names)
    ax.set_xlim(0, 0.65)
    ax.axvline(0.65, color="#666", lw=0.8, ls=":")
    ax.set_xlabel("вклад BS в Score = 0.35·IoU_burn + 0.30·mIoU_sev (максимум 0.65)\nOOF, 5-fold CV по пространственным кластерам")
    ax.set_title("Модуль BS: от индексного правила к ансамблю моделей области события", loc="left")
    fig.tight_layout()
    fig.savefig(out / "results_bs_ablation.png", bbox_inches="tight")
    plt.close(fig)


def fig_bs_examples(data_dir: str, runs, weights, th: float, out: Path, severity_mode: str = "legacy", min_area: int = 0) -> None:
    meta = pd.read_csv(Path(data_dir) / "bs" / "meta.csv")
    stats = []
    for cid in meta.chip_id:
        c = load_bs(data_dir, cid, with_mask=True)
        rule = output_rule(c.s2_pre, c.s2_post, c.aux[2], severity_mode)
        ok = (rule > 0) & ~excluded(c.s2_pre[9], c.s2_post[9])
        p = load_oof(runs, cid, weights)[0]
        pred = np.where(sieve_small((p > th) & ok, min_area), np.clip(rule, 1, 3), 0)
        g, pb = c.mask > 0, pred > 0
        inter, union = (g & pb).sum(), (g | pb).sum()
        stats.append((cid, inter / max(union, 1), int(g.sum())))
    df = pd.DataFrame(stats, columns=["chip", "iou", "n"]).query("n > 20000").sort_values("iou")
    pick = [df.iloc[-1].chip, df.iloc[-len(df) // 5].chip, df.iloc[len(df) // 2].chip, df.iloc[len(df) // 5].chip, df.iloc[1].chip]
    labels = ["лучший", "хороший (80-й процентиль)", "медианный", "слабый (20-й процентиль)", "плохой"]
    fig, axs = plt.subplots(len(pick), 4, figsize=(15, 3.8 * len(pick)))
    for row, cid, lab in zip(axs, pick, labels):
        c = load_bs(data_dir, cid, with_mask=True)
        rule = output_rule(c.s2_pre, c.s2_post, c.aux[2], severity_mode)
        ok = (rule > 0) & ~excluded(c.s2_pre[9], c.s2_post[9])
        p = load_oof(runs, cid, weights)[0]
        pred = np.where(sieve_small((p > th) & ok, min_area), np.clip(rule, 1, 3), 0)
        rgb = lambda s: np.clip(np.stack([s[8], s[6], s[2]], -1) / 4000, 0, 1)
        acc = ScoreAccumulator()
        acc.add_bs(pred, c.mask)
        r = acc.result()
        row[0].imshow(rgb(c.s2_post))
        row[0].set_title(f"{cid} — {lab}\nпосле пожара (B12-B8A-B4)")
        row[1].imshow(p, vmin=0, vmax=1, cmap="magma")
        row[1].set_title("вероятность области события (ансамбль)")
        row[2].imshow(c.mask, cmap=SEV, vmin=0, vmax=3, interpolation="nearest")
        row[2].set_title("эталон")
        row[3].imshow(pred, cmap=SEV, vmin=0, vmax=3, interpolation="nearest")
        row[3].set_title(f"прогноз: IoU гари {r['iou_burn']:.2f}, mIoU степ. {r['miou_sev']:.2f}")
        for a in row:
            a.set_xticks([])
            a.set_yticks([])
    fig.suptitle("Модуль BS: примеры OOF-прогнозов (чипы не участвовали в обучении своих моделей)", y=1.005, fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "results_bs_examples.png", bbox_inches="tight")
    plt.close(fig)


def fig_af_examples(data_dir: str, af_oof: str, out: Path) -> None:
    meta = pd.read_csv(Path(data_dir) / "af" / "meta.csv")
    pick = [c for c in meta.sort_values("n_fire_px", ascending=False).chip_id[::37]][:5]
    fig, axs = plt.subplots(1, len(pick), figsize=(4 * len(pick), 4.4))
    for ax, cid in zip(axs, pick):
        c = load_af(data_dir, cid, with_mask=True)
        z = np.load(Path(af_oof) / f"{cid}.npz")
        m, pm = c.mask == 1, z["mask"] == 1
        # окно 48x48 на самом плотном скоплении горящих пикселей
        dens = ndi.uniform_filter((m | pm).astype(np.float32), 31, mode="constant")
        cy, cx = np.unravel_index(int(dens.argmax()), dens.shape)
        y0, x0 = int(np.clip(cy - 24, 0, 208)), int(np.clip(cx - 24, 0, 208))
        sl = (slice(y0, y0 + 48), slice(x0, x0 + 48))
        ax.imshow(np.nan_to_num(c.viirs[3][sl], nan=280), cmap="inferno", vmin=285, vmax=345)
        tp, fp, fn = m & pm, pm & ~m, m & ~pm
        for mask, col, mk in ((tp, "#39d353", "o"), (fp, "#00b7ff", "x"), (fn, "#ffffff", "s")):
            yy, xx = np.nonzero(mask[sl])
            if mk == "x":
                ax.scatter(xx, yy, s=30, c=col, marker=mk, linewidths=1.5)
            else:
                ax.scatter(xx, yy, s=30, facecolors="none", edgecolors=col, marker=mk, linewidths=1.5)
        ax.set_title(f"{cid}\nTP {tp.sum()} · FP {fp.sum()} · FN {fn.sum()}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Модуль AF, OOF-прогноз: фон — I4 (K), ○ верно (зелёный), × ложное (голубой), □ пропуск (белый); окно 18×18 км", fontsize=12, y=1.04)
    fig.tight_layout()
    fig.savefig(out / "results_af_examples.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--bs-runs", nargs="+", required=True)
    ap.add_argument("--bs-weights", nargs="+", type=float, default=None)
    ap.add_argument("--bs-threshold", type=float, required=True)
    ap.add_argument("--bs-severity-mode", choices=["legacy", "class_median"], default="legacy")
    ap.add_argument("--bs-min-area", type=int, default=0)
    ap.add_argument("--af-oof", required=True)
    ap.add_argument("--out", default="reports/figures")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = json.loads(Path(a.results).read_text())
    fig_ablation(res, out)
    fig_af_examples(a.data_dir, a.af_oof, out)
    fig_bs_examples(a.data_dir, a.bs_runs, a.bs_weights, a.bs_threshold, out, a.bs_severity_mode, a.bs_min_area)
    print("figures in", out)


if __name__ == "__main__":
    main()
