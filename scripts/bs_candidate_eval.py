"""Paired OOF comparison of BS ensembles and severity rules on every train chip.

Reports fixed-threshold and grid results, per-fold metrics and a paired spatial
cluster bootstrap. The bootstrap describes sampling uncertainty; it does not
correct for selecting hypotheses on the same OOF sample.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from firemon.bs.rules import THRESHOLDS, THRESHOLDS_STATEMENT, dnbr, excluded, severity_from_dnbr
from firemon.io import load_bs, parallel_map
from scripts.bs_oof_analysis import burn_prob


def metrics(cm):
    cm = np.asarray(cm, dtype=np.float64)
    tp = cm[..., 1:, 1:].sum(axis=(-2, -1))
    union = cm.sum(axis=(-2, -1)) - cm[..., 0, 0]
    iou = np.divide(tp, union, out=np.ones_like(tp), where=union != 0)
    diag = np.diagonal(cm, axis1=-2, axis2=-1)[..., 1:]
    den = cm.sum(axis=-2)[..., 1:] + cm.sum(axis=-1)[..., 1:] - diag
    sev = np.divide(diag, den, out=np.ones_like(diag), where=den != 0)
    miou = sev.mean(axis=-1)
    return iou, miou, 0.35 * iou + 0.30 * miou, sev


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--configs', nargs='+', required=True, help='name:w1,w2,...')
    ap.add_argument('--rules', nargs='+', default=['legacy', 'class_median', 'class_median_statement', 'class_median_calibrated'])
    ap.add_argument('--thresholds', nargs='+', type=float, default=[.425, .45, .475, .5, .525])
    ap.add_argument('--folds', default='configs/folds_bs.csv')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    meta = pd.read_csv(Path(a.data_dir) / 'bs/meta.csv')
    folds = pd.read_csv(a.folds).set_index('chip_id').loc[meta.chip_id]
    cfgs = {}
    for spec in a.configs:
        name, weights = spec.split(':', 1)
        w = np.asarray([float(x) for x in weights.split(',')], np.float32)
        if len(w) != len(a.runs) or (w < 0).any() or w.sum() <= 0:
            raise ValueError(f'Invalid weights: {spec}')
        cfgs[name] = w / w.sum()
    keys = [(name, rule, th) for name in cfgs for rule in a.rules for th in a.thresholds]
    calibrated = dict(THRESHOLDS_STATEMENT, forest=(.1, .27, .6), wetland=(.075, .331, .617))

    def one(cid):
        c = load_bs(a.data_dir, cid, with_mask=True)
        if c.mask is None:
            raise ValueError(f'Missing label for {cid}')
        ex = excluded(c.s2_pre[9], c.s2_post[9])
        d = dnbr(c.s2_pre, c.s2_post, median=0)
        rules = {}
        for mode in a.rules:
            if mode == 'legacy':
                r = severity_from_dnbr(ndi.median_filter(d, size=3), c.aux[2])
            elif mode.startswith('class_median'):
                th = (THRESHOLDS_STATEMENT if mode.endswith('statement') else
                      calibrated if mode.endswith('calibrated') else THRESHOLDS)
                raw = severity_from_dnbr(d, c.aux[2], th)
                if not mode.endswith('nomask'):
                    raw[ex] = 0
                r = ndi.median_filter(raw, size=3)
            else:
                raise ValueError(mode)
            r[ex] = 0
            rules[mode] = r
        ps = []
        for run in a.runs:
            hits = list(Path(run).glob(f'f*/oof/{cid}.npy'))
            if len(hits) != 1:
                raise ValueError(f'{run}: expected one OOF for {cid}, got {len(hits)}')
            ps.append(burn_prob(np.load(hits[0]).astype(np.float32) / 255))
        p = np.stack(ps)
        probs = {name: np.einsum('n,nhw->hw', w, p) for name, w in cfgs.items()}
        cms = []
        for name, mode, th in keys:
            pred = np.where(probs[name] > th, rules[mode], 0)
            cms.append(np.bincount((c.mask.astype(np.int64) * 4 + pred).ravel(), minlength=16).reshape(4, 4))
        return np.stack(cms)

    cms = np.stack(parallel_map(one, meta.chip_id, workers=a.workers))
    baseline_key = (next(iter(cfgs)), 'legacy', .475)
    bi = keys.index(baseline_key)
    groups = folds.cluster.to_numpy()
    clustered = np.stack([cms[groups == g].sum(axis=0) for g in np.unique(groups)])
    rng = np.random.default_rng(20260919)
    draw = rng.integers(0, len(clustered), size=(2000, len(clustered)))
    boot_cm = clustered[draw].sum(axis=1)
    boot_score = metrics(boot_cm)[2]
    total = metrics(cms.sum(axis=0))
    rows = []
    for i, (name, mode, th) in enumerate(keys):
        fold_scores = {}
        for f in sorted(folds.fold.unique()):
            fold_scores[str(f)] = float(metrics(cms[folds.fold.to_numpy() == f, i].sum(axis=0))[2])
        diff = boot_score[:, i] - boot_score[:, bi]
        row = dict(name=name, rule=mode, threshold=th, iou_burn=float(total[0][i]),
                   miou_sev=float(total[1][i]), bs_part=float(total[2][i]), iou_sev=total[3][i].tolist(),
                   folds=fold_scores, delta=float(total[2][i] - total[2][bi]),
                   paired_cluster_bootstrap_95=np.quantile(diff, [.025, .975]).tolist())
        rows.append(row)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(runs=a.runs, configs=a.configs, baseline=list(baseline_key),
                                  chips=len(meta), rows=rows), indent=2) + '\n')
    np.savez_compressed(out.with_suffix('.npz'), cm=cms, chip_id=meta.chip_id.to_numpy(dtype=str),
                        cluster=groups, fold=folds.fold.to_numpy())
    print('All chips:', len(meta), 'baseline:', total[2][bi], flush=True)
    for row in rows:
        if row['threshold'] == .475:
            print(json.dumps(row), flush=True)
    print('BEST PER COMBINATION', flush=True)
    for name in cfgs:
        for mode in a.rules:
            row = max((r for r in rows if r['name'] == name and r['rule'] == mode), key=lambda r: r['bs_part'])
            print(name, mode, row['threshold'], round(row['bs_part'], 6), row['folds'], flush=True)


if __name__ == '__main__':
    main()
