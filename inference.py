#!/usr/bin/env python
"""Инференс решения команды LitWin: submission.csv по каталогу тестовых чипов.

    python inference.py --data-dir /path/to/test --output /path/to/submission.csv

Каталог — в структуре выданного набора (sample_submission.csv, af/, bs/).
Скрипт обрабатывает все чипы шаблона, включая облачные, никогда не падает на
отдельном чипе (при ошибке — безопасный запасной ответ с предупреждением)
и завершается с кодом 0.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from firemon.bs.gbdt import upsample  # noqa: E402
from firemon.io import load_af, load_bs  # noqa: E402
from firemon.rle import rle_encode  # noqa: E402

log = logging.getLogger("inference")
BS_BATCH = 8  # чипов BS за один проход сети


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="каталог с тестовыми чипами и sample_submission.csv")
    ap.add_argument("--output", required=True, help="путь к итоговому submission.csv")
    ap.add_argument("--weights", default=str(REPO / "weights"), help="каталог с весами (по умолчанию ./weights)")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--workers", type=int, default=16, help="потоков чтения растров")
    ap.add_argument("--gbdt-workers", type=int, default=0, help="процессов GBDT (0 = число ядер - 1)")
    ap.add_argument("--dump-probs", default=None, help="отладка: каталог для вероятностей по чипам (npz); на результат не влияет")
    return ap.parse_args()


def main() -> int:
    a = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    t0 = time.perf_counter()
    data_dir = Path(a.data_dir)
    sample = pd.read_csv(data_dir / "sample_submission.csv", dtype={"chip_id": str}, keep_default_na=False)
    chip_ids = list(dict.fromkeys(sample.chip_id))
    # тип чипа — по наличию файла VIIRS; если файла нет, но идентификатор начинается с «AF», это всё равно AF-чип
    # (строка получит пустую маску с предупреждением, а не уйдёт в модуль гарей)
    kind = {c: ("af" if (data_dir / "af" / "viirs" / f"{c}_VIIRS_I1-I5.tif").exists() or c.upper().startswith("AF") else "bs")
            for c in chip_ids}
    af_ids = [c for c in chip_ids if kind[c] == "af"]
    bs_ids = [c for c in chip_ids if kind[c] == "bs"]
    log.info("chips: AF %d, BS %d", len(af_ids), len(bs_ids))

    # GBDT-ступени модуля гарей: пул процессов создаётся первым — до потоков
    # чтения и до инициализации CUDA (fork безопасен только так)
    bs_cfg = json.loads((Path(a.weights) / "bs_config.json").read_text())
    stages = None
    if bs_cfg.get("gbdt") and bs_cfg.get("final", "blend") != "cnn" and bs_ids:
        from firemon.bs.gbdt_infer import GBDTStages

        g = bs_cfg["gbdt"]
        stages = GBDTStages(str(data_dir), [Path(a.weights) / f for f in g["s1"]], [Path(a.weights) / f for f in g.get("s2", [])],
                            int(g["block"]), workers=a.gbdt_workers or None, sparse=bool(g.get("sparse", False)))
        stages.warmup()
        s1_results = stages.stage1(bs_ids)  # воркеры сами читают чипы
        log.info("GBDT workers started (%.1fs)", time.perf_counter() - t0)

    pool = ThreadPoolExecutor(a.workers)
    af_fut = {c: pool.submit(_safe_call, load_af, data_dir, c) for c in af_ids}
    bs_fut = {c: pool.submit(_safe_call, load_bs, data_dir, c) for c in bs_ids}

    # Модели грузятся, пока идёт чтение растров
    import torch

    from firemon.af.model import AFModel, predict_af
    from firemon.bs.predict import BSEnsemble

    torch.backends.cudnn.benchmark = False
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    af_model = AFModel.load(a.weights)
    bs_model = BSEnsemble.load(a.weights, device)
    log.info("models loaded on %s (%.1fs)", device, time.perf_counter() - t0)

    masks: dict[str, np.ndarray] = {}

    # --- AF: все чипы одним проходом признаков (группировка по размеру) ---
    af_chips = {c: f.result() for c, f in af_fut.items()}
    groups: dict[tuple, list[str]] = {}
    for c, ch in af_chips.items():
        if ch is not None:
            groups.setdefault(ch.viirs.shape, []).append(c)
    for shape, ids in groups.items():
        try:
            m, pr = predict_af(af_model, np.stack([af_chips[c].viirs for c in ids]), np.stack([af_chips[c].aux for c in ids]), device)
            masks.update(zip(ids, m))
            if a.dump_probs:
                for c, p_ in zip(ids, pr):
                    _dump(a.dump_probs, c, prob=p_.astype(np.float32))
        except Exception:  # noqa: BLE001 — не роняем весь прогон
            log.exception("AF batch failed, falling back per chip")
            for c in ids:
                r = _safe_call(predict_af, af_model, af_chips[c].viirs[None], af_chips[c].aux[None], device)
                if r is not None:
                    masks[c] = r[0][0]
    del af_chips
    log.info("AF done (%.1fs)", time.perf_counter() - t0)

    # --- BS: сети батчами по мере чтения ---
    cnn_out: dict[str, tuple] = {}  # cid -> ({группа: P}, rule, ok)
    batch: list = []

    cpu_model = []  # ленивая CPU-копия ансамбля на случай нехватки памяти GPU

    def run_cnn(model, batch):
        st = lambda k: np.stack([getattr(ch, k) for _, ch in batch])
        probs, rule, ok = model.cnn_arrays(st("s2_pre"), st("s2_post"), st("s1_pre"), st("s1_post"), st("aux"))
        for i, (c, _) in enumerate(batch):
            cnn_out[c] = ({g: v[i] for g, v in probs.items()}, rule[i], ok[i])

    def flush(batch):
        if not batch:
            return
        try:
            run_cnn(bs_model, batch)
        except torch.OutOfMemoryError:  # делим батч пополам, в пределе — CPU
            torch.cuda.empty_cache()
            if len(batch) > 1:
                log.warning("CUDA OOM on BS batch of %d, splitting", len(batch))
                flush(batch[: len(batch) // 2])
                flush(batch[len(batch) // 2 :])
                return
            log.warning("CUDA OOM on single BS chip, using CPU")
            if not cpu_model:
                cpu_model.append(BSEnsemble.load(a.weights, "cpu"))
            _safe_call(run_cnn, cpu_model[0], batch)
        except Exception:  # noqa: BLE001
            log.exception("BS batch failed, falling back per chip")
            for item in batch:
                _safe_call(run_cnn, bs_model, [item])

    for c in bs_ids:
        ch = bs_fut[c].result()
        if ch is None:
            continue
        if batch and ch.s2_pre.shape != batch[0][1].s2_pre.shape:
            flush(batch)
            batch = []
        batch.append((c, ch))
        if len(batch) == BS_BATCH:
            flush(batch)
            batch = []
    flush(batch)
    log.info("BS networks done (%.1fs)", time.perf_counter() - t0)

    # --- BS: GBDT-ступени и сборка маски ---
    final_prob: dict[str, np.ndarray] = {}
    if stages is not None:
        try:
            p1 = {c: p for c, p, _ in s1_results}
            log.info("GBDT stage 1 done (%.1fs)", time.perf_counter() - t0)
            items = []
            for c, (probs, rule, ok) in cnn_out.items():
                if c in p1:
                    items.append((c, bs_model.blend_prob(upsample(p1[c], rule.shape), probs)))
            if bs_model.cfg.final == "s2" and bs_model.cfg.gbdt.get("s2"):
                for c, p2 in stages.stage2(items):
                    final_prob[c] = upsample(p2, cnn_out[c][1].shape)
            else:
                final_prob.update(items)
            log.info("GBDT blend done (%.1fs)", time.perf_counter() - t0)
        except Exception:  # noqa: BLE001 — при сбое GBDT остаётся ансамбль сетей
            log.exception("GBDT stages failed, using networks only")
        finally:
            stages.close()
    for c, (probs, rule, ok) in cnn_out.items():
        p = final_prob.get(c)
        if p is None:
            p = np.mean(list(probs.values()), 0)
        masks[c] = bs_model.assemble(p, rule, ok)
        if a.dump_probs:
            _dump(a.dump_probs, c, prob=p.astype(np.float32), rule=rule, ok=ok, **{f"m_{g}": v.astype(np.float16) for g, v in probs.items()})
    log.info("BS done (%.1fs)", time.perf_counter() - t0)

    # --- submission.csv ---
    rows = []
    n_fallback = 0
    for cid, cls in zip(sample.chip_id, sample.class_id.astype(int)):
        m = masks.get(cid)
        if m is None:
            n_fallback += 1
            rle = ""
        else:
            rle = rle_encode(m == cls)
        rows.append(f'{cid},{cls},"{rle}"')
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("chip_id,class_id,rle\n" + "\n".join(rows) + "\n", encoding="utf-8")
    if n_fallback:
        log.warning("%d rows written as empty masks (chip failed)", n_fallback)
    pool.shutdown(wait=False, cancel_futures=True)
    log.info("wrote %s: %d rows, total %.1fs", out, len(rows), time.perf_counter() - t0)
    return 0


def _dump(out_dir: str, chip_id: str, **arrays) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(out_dir) / f"{chip_id}.npz", **arrays)


def _safe_call(fn, *args):
    try:
        return fn(*args)
    except Exception:  # noqa: BLE001
        log.exception("chip processing failed")
        return None


if __name__ == "__main__":
    try:
        code = main()
    except Exception:  # noqa: BLE001 — сюда попадают только сбои вне обработки чипов (нет шаблона, нет весов): код 1
        logging.getLogger("inference").exception("fatal error")
        code = 1
    sys.exit(code)
