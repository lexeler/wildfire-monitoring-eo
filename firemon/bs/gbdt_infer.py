"""Параллельный CPU-инференс GBDT-ступеней модуля гарей.

Ступень 1: признаки чипа на сетке блоков -> P1 (доля области в блоке).
Ступень 2 (автоконтекст): те же признаки + многомасштабный контекст карты
первой ступени (смесь P1 и вероятностей сетей) -> итоговая вероятность.

Процессы-воркеры сами читают растры (не гоняем мегабайты через pickle),
признаки ступени 1 кэшируются на диск только если есть вторая ступень
(/dev/shm при достатке места), LightGBM в каждом воркере работает в один
поток — параллелизм по чипам.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

_STATE: dict = {}


def _init(data_dir: str, s1_files: list[str], s2_files: list[str], block: int, cache_dir: str, sparse: bool = False) -> None:
    import lightgbm as lgb

    # Воркеры GBDT — фоновая нагрузка: критический путь инференса — главный процесс (загрузка сетей, AF, сети BS
    # на GPU). Пониженный приоритет не даёт воркерам отбирать у него CPU на машинах с небольшим числом ядер.
    try:
        os.nice(10)
    except (AttributeError, OSError):
        pass

    _STATE.update(
        data_dir=data_dir,
        s1=[lgb.Booster(model_file=f) for f in s1_files],
        s2=[lgb.Booster(model_file=f) for f in s2_files],
        block=block,
        cache=cache_dir,
        sparse=sparse,
    )


def available_cpus() -> int:
    """Ядра, реально доступные процессу: cpuset контейнера (affinity) и квота
    CPU cgroup (docker run --cpus=N не меняет affinity, но ограничивает время)."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:  # macOS / Windows
        n = os.cpu_count() or 2
    try:  # cgroup v2: "<квота> <период>" или "max <период>"
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            n = min(n, max(1, int(float(quota) / float(period) + 0.5)))
    except (OSError, ValueError):
        try:  # cgroup v1
            quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if quota > 0 and period > 0:
                n = min(n, max(1, int(quota / period + 0.5)))
        except (OSError, ValueError):
            pass
    return n


def _noop(_):
    return None


def _predict(boosters, X: np.ndarray) -> np.ndarray:
    return np.mean([b.predict(X, num_threads=1) for b in boosters], 0).astype(np.float32)


def _stage1(cid: str) -> tuple[str, np.ndarray, np.ndarray]:
    from ..io import load_bs
    from .gbdt import chip_features, predict_blocks

    c = load_bs(_STATE["data_dir"], cid)
    X, _, full = chip_features(c, _STATE["block"])
    p = predict_blocks(lambda rows: _predict(_STATE["s1"], rows), X, _STATE["sparse"])
    if _STATE["s2"]:  # признаки нужны только второй ступени
        np.save(os.path.join(_STATE["cache"], f"{cid}.npy"), X)
    return cid, p, None


def _stage2(args) -> tuple[str, np.ndarray]:
    from .gbdt import stack_features

    cid, stack_prob = args
    X = np.load(os.path.join(_STATE["cache"], f"{cid}.npy"), mmap_mode="r")
    S, _ = stack_features(stack_prob.astype(np.float32), _STATE["block"])
    sel = X[0] > 0
    p = np.zeros(sel.shape, np.float32)
    if sel.any():
        rows = np.concatenate([X[:, sel], S[:, sel]]).T
        p[sel] = _predict(_STATE["s2"], rows)
    return cid, p


class GBDTStages:
    """Пул процессов создаётся до инициализации CUDA (fork безопасен)."""

    def __init__(self, data_dir: str, s1_files: list[str], s2_files: list[str], block: int, workers: int | None = None,
                 sparse: bool = False):
        workers = workers or max(1, min(32, available_cpus() - 1))
        # /dev/shm в Docker по умолчанию 64 МБ — берём его только при достатке места
        base = None
        try:
            if os.path.isdir("/dev/shm") and shutil.disk_usage("/dev/shm").free > 8 * 2**30:
                base = "/dev/shm"
        except OSError:
            pass
        self._tmp = tempfile.TemporaryDirectory(dir=base, prefix="firemon_")
        ctx = mp.get_context("fork") if hasattr(os, "fork") else mp.get_context("spawn")
        self.pool = ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init,
                                        initargs=(str(data_dir), [str(f) for f in s1_files], [str(f) for f in s2_files], block, self._tmp.name, sparse))
        self.block = block

    def warmup(self) -> None:
        """Запускает все процессы сразу (до потоков чтения и CUDA)."""
        list(self.pool.map(_noop, range(self.pool._max_workers)))

    def stage1(self, ids: list[str]):
        """Асинхронно: итератор (cid, P1 на сетке блоков, ok полноразмерная)."""
        return self.pool.map(_stage1, ids, chunksize=1)

    def stage2(self, items: list[tuple[str, np.ndarray]]):
        return self.pool.map(_stage2, items, chunksize=1)

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)
        self._tmp.cleanup()


def cleanup_path(p: str | Path) -> None:
    try:
        Path(p).unlink()
    except OSError:
        pass
