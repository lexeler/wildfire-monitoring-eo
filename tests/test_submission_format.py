"""Формат готового submission.csv и согласованность конфигураций весов.

Тесты не требуют GPU, torch и тестовых данных: проверяются только файлы
репозитория — `submissions/submission.csv`, `weights/*_config.json` и
`configs/bs_ensemble.json`. Полная проверка сабмита вместе с чтением растров
и подсчётом метрики — `scripts/validate_submission.py`.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from firemon.rle import rle_decode, rle_encode  # noqa: E402

SUB = REPO / "submissions" / "submission.csv"
WEIGHTS = REPO / "weights"
CONFIGS = REPO / "configs"

# Размеры чипов из постановки: AF — VIIRS 375 м, BS — Sentinel-2 20 м
CHIP_SHAPE = {"AF": (256, 256), "BS": (512, 512)}
# Строк на чип в шаблоне: AF — только класс 1, BS — степени 1..3
CHIP_CLASSES = {"AF": {1}, "BS": {1, 2, 3}}


def kind(chip_id: str) -> str:
    return chip_id.split("_")[0]


@pytest.fixture(scope="module")
def raw_text() -> str:
    if not SUB.exists():
        pytest.skip(f"{SUB} отсутствует (запустите inference.py)")
    return SUB.read_bytes().decode("utf-8")


@pytest.fixture(scope="module")
def rows(raw_text) -> list[dict]:
    return list(csv.DictReader(raw_text.splitlines()))


@pytest.fixture(scope="module")
def masks(rows) -> dict[str, dict[int, np.ndarray]]:
    per_chip: dict[str, dict[int, np.ndarray]] = {}
    for r in rows:
        shape = CHIP_SHAPE[kind(r["chip_id"])]
        per_chip.setdefault(r["chip_id"], {})[int(r["class_id"])] = rle_decode(r["rle"], shape)
    return per_chip


def test_header_and_quoting(raw_text):
    lines = raw_text.splitlines()
    assert lines[0] == "chip_id,class_id,rle"
    assert raw_text.endswith("\n") and "\r" not in raw_text
    assert not raw_text.startswith("﻿"), "BOM в файле"
    # поле rle всегда в кавычках, в том числе пустая маска ("" из постановки)
    assert all(line.count('"') == 2 and line.endswith('"') for line in lines[1:])


def test_no_missing_values(rows):
    for r in rows:
        assert r["chip_id"] and r["class_id"] is not None and r["rle"] is not None
        assert r["rle"].strip().lower() not in {"nan", "none", "null"}
        assert int(r["class_id"]) in CHIP_CLASSES[kind(r["chip_id"])]


def test_pairs_are_unique_and_cover_every_chip(rows):
    pairs = [(r["chip_id"], r["class_id"]) for r in rows]
    assert len(set(pairs)) == len(pairs), "дубли пар (chip_id, class_id)"
    per_chip: dict[str, set[int]] = {}
    for cid, cls in pairs:
        per_chip.setdefault(cid, set()).add(int(cls))
    for cid, classes in per_chip.items():
        assert classes == CHIP_CLASSES[kind(cid)], f"{cid}: набор классов {sorted(classes)}"
    expected = sum(len(CHIP_CLASSES[kind(cid)]) for cid in per_chip)
    assert len(rows) == expected


def test_rle_is_canonical(rows):
    """Серии в пределах чипа, упорядочены, не пересекаются и не касаются.

    rle_decode поднимает ValueError на любое из нарушений; совпадение
    повторного кодирования с исходной строкой дополнительно исключает
    нештатные записи (лишние пробелы, склейку соседних серий).
    """
    for r in rows:
        m = rle_decode(r["rle"], CHIP_SHAPE[kind(r["chip_id"])])
        assert rle_encode(m) == r["rle"], f"{r['chip_id']},{r['class_id']}: неканоничный RLE"


def test_classes_do_not_overlap(masks):
    for cid, d in masks.items():
        if len(d) > 1:
            total = np.sum([m.astype(np.uint8) for m in d.values()], axis=0)
            assert total.max() <= 1, f"{cid}: маски классов пересекаются"


def test_submission_is_not_degenerate(masks):
    """Защита от «пустого» сабмита: прогноз есть хотя бы у части чипов."""
    assert any(m.any() for d in masks.values() for m in d.values())


# --- конфигурации весов -------------------------------------------------


@pytest.fixture(scope="module")
def af_cfg() -> dict:
    p = WEIGHTS / "af_config.json"
    if not p.exists():
        pytest.skip("weights/ отсутствует (см. README, раздел «Веса»)")
    return json.loads(p.read_text())


@pytest.fixture(scope="module")
def bs_cfg() -> dict:
    p = WEIGHTS / "bs_config.json"
    if not p.exists():
        pytest.skip("weights/ отсутствует (см. README, раздел «Веса»)")
    return json.loads(p.read_text())


def test_af_config(af_cfg):
    assert 0.0 < af_cfg["threshold"] < 1.0
    for f in af_cfg["model_files"]:
        assert (WEIGHTS / f).exists(), f"нет файла модели {f}"
    names = af_cfg["feature_names"]
    assert names and len(set(names)) == len(names)
    assert af_cfg["params"]["deterministic"] is True, "LightGBM без deterministic=True"


def test_af_model_feature_count(af_cfg):
    lgb = pytest.importorskip("lightgbm")
    for f in af_cfg["model_files"]:
        b = lgb.Booster(model_file=str(WEIGHTS / f))
        assert b.num_feature() == len(af_cfg["feature_names"])


def test_bs_config(bs_cfg):
    assert 0.0 < bs_cfg["threshold"] < 1.0
    groups = [m["group"] for m in bs_cfg["cnn"]]
    assert len(set(groups)) == len(groups)
    for f in bs_cfg["gbdt"]["s1"] + bs_cfg["gbdt"].get("s2", []):
        assert (WEIGHTS / f).exists(), f"нет модели GBDT {f}"
    assert len(bs_cfg["blend_sources"]) == len(bs_cfg["blend_weights"])
    known = set(groups) | {"gbdt_s1", "gbdt_s2"}
    assert set(bs_cfg["blend_sources"]) <= known
    assert all(w >= 0 for w in bs_cfg["blend_weights"]) and sum(bs_cfg["blend_weights"]) > 0
    # вторая ступень GBDT — отвергнутый эксперимент, в финале выключена
    assert bs_cfg["final"] in {"blend", "cnn", "s2"}


def test_bs_checkpoints(bs_cfg):
    """Чекпойнты сетей публикуются в Releases: bash scripts/download_weights.sh."""
    missing = [m["file"] for m in bs_cfg["cnn"] if not (WEIGHTS / m["file"]).exists()]
    if len(missing) == len(bs_cfg["cnn"]):
        pytest.skip("чекпойнты сетей не скачаны (bash scripts/download_weights.sh)")
    assert not missing, f"нет чекпойнтов {missing}"


def test_bs_config_matches_training_recipe(bs_cfg):
    """weights/bs_config.json собирается train.py из configs/bs_ensemble.json."""
    recipe = json.loads((CONFIGS / "bs_ensemble.json").read_text())
    assert [m["group"] for m in bs_cfg["cnn"]] == [m["name"] for m in recipe["cnn"]]
    assert [m["file"] for m in bs_cfg["cnn"]] == [f"bs_{m['name']}.pt" for m in recipe["cnn"]]
    assert bs_cfg["threshold"] == recipe["blend"]["threshold"]
    assert bs_cfg["blend_sources"] == recipe["blend"]["sources"]
    assert bs_cfg["blend_weights"] == recipe["blend"]["weights"]
    assert bs_cfg["tta"] == recipe["tta"]
    assert bs_cfg["gbdt"]["block"] == recipe["gbdt"]["block"]


def test_folds_cover_train_chips():
    """Фолды CV зафиксированы в репозитории и разбивают чипы без пересечений."""
    for name in ("folds_af.csv", "folds_bs.csv"):
        rows = list(csv.DictReader((CONFIGS / name).read_text().splitlines()))
        assert rows, f"{name} пуст"
        ids = [r["chip_id"] for r in rows]
        assert len(set(ids)) == len(ids), f"{name}: дубли chip_id"
        assert {int(r["fold"]) for r in rows} == set(range(5))
