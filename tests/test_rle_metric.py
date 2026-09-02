"""Юнит-тесты кодирования RLE и метрики (pytest tests/)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from firemon.metric import ScoreAccumulator  # noqa: E402
from firemon.rle import rle_decode, rle_encode  # noqa: E402


def test_example_from_statement():
    # рис. 8 постановки: чип 8x8, строка 3 столбцы 4-5 и строка 5 столбцы 2-4
    m = np.zeros((8, 8), bool)
    m[2, 3:5] = True
    m[4, 1:4] = True
    assert rle_encode(m) == "20 2 34 3"
    assert (rle_decode("20 2 34 3", (8, 8)) == m).all()


@pytest.mark.parametrize("seed", range(5))
def test_roundtrip_random(seed):
    rng = np.random.default_rng(seed)
    m = rng.random((37, 53)) > 0.7
    m[0, 0] = m[-1, -1] = True  # края
    assert (rle_decode(rle_encode(m), m.shape) == m).all()


def test_empty():
    assert rle_encode(np.zeros((4, 4))) == ""
    assert not rle_decode("", (4, 4)).any()


@pytest.mark.parametrize("bad", ["1 0", "0 2", "15 3", "5 2 6 1", "5 2 7 1", "3"])
def test_invalid(bad):
    with pytest.raises(ValueError):
        rle_decode(bad, (4, 4))


def test_metric_perfect_and_empty_classes():
    acc = ScoreAccumulator()
    af = np.zeros((5, 5), np.uint8); af[1, 1] = 1
    bs = np.zeros((5, 5), np.uint8); bs[2:4, 2:4] = 2
    acc.add_af(af, af)
    acc.add_bs(bs, bs)
    r = acc.result()
    # классы 1 и 3 отсутствуют и в эталоне, и в прогнозе -> IoU = 1
    assert r["score"] == pytest.approx(1.0)


def test_metric_micro_average():
    acc = ScoreAccumulator()
    t = np.array([[1, 1, 0, 0]], np.uint8)
    p = np.array([[1, 0, 1, 0]], np.uint8)
    acc.add_af(p, t)
    assert acc.result()["f1_af"] == pytest.approx(0.5)
