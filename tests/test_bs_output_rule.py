"""The output rule must match between OOF evaluation and CPU/GPU inference."""
import numpy as np
import pytest
import torch

from firemon.bs.features import output_class_median
from firemon.bs.rules import MIN_AREA_PX, output_rule, sieve_small


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_output_rule_backends_match(device):
    rng = np.random.default_rng(42)
    pre = rng.integers(0, 10001, (3, 10, 17, 19), dtype=np.int32)
    post = rng.integers(0, 10001, pre.shape, dtype=np.int32)
    pre[:, 9] = rng.integers(0, 12, pre[:, 9].shape)
    post[:, 9] = rng.integers(0, 12, post[:, 9].shape)
    aux = np.zeros((3, 3, 17, 19), dtype=np.int32)
    aux[:, 2] = rng.choice([0, 10, 20, 30, 40, 50, 60, 80, 90, 95, 100], aux[:, 2].shape)
    got = output_class_median(*(torch.from_numpy(x).to(device) for x in (pre, post, aux))).cpu().numpy()
    expected = np.stack([output_rule(p, q, a[2], "class_median") for p, q, a in zip(pre, post, aux)])
    np.testing.assert_array_equal(got, expected)


def test_output_rule_rejects_unknown_mode():
    x = np.zeros((10, 3, 3), np.uint16)
    with pytest.raises(ValueError, match="Unknown severity"):
        output_rule(x, x, x[0], "typo")


def test_sieve_removes_only_small_components():
    m = np.zeros((20, 20), bool)
    m[1:6, 1:6] = True            # 25 px — остаётся (минимальная площадь объекта эталона — 1 га)
    m[10:14, 10:16] = True        # 24 px — удаляется
    m[14, 16] = True              # +1 px по диагонали (8-связность): 25 px — остаётся
    out = sieve_small(m, MIN_AREA_PX)
    assert out[1:6, 1:6].all() and out[10:14, 10:16].all() and out[14, 16]
    m[14, 16] = False
    out = sieve_small(m, MIN_AREA_PX)
    assert out[1:6, 1:6].all() and not out[10:14, 10:16].any()
    assert sieve_small(m, 0) is m
