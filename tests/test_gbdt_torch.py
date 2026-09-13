"""TorchForest совпадает с lgb.Booster.predict."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# torch импортируется ДО lightgbm: при обратном порядке две копии OpenMP
# (libomp в колесе lightgbm и в колесе torch) намертво блокируют первую же
# параллельную операцию torch. Тот же порядок соблюдён в inference.py.
torch = pytest.importorskip("torch")
lgb = pytest.importorskip("lightgbm")
from firemon.gbdt_torch import TorchForest  # noqa: E402


@pytest.mark.parametrize("objective", ["binary", "cross_entropy", "regression"])
def test_matches_lightgbm(objective):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5000, 12)).astype(np.float32)
    X[:, 3] = np.round(X[:, 3], 1)  # много совпадающих значений у порогов
    y = (X[:, 0] + X[:, 1] * X[:, 3] > 0).astype(float) if objective != "regression" else X[:, 0] * 2 + X[:, 5]
    b = lgb.train(dict(objective=objective, num_leaves=31, verbose=-1, seed=0), lgb.Dataset(X, y), 60)
    ref = b.predict(X)
    got = TorchForest(b).predict(X)
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)
