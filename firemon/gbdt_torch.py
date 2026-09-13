"""Инференс моделей LightGBM на GPU средствами torch.

Модель LightGBM (числовые признаки, решающее правило «x <= порог -> влево»)
раскладывается в плоские массивы узлов; обход всех деревьев выполняется
векторно для блока строк: на каждом шаге глубины каждая пара (строка, дерево)
переходит к левому или правому потомку. Результат совпадает с
lgb.Booster.predict (проверяется в tests/test_gbdt_torch.py) и позволяет
уложить предсказание ~10^6 строк x 10^3 деревьев в доли секунды на GPU.

ЭКСПЕРИМЕНТ: в финальном инференсе не используется (GBDT считается пулом
процессов на CPU параллельно с сетями, см. bs/gbdt_infer.py); покрыт тестами.

Поддерживается: числовые признаки, missing_type None/NaN (NaN -> default_left),
objective binary / cross_entropy (сигмоида) и регрессия (без преобразования).
"""
from __future__ import annotations

import numpy as np
import torch


class TorchForest:
    def __init__(self, booster, device: str | torch.device = "cpu"):
        dump = booster.dump_model()
        self.objective = dump.get("objective", "")
        trees = dump["tree_info"]
        feat, thr, left, right, dleft, is_leaf, value = [], [], [], [], [], [], []
        roots = []
        offset = 0

        def walk(node) -> int:
            nonlocal offset
            idx = offset
            offset += 1
            feat.append(0)
            thr.append(0.0)
            left.append(idx)
            right.append(idx)
            dleft.append(True)
            is_leaf.append(True)
            value.append(0.0)
            if "leaf_value" in node and "split_feature" not in node:
                value[idx] = float(node["leaf_value"])
                return idx
            if node.get("decision_type", "<=") != "<=":
                raise ValueError("only numerical <= splits are supported")
            is_leaf[idx] = False
            feat[idx] = int(node["split_feature"])
            thr[idx] = float(node["threshold"])
            dleft[idx] = bool(node.get("default_left", True))
            left[idx] = walk(node["left_child"])
            right[idx] = walk(node["right_child"])
            return idx

        depth = 0

        def max_depth(node, d=0):
            if "split_feature" not in node:
                return d
            return max(max_depth(node["left_child"], d + 1), max_depth(node["right_child"], d + 1))

        for t in trees:
            roots.append(walk(t["tree_structure"]))
            depth = max(depth, max_depth(t["tree_structure"]))
        dev = torch.device(device)
        self.depth = depth
        self.roots = torch.tensor(roots, dtype=torch.long, device=dev)
        self.feat = torch.tensor(feat, dtype=torch.long, device=dev)
        # float64-порог приводим к float32 «вниз», чтобы x<=thr совпадал с LightGBM для float32-входа
        t64 = np.asarray(thr, np.float64)
        t32 = t64.astype(np.float32)
        t32 = np.where(t32.astype(np.float64) > t64, np.nextafter(t32, -np.inf, dtype=np.float32), t32)
        self.thr = torch.tensor(t32, dtype=torch.float32, device=dev)
        self.left = torch.tensor(left, dtype=torch.long, device=dev)
        self.right = torch.tensor(right, dtype=torch.long, device=dev)
        self.dleft = torch.tensor(dleft, dtype=torch.bool, device=dev)
        self.value = torch.tensor(value, dtype=torch.float32, device=dev)
        self.device = dev

    @torch.inference_mode()
    def raw(self, X: torch.Tensor, chunk: int = 65536) -> torch.Tensor:
        X = X.to(self.device, torch.float32)
        out = torch.empty(len(X), dtype=torch.float32, device=self.device)
        n_trees = len(self.roots)
        for s in range(0, len(X), chunk):
            x = X[s : s + chunk]
            node = self.roots.unsqueeze(0).expand(len(x), n_trees).clone()
            rows = torch.arange(len(x), device=self.device).unsqueeze(1).expand_as(node)
            for _ in range(self.depth):
                f = self.feat[node]
                v = x[rows, f]
                go_left = torch.where(torch.isnan(v), self.dleft[node], v <= self.thr[node])
                nxt = torch.where(go_left, self.left[node], self.right[node])
                node = nxt
            out[s : s + chunk] = self.value[node].sum(1, dtype=torch.float64).float()
        return out

    def predict(self, X) -> np.ndarray:
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        r = self.raw(X)
        if self.objective.startswith(("binary", "cross_entropy", "xentropy")):
            r = torch.sigmoid(r)
        return r.cpu().numpy()
