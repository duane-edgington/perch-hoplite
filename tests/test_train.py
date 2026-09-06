"""Focused tests for train_proposed.py.

These tests use tiny fake DataManager objects and stub the downstream
perch_hoplite/F1 modules, so they can run without the full project database.
"""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import numpy as np
import pytest

from src import train as train_mod


@dataclass
class Batch:
    embedding: np.ndarray
    multihot: np.ndarray
    is_labeled_mask: np.ndarray
    idx: np.ndarray


class FakeDB:
    def __init__(self, embedding_dim=3):
        self.embedding_dim = embedding_dim

    def get_embedding_dim(self):
        return self.embedding_dim


class FakeDataManager:
    def __init__(self, train_ids=(0, 1, 2, 3), eval_ids=(4, 5)):
        self.db = FakeDB(3)
        self._train_ids = list(train_ids)
        self._eval_ids = list(eval_ids)

    def get_target_labels(self):
        return ["class_a", "class_b"]

    def get_train_test_split(self):
        return self._train_ids, self._eval_ids

    def batched_example_iterator(self, ids, add_weak_negatives, repeat=False):
        assert repeat is False
        ids = list(ids)
        if not ids:
            return iter(())
        emb = np.array([[i, i + 0.5, 1.0] for i in ids], dtype=np.float32)
        mh = np.array(
            [[1.0, 0.0] if i % 2 == 0 else [0.0, 1.0] for i in ids],
            dtype=np.float32,
        )
        # Explicit labels for all entries in this smoke test.
        ilm = np.ones_like(mh, dtype=np.float32)
        batch = Batch(emb, mh, ilm, np.asarray(ids))
        return iter((batch,))


def install_downstream_stubs(monkeypatch):
    classifier_mod = types.ModuleType("perch_hoplite.agile.classifier")

    class LinearClassifier:
        def __init__(self, beta, beta_bias, classes, embedding_model_config):
            self.beta = beta
            self.beta_bias = beta_bias
            self.classes = classes
            self.embedding_model_config = embedding_model_config

    classifier_mod.LinearClassifier = LinearClassifier

    metrics_mod = types.ModuleType("perch_hoplite.agile.metrics")
    metrics_mod.roc_auc = lambda **kwargs: {"macro": 0.75}
    metrics_mod.cmap = lambda **kwargs: {"macro": 0.50}

    agile_mod = types.ModuleType("perch_hoplite.agile")
    agile_mod.classifier = classifier_mod
    agile_mod.metrics = metrics_mod

    perch_mod = types.ModuleType("perch_hoplite")
    perch_mod.agile = agile_mod

    config_dict_mod = types.ModuleType("ml_collections.config_dict")

    class ConfigDict(dict):
        pass

    config_dict_mod.ConfigDict = ConfigDict
    ml_collections_mod = types.ModuleType("ml_collections")
    ml_collections_mod.config_dict = config_dict_mod

    f1_mod = types.ModuleType("src.f1_metrics")
    f1_mod.per_class_f1 = lambda logits, labels, target_labels: {
        "macro_f1_at_0": 0.60,
        "macro_f1_opt": 0.70,
        "per_class": {label: 0.60 for label in target_labels},
    }
    src_mod = types.ModuleType("src")
    src_mod.f1_metrics = f1_mod

    monkeypatch.setitem(sys.modules, "perch_hoplite", perch_mod)
    monkeypatch.setitem(sys.modules, "perch_hoplite.agile", agile_mod)
    monkeypatch.setitem(sys.modules, "perch_hoplite.agile.classifier", classifier_mod)
    monkeypatch.setitem(sys.modules, "perch_hoplite.agile.metrics", metrics_mod)
    monkeypatch.setitem(sys.modules, "ml_collections", ml_collections_mod)
    monkeypatch.setitem(sys.modules, "ml_collections.config_dict", config_dict_mod)
    monkeypatch.setitem(sys.modules, "src", src_mod)
    monkeypatch.setitem(sys.modules, "src.f1_metrics", f1_mod)


def test_invalid_loss_is_rejected():
    dm = FakeDataManager()
    with pytest.raises(ValueError, match="Unsupported loss"):
        train_mod.torch_train_linear_classifier(dm, 1e-3, 0.1, 2, loss="typo")


def test_empty_training_split_is_rejected():
    dm = FakeDataManager(train_ids=(), eval_ids=(1,))
    with pytest.raises(ValueError, match="Training split is empty"):
        train_mod.torch_train_linear_classifier(dm, 1e-3, 0.1, 2)


def test_empty_eval_split_is_rejected():
    dm = FakeDataManager(train_ids=(1,), eval_ids=())
    with pytest.raises(ValueError, match="Evaluation split is empty"):
        train_mod.torch_train_linear_classifier(dm, 1e-3, 0.1, 2)


def test_payload_nbytes_counts_actual_arrays():
    emb = np.zeros((50_000, 1536), dtype=np.float32)
    labels = np.zeros((50_000, 2), dtype=np.float32)
    mask = np.zeros((50_000, 2), dtype=np.float32)
    expected = emb.nbytes + labels.nbytes + mask.nbytes
    assert train_mod._payload_nbytes(emb, labels, mask) == expected
    # 50k x 1536 float32 embeddings are ~293 MiB, not ~4 GiB.
    assert emb.nbytes / 1024**2 == pytest.approx(292.96875)
    assert expected < 4 * 1024**3


@pytest.mark.parametrize("loss", ["bce", "hinge"])
def test_end_to_end_cpu_smoke(monkeypatch, loss):
    install_downstream_stubs(monkeypatch)
    dm = FakeDataManager()
    classifier, scores = train_mod.torch_train_linear_classifier(
        dm,
        learning_rate=1e-2,
        weak_neg_weight=0.1,
        num_train_steps=3,
        loss=loss,
    )
    assert classifier.beta.shape == (3, 2)
    assert classifier.beta_bias.shape == (2,)
    assert classifier.classes == ["class_a", "class_b"]
    assert set(scores) == {
        "top1_acc",
        "roc_auc",
        "cmap",
        "macro_f1",
        "macro_f1_opt",
        "per_class_f1",
    }
