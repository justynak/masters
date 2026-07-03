"""Behaviour classifiers and window-feature construction.

Two feature variants are supported at inference time:

    hog      -- 168-dim multi-scale HOG of the keyframe's background-difference
                silhouette (the original thesis feature)
    rt_stats -- mean and std over the window's 20 per-frame R-transforms
                (128-dim). The R-transform is computed from the per-frame Otsu
                silhouette, which is far more reliable than the hue-difference
                keyframe silhouette; this variant was selected in Phase 4
                (see scripts/train_le2i.py and REFACTORING.md).

train_classifier() returns the Le2i-trained model (data/le2i_train.npz) when
available, falling back to the legacy MuHAVi model (test/data.txt).
"""

from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LABELS = ("walk", "run", "fall")

REPO = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO / "test" / "data.txt"      # legacy MuHAVi HOG features
LE2I_TRAIN = REPO / "data" / "le2i_train.npz"  # written by scripts/train_le2i.py
TRAIN_COUNTS = (140, 87, 120)                  # label layout of data.txt


def build_window_features(variant, hog=None, rtransforms=None):
    """Feature vector for one 20-frame window, or None if the inputs the
    variant needs are unavailable."""
    if variant == "hog":
        if hog is None:
            return None
        return np.asarray(hog, np.float32)
    if variant == "rt_stats":
        if rtransforms is None or len(rtransforms) == 0:
            return None
        rt = np.asarray(rtransforms, np.float32)
        return np.concatenate([rt.mean(axis=0), rt.std(axis=0)])
    raise ValueError(f"unknown feature variant {variant!r}")


class FallClassifier:
    """A fitted estimator plus the feature variant it consumes."""

    def __init__(self, estimator, variant):
        self.estimator = estimator
        self.variant = variant

    @property
    def needs_hog(self):
        return self.variant == "hog"

    @property
    def needs_rtransforms(self):
        return self.variant == "rt_stats"

    def predict_window(self, hog=None, rtransforms=None):
        """Label for one window, or None if the needed features are missing."""
        X = build_window_features(self.variant, hog, rtransforms)
        if X is None:
            return None
        label = self.estimator.predict(X.reshape(1, -1))[0]
        return LABELS[int(round(float(label)))]


def load_training_data(path=DEFAULT_DATA):
    """Legacy MuHAVi training matrix (labels from the hardcoded layout)."""
    X = np.loadtxt(path).astype(np.float32)
    y = np.concatenate(
        [np.full(count, i, np.float32) for i, count in enumerate(TRAIN_COUNTS)]
    )
    if len(X) != len(y):
        raise ValueError(f"{path}: {len(X)} rows, expected {len(y)}")
    return X, y


def legacy_muhavi(path=DEFAULT_DATA):
    """The original thesis classifier: kNN(12) on MuHAVi HOG features."""
    X, y = load_training_data(path)
    estimator = KNeighborsClassifier(12)
    estimator.fit(X, y)
    return FallClassifier(estimator, "hog")


def train_from_windows(path=LE2I_TRAIN):
    """Classifier from a persisted training set (scripts/train_le2i.py)."""
    data = np.load(path)
    estimator = KNeighborsClassifier(int(data["k"]))
    if bool(data["scaled"]):
        estimator = make_pipeline(StandardScaler(), estimator)
    estimator.fit(data["X"], data["y"])
    return FallClassifier(estimator, str(data["variant"]))


def train_classifier():
    """The default classifier: Le2i-trained if available, else legacy."""
    if LE2I_TRAIN.exists():
        return train_from_windows()
    return legacy_muhavi()
