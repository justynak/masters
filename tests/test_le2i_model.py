"""Regression tests for the Phase 4 Le2i-trained classifier.

Uses only the committed data files (data/le2i_*.npz) -- the dataset videos
are not needed, so these run in CI. The recall floors pin the measured
Phase 4 quality; if a change drops below them, something regressed.
"""

from pathlib import Path

import numpy as np
import pytest

from falldetect.classifier import (
    LABELS,
    build_window_features,
    legacy_muhavi,
    train_classifier,
    train_from_windows,
)

REPO = Path(__file__).resolve().parent.parent
WINDOWS = REPO / "data" / "le2i_windows.npz"
TEST_VIDEOS = REPO / "data" / "le2i_test_videos.txt"

pytestmark = pytest.mark.skipif(
    not WINDOWS.exists(), reason="data/le2i_windows.npz not present"
)


def test_default_classifier_is_le2i_rt_stats():
    clf = train_classifier()
    assert clf.variant == "rt_stats"
    assert clf.needs_rtransforms and not clf.needs_hog


def test_legacy_classifier_still_available():
    clf = legacy_muhavi()
    assert clf.variant == "hog"
    assert clf.estimator.n_samples_fit_ == 347


def test_build_window_features_variants():
    rng = np.random.RandomState(0)
    rts = [rng.rand(64) for _ in range(20)]
    assert build_window_features("rt_stats", rtransforms=rts).shape == (128,)
    assert build_window_features("hog", hog=rng.rand(168)).shape == (168,)
    assert build_window_features("rt_stats", rtransforms=[]) is None
    assert build_window_features("hog", hog=None) is None
    with pytest.raises(ValueError):
        build_window_features("nope")


def test_held_out_recall_floors():
    """The Phase 4 measurement, reproduced from committed features:
    fall recall 55.6%, walk recall 98.2% on held-out videos."""
    clf = train_from_windows()
    data = np.load(WINDOWS)
    held_out = {line.strip() for line in TEST_VIDEOS.read_text().splitlines() if line.strip()}
    mask = np.isin(data["video"], sorted(held_out))
    assert mask.sum() > 200  # sanity: the split exists

    raw = data["rt_raw"][mask]
    y = data["label"][mask]
    X = np.hstack([raw.mean(axis=1), raw.std(axis=1)])
    pred = clf.estimator.predict(X)

    fall = y == LABELS.index("fall")
    walk = y == LABELS.index("walk")
    fall_recall = (pred[fall] == LABELS.index("fall")).mean()
    walk_recall = (pred[walk] == LABELS.index("walk")).mean()
    assert fall_recall >= 0.50, f"fall recall regressed: {fall_recall:.2%}"
    assert walk_recall >= 0.95, f"walk recall regressed: {walk_recall:.2%}"
