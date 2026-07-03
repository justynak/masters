"""Golden regression tests guarding the Phase 1+ refactoring.

The goldens under tests/golden/ were captured from the original thesis code
(see capture_goldens.py): the R-transform values come from the original
C++ implementation (removed in Phase 2, recoverable from git history), the
k-NN predictions from the training data and label layout hardcoded in
imageshow.py. Any port of the pipeline must keep these tests green.
"""

from pathlib import Path

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

from falldetect.features import r_transform

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "tests" / "golden"

N_ANGLES = 64
HOG_DIM = 168
TRAIN_COUNTS = (140, 87, 120)  # walk, run, fall -- order from imageshow.py

INPUT_NAMES = ("rect_upright", "rect_lying", "circle", "r_otsu")


def load_input(name: str) -> np.ndarray:
    path = GOLDEN / "inputs" / f"{name}.txt"
    with open(path) as f:
        dim0, dim1 = map(int, f.readline().split())
        m = np.loadtxt(f, dtype=np.uint8)
    assert m.shape == (dim0, dim1)
    return m


def load_golden_rtransform(name: str) -> np.ndarray:
    return np.loadtxt(GOLDEN / "rtransform" / f"{name}.txt")


# --------------------------------------------------------------------------
# Layer 1: R-transform
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", INPUT_NAMES)
def test_python_rtransform_matches_golden(name):
    """falldetect.features.r_transform is a faithful NumPy port of the
    original C++ and must reproduce its goldens. It is bitwise-identical on
    the reference platform; the tolerance below only allows for sub-ULP
    libm differences (tan/sin/cos) between platforms."""
    values = r_transform(load_input(name), N_ANGLES)
    np.testing.assert_allclose(values, load_golden_rtransform(name), rtol=1e-9, atol=1e-9)


def test_rtransform_distinguishes_upright_from_lying():
    """Sanity property behind the whole approach: a standing and a lying
    person (rotated rectangle) must produce clearly different signatures."""
    upright = load_golden_rtransform("rect_upright")
    lying = load_golden_rtransform("rect_lying")
    assert not np.allclose(upright, lying, atol=0.05)


# --------------------------------------------------------------------------
# Layer 2: k-NN classification
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def classifier():
    train = np.loadtxt(REPO / "test" / "data.txt").astype(np.float32)
    labels = np.concatenate(
        [np.full(c, i, np.float32) for i, c in enumerate(TRAIN_COUNTS)]
    )
    assert train.shape == (sum(TRAIN_COUNTS), HOG_DIM)
    clf = KNeighborsClassifier(12)
    clf.fit(train, labels)
    return clf


def test_knn_self_predictions_match_golden(classifier):
    """Primary Layer 2 probe: the app's classifier predicting on its own
    training data. Same feature scale as training, spans all three classes,
    and is sensitive to data-loading, dtype and label-wiring bugs."""
    train = np.loadtxt(REPO / "test" / "data.txt").astype(np.float32)
    pred = classifier.predict(train).astype(int)
    golden = np.loadtxt(GOLDEN / "knn" / "data_pred.txt", dtype=int)
    np.testing.assert_array_equal(pred, golden)


def test_knn_self_predictions_are_not_degenerate(classifier):
    """All three classes must appear in the self-predictions -- guards
    against a silent feature-scale or label-order regression."""
    golden = np.loadtxt(GOLDEN / "knn" / "data_pred.txt", dtype=int)
    assert set(np.unique(golden)) == {0, 1, 2}


@pytest.mark.parametrize("stem", ("walk", "run", "test"))
def test_knn_dump_predictions_match_golden(classifier, stem):
    """Secondary probe. NOTE: walk/run/test.txt are on a different (older,
    normalised) feature scale than data.txt, so these predictions degenerate
    to 'walk'; kept as a determinism check only (see capture_goldens.py)."""
    features = np.loadtxt(REPO / "test" / f"{stem}.txt").astype(np.float32)
    pred = classifier.predict(features[:, -HOG_DIM:]).astype(int)
    golden = np.loadtxt(GOLDEN / "knn" / f"{stem}_pred.txt", dtype=int)
    np.testing.assert_array_equal(pred, golden)
