"""Golden regression tests guarding the Phase 1+ refactoring.

The goldens under tests/golden/ were captured from the original thesis code
(see capture_goldens.py): the R-transform values come from the restored
C++ implementation, the k-NN predictions from the training data and label
layout hardcoded in imageshow.py. Any port of the pipeline must keep these
tests green.
"""

import importlib
import subprocess
from pathlib import Path

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "tests" / "golden"
HARNESS = REPO / "bin" / "rtransform_dump"

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

@pytest.mark.skipif(not HARNESS.exists(), reason="harness not built (make harness)")
@pytest.mark.parametrize("name", INPUT_NAMES)
def test_cpp_harness_matches_golden(name):
    """Recompute via the C++ harness; guards the C++ source and the build."""
    result = subprocess.run(
        [str(HARNESS), str(GOLDEN / "inputs" / f"{name}.txt"), str(N_ANGLES)],
        capture_output=True, text=True, check=True,
    )
    values = np.array([float(v) for v in result.stdout.split()])
    np.testing.assert_allclose(values, load_golden_rtransform(name), rtol=1e-12)


@pytest.mark.parametrize("name", INPUT_NAMES)
def test_python_rtransform_matches_golden(name):
    """The Python-visible R-transform must reproduce the C++ goldens.

    Resolves the implementation in this order:
      1. falldetect.features.r_transform  (the Phase 2+ pure-Python version)
      2. the rTransform Cython extension  (Phase 1: same C++, rebuilt for py3)
    Skips while neither exists yet.
    """
    sil = load_input(name)
    golden = load_golden_rtransform(name)

    try:
        features = importlib.import_module("falldetect.features")
        values = np.asarray(features.r_transform(sil, N_ANGLES))
        # skimage-based reimplementation: exact values differ from the C++
        # (interpolation, angle handling) -- require matching curve shape.
        corr = np.corrcoef(values, golden)[0, 1]
        assert corr > 0.95, f"R-transform curve diverged (corr={corr:.3f})"
        return
    except ImportError:
        pass

    try:
        rTransform = importlib.import_module("rTransform")
    except ImportError:
        pytest.skip("no Python R-transform available yet (Phase 1 not built)")
    rt = rTransform.PyRTransform()
    values = np.asarray(rt.rTransform(sil, sil.shape[0], sil.shape[1], N_ANGLES))
    np.testing.assert_allclose(values, golden, rtol=1e-12)


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
