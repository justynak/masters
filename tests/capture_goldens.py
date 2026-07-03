#!/usr/bin/env python3
"""Capture golden reference outputs for the refactoring test suite.

Run via `make goldens`. This regenerates the k-NN goldens under
tests/golden/knn/ -- only run it deliberately (when intentionally changing
the reference behaviour), never as part of a normal test run.

Layer 1: R-transform goldens -- FROZEN, not regenerated here.
    tests/golden/rtransform/ was captured from the original C++
    implementation via a standalone harness before Phase 2 removed the
    C++/Cython layer. The NumPy port (falldetect.features.r_transform) is
    bitwise-identical to it. To regenerate from first principles, recover
    the C++ and harness from git history:
        git show c55ca97:test/RTransform.cpp
        git show a27bea8:tools/rtransform_dump.cpp

Layer 2: k-NN classification goldens.
    Trains KNeighborsClassifier(12) on test/data.txt with the label layout
    hardcoded in imageshow.py (140 walk, 87 run, 120 fall) and records:
      * predictions on data.txt itself (the primary regression probe -- it is
        on the same feature scale as training and spans all three classes);
      * predictions on the HOG part (last 168 columns) of walk/run/test.txt.
        NOTE: these dumps are on a different feature scale than data.txt
        (normalised [0,1] vs raw magnitudes up to ~474k), evidently produced
        by an older HOG variant with the normalisation still enabled, so the
        classifier degenerates to "walk" for all of them. They are kept only
        as a determinism check, not as an accuracy oracle.
"""

from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "tests" / "golden"

N_ANGLES = 64
HOG_DIM = 168
TRAIN_COUNTS = {"walk": 140, "run": 87, "fall": 120}  # order matters, see imageshow.py


def train_labels() -> np.ndarray:
    labels = []
    for i, (_, count) in enumerate(TRAIN_COUNTS.items()):
        labels.append(np.full(count, i, np.float32))
    return np.concatenate(labels)


def capture_knn() -> None:
    out_dir = GOLDEN / "knn"
    out_dir.mkdir(parents=True, exist_ok=True)

    train = np.loadtxt(REPO / "test" / "data.txt").astype(np.float32)
    labels = train_labels()
    assert train.shape == (sum(TRAIN_COUNTS.values()), HOG_DIM), train.shape

    clf = KNeighborsClassifier(12)
    clf.fit(train, labels)
    names = list(TRAIN_COUNTS)

    # Primary probe: the app's classifier on its own training data.
    pred = clf.predict(train).astype(int)
    np.savetxt(out_dir / "data_pred.txt", pred, fmt="%d")
    accuracy = float((pred == labels.astype(int)).mean())
    dist = {names[i]: int((pred == i).sum()) for i in range(len(names))}
    print(f"  knn/data_pred: {len(pred)} rows -> {dist}, train accuracy {accuracy:.3f}")

    # Secondary probes: stale-scale dumps (see module docstring).
    for stem in ("walk", "run", "test"):
        features = np.loadtxt(REPO / "test" / f"{stem}.txt").astype(np.float32)
        hog = features[:, -HOG_DIM:]  # last 168 columns; first 20 are the LLE part
        pred = clf.predict(hog).astype(int)
        np.savetxt(out_dir / f"{stem}_pred.txt", pred, fmt="%d")
        dist = {names[i]: int((pred == i).sum()) for i in range(len(names))}
        print(f"  knn/{stem}_pred: {len(pred)} rows -> {dist}")


if __name__ == "__main__":
    print("R-transform goldens (Layer 1) are frozen -- see module docstring")
    print("Capturing k-NN goldens (Layer 2)")
    capture_knn()
    print(f"Done. Goldens written under {GOLDEN}")
