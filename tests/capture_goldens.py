#!/usr/bin/env python3
"""Capture golden reference outputs for the refactoring test suite.

Run via `make goldens`. This regenerates everything under tests/golden/ --
only run it deliberately (before the port, or when intentionally changing
the reference behaviour), never as part of a normal test run.

Layer 1: R-transform goldens.
    Deterministic silhouette images (synthetic shapes + Otsu-thresholded
    test/r.jpg) are written as text matrices, then fed through the compiled
    C++ harness (bin/rtransform_dump) built from the original thesis code.

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

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

REPO = Path(__file__).resolve().parent.parent
GOLDEN = REPO / "tests" / "golden"
HARNESS = REPO / "bin" / "rtransform_dump"

N_ANGLES = 64
HOG_DIM = 168
TRAIN_COUNTS = {"walk": 140, "run": 87, "fall": 120}  # order matters, see imageshow.py


def write_matrix(path: Path, m: np.ndarray) -> None:
    with open(path, "w") as f:
        f.write(f"{m.shape[0]} {m.shape[1]}\n")
        for row in m:
            f.write(" ".join(str(int(v)) for v in row) + "\n")


def synthetic_silhouettes() -> dict:
    sils = {}

    rect_upright = np.zeros((128, 128), np.uint8)
    rect_upright[20:110, 50:80] = 255
    sils["rect_upright"] = rect_upright

    rect_lying = np.zeros((128, 128), np.uint8)
    rect_lying[50:80, 20:110] = 255
    sils["rect_lying"] = rect_lying

    yy, xx = np.mgrid[0:128, 0:128]
    circle = (((yy - 64) ** 2 + (xx - 64) ** 2) <= 40**2).astype(np.uint8) * 255
    sils["circle"] = circle

    # Real image, processed the way silhouetteDetectionCropped does:
    # Otsu threshold, then a 128x128 binary crop.
    gray = cv2.imread(str(REPO / "test" / "r.jpg"), cv2.IMREAD_GRAYSCALE)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sils["r_otsu"] = cv2.resize(otsu, (128, 128), interpolation=cv2.INTER_NEAREST)

    return sils


def capture_rtransform() -> None:
    if not HARNESS.exists():
        sys.exit(f"harness not built: {HARNESS} (run `make harness` first)")

    inputs_dir = GOLDEN / "inputs"
    out_dir = GOLDEN / "rtransform"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, sil in synthetic_silhouettes().items():
        in_path = inputs_dir / f"{name}.txt"
        write_matrix(in_path, sil)
        result = subprocess.run(
            [str(HARNESS), str(in_path), str(N_ANGLES)],
            capture_output=True, text=True, check=True,
        )
        (out_dir / f"{name}.txt").write_text(result.stdout)
        values = np.loadtxt(out_dir / f"{name}.txt")
        assert values.shape == (N_ANGLES,), f"{name}: unexpected shape {values.shape}"
        print(f"  rtransform/{name}: {N_ANGLES} values, max at angle {int(values.argmax())}")


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
    print("Capturing R-transform goldens (Layer 1)")
    capture_rtransform()
    print("Capturing k-NN goldens (Layer 2)")
    capture_knn()
    print(f"Done. Goldens written under {GOLDEN}")
