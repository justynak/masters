#!/usr/bin/env python3
"""Train and compare classifier variants on the extracted Le2i windows.

Uses a video-level train/test split (GroupShuffleSplit) so windows of the
same video never appear on both sides. Compares:

    hog            -- 168-dim HOG, KNeighborsClassifier(12) (thesis setup)
    hog+scale      -- StandardScaler + kNN(12)
    combined+scale -- HOG concatenated with the 20-dim LLE-reduced
                      R-transform sequence (the thesis' combined feature)
    rt_stats       -- mean+std over the window's per-frame R-transforms
                      (128-dim), StandardScaler + kNN(5); Phase 4 winner

Writes the selected variant's training rows to data/le2i_train.npz (consumed
by falldetect.classifier.train_classifier) and the held-out video list to
data/le2i_test_videos.txt (for end-to-end evaluation on unseen videos).
Only 'hog' and 'rt_stats' can be built at inference time; the others are
reported for comparison but cannot be persisted.

Usage:
    python scripts/train_le2i.py [--select VARIANT]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from falldetect.classifier import LABELS

WINDOWS = REPO / "data" / "le2i_windows.npz"
TRAIN_OUT = REPO / "data" / "le2i_train.npz"
TEST_VIDEOS_OUT = REPO / "data" / "le2i_test_videos.txt"

SEED = 0
TEST_FRACTION = 0.3


PERSISTABLE = ("hog", "rt_stats")  # variants falldetect.classifier can rebuild


def variants(data):
    hog, rt, raw = data["hog"], data["rt"], data["rt_raw"]
    rt_stats = np.hstack([raw.mean(axis=1), raw.std(axis=1)])
    return {
        "hog": (hog, False, 12),
        "hog+scale": (hog, True, 12),
        "combined+scale": (np.hstack([rt, hog]), True, 12),
        "rt_stats": (rt_stats, True, 5),
    }


def make_classifier(scaled, k):
    knn = KNeighborsClassifier(k)
    return make_pipeline(StandardScaler(), knn) if scaled else knn


def evaluate_variant(X, scaled, k, y, train_idx, test_idx):
    clf = make_classifier(scaled, k)
    clf.fit(X[train_idx], y[train_idx])
    pred = clf.predict(X[test_idx])
    present = sorted(set(y))
    recalls = recall_score(y[test_idx], pred, labels=present, average=None, zero_division=0)
    cm = confusion_matrix(y[test_idx], pred, labels=present)
    return dict(zip((LABELS[i] for i in present), recalls)), cm, present


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--select", help="variant to persist (default: best macro recall)")
    args = parser.parse_args(argv)

    data = np.load(WINDOWS)
    y, groups = data["label"], data["video"]

    split = GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=SEED)
    train_idx, test_idx = next(split.split(y, y, groups))

    def describe(idx):
        return {LABELS[i]: int((y[idx] == i).sum()) for i in sorted(set(y[idx]))}

    print(f"split by video: {len(set(groups[train_idx]))} train videos "
          f"{describe(train_idx)} / {len(set(groups[test_idx]))} test videos "
          f"{describe(test_idx)}\n")

    scores = {}
    for name, (X, scaled, k) in variants(data).items():
        recalls, cm, present = evaluate_variant(X, scaled, k, y, train_idx, test_idx)
        macro = float(np.mean(list(recalls.values())))
        scores[name] = macro
        pretty = ", ".join(f"{k} {v:.2%}" for k, v in recalls.items())
        print(f"{name:>15}: macro recall {macro:.2%} ({pretty})")
        header = " ".join(f"{LABELS[i]:>6}" for i in present)
        print(f"{'':>15}  gt \\ pred {header}")
        for i, row in zip(present, cm):
            print(f"{'':>15}  {LABELS[i]:>9} " + " ".join(f"{c:>6}" for c in row))
        print()

    selected = args.select or max(
        (name for name in scores if name in PERSISTABLE), key=scores.get
    )
    if selected not in scores:
        sys.exit(f"unknown variant {selected!r}; choose from {sorted(scores)}")
    if selected not in PERSISTABLE:
        sys.exit(f"variant {selected!r} cannot be built at inference time; "
                 f"choose from {PERSISTABLE}")
    X, scaled, k = variants(data)[selected]

    np.savez_compressed(
        TRAIN_OUT,
        X=X[train_idx],
        y=y[train_idx],
        variant=np.array(selected),
        scaled=np.array(scaled),
        k=np.array(k),
    )
    test_videos = sorted(set(groups[test_idx]))
    TEST_VIDEOS_OUT.write_text("\n".join(test_videos) + "\n")
    print(f"selected {selected!r}: {len(train_idx)} training windows -> {TRAIN_OUT}")
    print(f"{len(test_videos)} held-out videos -> {TEST_VIDEOS_OUT}")


if __name__ == "__main__":
    main()
