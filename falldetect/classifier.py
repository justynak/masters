"""Training-data loading and the k-NN behaviour classifier.

Ports the classifier setup from ImageWidget.__init__: KNeighborsClassifier(12)
trained on the 168-dim HOG vectors in test/data.txt with the hardcoded label
layout 140 walk / 87 run / 120 fall.
"""

from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

LABELS = ("walk", "run", "fall")
TRAIN_COUNTS = (140, 87, 120)
DEFAULT_DATA = Path(__file__).resolve().parent.parent / "test" / "data.txt"


def load_training_data(path=DEFAULT_DATA):
    X = np.loadtxt(path).astype(np.float32)
    y = np.concatenate(
        [np.full(count, i, np.float32) for i, count in enumerate(TRAIN_COUNTS)]
    )
    if len(X) != len(y):
        raise ValueError(f"{path}: {len(X)} rows, expected {len(y)}")
    return X, y


def train_classifier(path=DEFAULT_DATA):
    X, y = load_training_data(path)
    clf = KNeighborsClassifier(12)
    clf.fit(X, y)
    return clf


def predict_label(classifier, hog_vector):
    """Predict a behaviour label from a 168-dim HOG vector, exactly the way
    the GUI does (float32, int(round(...)) of the numeric class)."""
    arr = np.asarray(hog_vector, np.float32).reshape(1, -1)
    return LABELS[int(round(classifier.predict(arr)[0]))]
