"""Headless smoke tests for the ported PyQt application (Phase 1).

Runs the GUI offscreen and pushes a synthetic frame through the same
methods the timer callback uses: silhouette extraction -> HOG -> k-NN.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

import imageshow
from falldetect.features import r_transform


@pytest.fixture(scope="module")
def widget():
    app = QApplication.instance() or QApplication([])
    return imageshow.ImageWidget()


def synthetic_gray_frame():
    """A 300x500 dark frame with a bright person-sized rectangle."""
    frame = np.full((300, 500), 30, np.uint8)
    frame[80:230, 200:260] = 220
    return frame


def test_classifier_is_trained(widget):
    assert widget.classifier.n_samples_fit_ == 347


def test_silhouette_extraction(widget):
    sil = widget.silhouetteDetectionCropped(synthetic_gray_frame())
    assert sil is not None
    assert sil.shape == (128, 128)
    assert set(np.unique(sil)) <= {0, 255}


def test_silhouette_extraction_empty_frame(widget):
    assert widget.silhouetteDetectionCropped(np.zeros((300, 500), np.uint8)) is None


def test_hog_predict_path(widget):
    """The exact feature+classify path of onTimerTimeout."""
    import cv2

    sil = widget.silhouetteDetectionCropped(synthetic_gray_frame())
    edges = cv2.Canny(sil, 50, 5)
    sdeg = imageshow.HOG(edges, 8)
    assert sdeg.shape == (168,)

    arr = np.array(sdeg, np.float32)
    result = int(round(widget.classifier.predict(arr.reshape(1, -1))[0]))
    assert result in (0, 1, 2)
    assert widget.behaviourLabelTable[result] in ("walk", "run", "fall")


def test_rtransform_on_extracted_silhouette(widget):
    sil = widget.silhouetteDetectionCropped(synthetic_gray_frame())
    values = np.asarray(r_transform(sil, 64))
    assert values.shape == (64,)
    assert values.max() == pytest.approx(1.0)


def test_keyframe_selection(widget):
    rng = np.random.RandomState(0)
    frames = [rng.randint(0, 255, (300, 500, 3), np.uint8) for _ in range(20)]
    widget.frames = frames
    keyframe = widget.getKeyframe(frames)
    assert keyframe.shape == (300, 500, 3)
    widget.frames = []
