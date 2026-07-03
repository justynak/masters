"""Tests for the headless pipeline (falldetect.pipeline) -- no Qt involved."""

import cv2
import numpy as np
import pytest

from falldetect.classifier import LABELS, train_classifier
from falldetect.pipeline import Pipeline, WindowResult, median_background, run_video


@pytest.fixture(scope="module")
def classifier():
    return train_classifier()


SCENE_BGR = (120, 60, 30)   # dark blue-ish scene
PERSON_BGR = (30, 60, 220)  # red-ish figure: distinct hue AND brightness

# The background subtraction works on the hue channel, so the synthetic
# scene must be chromatic -- on pure gray frames it sees nothing.


def synthetic_frames(n=30, size=(300, 500)):
    """Colored scene with a bright person-sized rectangle walking across."""
    frames = []
    for t in range(n):
        frame = np.full((*size, 3), SCENE_BGR, np.uint8)
        x = 100 + t * 5
        frame[80:230, x:x + 60] = PERSON_BGR
        frames.append(frame)
    return frames


def dark_background(size=(300, 500)):
    return np.full((*size, 3), SCENE_BGR, np.uint8)


def test_pipeline_produces_window_result(classifier):
    pipeline = Pipeline(dark_background(), classifier=classifier)
    results = [r for f in synthetic_frames(30) if (r := pipeline.push(f)) is not None]

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, WindowResult)
    assert result.label in LABELS
    assert 0 <= result.start_frame < result.end_frame < 30
    # buffer was reset after the window completed
    assert len(pipeline.frames) < 20


def test_pipeline_no_silhouette_no_window(classifier):
    # all-black frames: Otsu finds nothing, so no window ever completes.
    # (NB a uniform non-black frame would fool Otsu into an all-white
    # "silhouette" -- a known quirk inherited from the original code.)
    pipeline = Pipeline(dark_background(), classifier=classifier)
    black = np.zeros((300, 500, 3), np.uint8)
    for _ in range(30):
        assert pipeline.push(black) is None


def test_median_background_and_run_video(classifier, tmp_path):
    video = str(tmp_path / "synthetic.avi")
    writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), 20, (500, 300))
    for frame in synthetic_frames(40):
        writer.write(frame)
    writer.release()

    # the median over the clip must recover the person-free dark scene
    bg = median_background(video)
    assert bg.shape == (300, 500, 3)
    assert np.abs(bg.astype(int) - np.array(SCENE_BGR)).max() <= 5

    results = run_video(video, classifier=classifier)
    assert len(results) >= 1
    assert all(r.label in LABELS or r.label is None for r in results)

    # frame-range restriction: too few frames for a window -> no results
    assert run_video(video, start=0, end=10, classifier=classifier) == []
