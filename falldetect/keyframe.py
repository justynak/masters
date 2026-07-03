"""Keyframe selection among buffered frames.

Port of ImageWidget.getKeyframe / getFrameHistogram /
calculateHistogramDistance: among frames 0/4/9/14 of the window, pick the
pair with the smallest Canberra histogram distance in HSV space and return
its first member (converted back to RGB).
"""

import cv2
import numpy as np
import scipy.spatial.distance


def frame_histogram(frame):
    hist = cv2.calcHist([frame], [0], None, [8], [0, 256])
    return cv2.normalize(hist, None).flatten()


def histogram_distance(frame0, frame1):
    return scipy.spatial.distance.canberra(
        frame_histogram(frame0), frame_histogram(frame1)
    )


def select_keyframe(frames):
    """frames: RGB frames of one window (at least 15 entries)."""
    frame0 = cv2.cvtColor(frames[0], cv2.COLOR_RGB2HSV)
    frame1 = cv2.cvtColor(frames[4], cv2.COLOR_RGB2HSV)
    frame2 = cv2.cvtColor(frames[9], cv2.COLOR_RGB2HSV)
    frame3 = cv2.cvtColor(frames[14], cv2.COLOR_RGB2HSV)

    hd0 = histogram_distance(frame0, frame1)
    hd1 = histogram_distance(frame1, frame2)
    hd2 = histogram_distance(frame2, frame3)

    hd = np.argmin([hd0, hd1, hd2])

    keyframe = frame0
    if hd == 1:
        keyframe = frame1
    elif hd == 2:
        keyframe = frame2

    return cv2.cvtColor(keyframe, cv2.COLOR_HSV2RGB)
