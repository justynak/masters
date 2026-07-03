"""Headless windowed fall-detection pipeline.

Mirrors ImageWidget.onTimerTimeout frame for frame: resize to 500x300,
BGR->RGB, horizontal flip, grayscale, blur, silhouette extraction; once 20
frames with a silhouette are buffered, select a keyframe, subtract the
background by hue difference, extract the keyframe silhouette's Canny-edge
HOG and classify it with the k-NN.

Two deliberate differences from the GUI, neither of which affects the label:
* the per-frame R-transform is only computed when collect_rtransforms=True
  (the GUI computes it and an LLE reduction whose output is then discarded);
* a missing background image can be synthesised as the temporal median of
  the video (median_background), instead of being a required file.

Run as a script to print per-window predictions for a video:
    python -m falldetect.pipeline video.avi [--background bg.png]
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .classifier import predict_label, train_classifier
from .features import hog_multiscale, r_transform, silhouette_cropped
from .keyframe import select_keyframe

FRAME_SIZE = (500, 300)  # (width, height), as in the GUI
WINDOW_SIZE = 20


@dataclass
class WindowResult:
    start_frame: int   # video frame index of the window's first silhouette frame
    end_frame: int     # ... and its last
    label: str | None  # None: no silhouette found in the keyframe


class Pipeline:
    """Push video frames (BGR, any size) and collect WindowResults.

    `background` is a BGR image as loaded by cv2.imread; it is resized to
    FRAME_SIZE and, following the original code, treated as RGB and NOT
    flipped -- faithful to how the GUI uses the loaded background file.
    """

    def __init__(self, background, classifier=None, collect_rtransforms=False):
        self.background = cv2.resize(background, FRAME_SIZE)
        self.classifier = classifier if classifier is not None else train_classifier()
        self.collect_rtransforms = collect_rtransforms

        self.frames = []
        self.frame_indices = []
        self.rtransforms = []
        self._frame_index = -1

    def push(self, frame_bgr):
        """Process one video frame; returns a WindowResult when a 20-frame
        window completes, else None."""
        self._frame_index += 1

        frame = cv2.resize(frame_bgr, FRAME_SIZE)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        sil = silhouette_cropped(gray)
        if sil is not None:
            self.frames.append(frame)
            self.frame_indices.append(self._frame_index)
            if self.collect_rtransforms:
                self.rtransforms.append(r_transform(sil, 64))

        if len(self.frames) < WINDOW_SIZE:
            return None

        result = WindowResult(self.frame_indices[0], self.frame_indices[-1],
                              self._classify_window())
        self.frames = []
        self.frame_indices = []
        self.rtransforms = []
        return result

    def _classify_window(self):
        keyframe = select_keyframe(self.frames)
        keyframe = cv2.cvtColor(keyframe, cv2.COLOR_RGB2HSV)
        background = cv2.cvtColor(self.background, cv2.COLOR_RGB2HSV)

        h = cv2.split(keyframe)[0] - cv2.split(background)[0]
        h = cv2.GaussianBlur(h, (3, 3), 0)

        silhouette = silhouette_cropped(h)
        if silhouette is None:
            return None

        edges = cv2.Canny(silhouette, 50, 5)
        return predict_label(self.classifier, hog_multiscale(edges, 8))


def median_background(video_path, samples=25):
    """Temporal median of frames sampled across the video: a person-free
    background estimate for scenes where no empty frame is available."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"cannot open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    picks = np.linspace(0, max(total - 1, 0), min(samples, max(total, 1)), dtype=int)

    frames = []
    for idx in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.resize(frame, FRAME_SIZE))
    cap.release()
    if not frames:
        raise IOError(f"no readable frames in {video_path}")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def run_video(video_path, background=None, start=0, end=None, classifier=None):
    """Run the pipeline over a video (or a frame range of it); returns a
    list of WindowResults. If background is None, a temporal median over the
    whole video is used."""
    if background is None:
        background = median_background(video_path)

    pipeline = Pipeline(background, classifier=classifier)
    results = []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"cannot open {video_path}")
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    index = start
    while end is None or index < end:
        ok, frame = cap.read()
        if not ok:
            break
        result = pipeline.push(frame)
        if result is not None:
            # report positions in original video coordinates
            result.start_frame += start
            result.end_frame += start
            results.append(result)
        index += 1

    cap.release()
    return results


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Run the fall-detection pipeline over a video")
    parser.add_argument("video")
    parser.add_argument("--background", help="empty-scene image (default: temporal median of the video)")
    args = parser.parse_args(argv)

    background = cv2.imread(args.background) if args.background else None
    if args.background and background is None:
        parser.error(f"cannot read background image {args.background}")

    for result in run_video(args.video, background):
        label = result.label if result.label is not None else "(no keyframe silhouette)"
        print(f"frames {result.start_frame:5d}-{result.end_frame:5d}: {label}")


if __name__ == "__main__":
    main()
