#!/usr/bin/env python3
"""Segment Le2i fall-dataset videos into labelled walk/fall clips.

The Le2i dataset (Charfi et al., "Optimised spatio-temporal descriptors for
real-time fall detection", JEI 2013) ships per-video annotations:
    line 1: frame number where the fall starts (0 = no fall)
    line 2: frame number where the fall ends
    then per frame: frame,pose_flag,x1,y1,x2,y2   (person bounding box)

Segments are derived as:
  fall -- from fall_start to max(fall_end, fall_start+25)+10 (the pipeline
          needs >= 20 silhouette frames per window, and the immediate
          lying-down aftermath belongs to the fall event for alerting);
  walk -- contiguous pre-fall runs where the person is present, the box is
          upright (height > 1.2*width, standing rather than sitting or
          lying) and the box centre moves (actually walking).

Run as a script to print the segment inventory:
    python scripts/le2i.py [dataset_root]
"""

import re
import sys
from dataclasses import dataclass
from pathlib import Path

# the OpenCV-readable mirror produced by scripts/prepare_le2i.py
# (the raw download's rawvideo AVIs crash opencv-python 5.0)
DEFAULT_ROOT = Path.home() / "datasets" / "le2i" / "converted"

MIN_WALK_LEN = 25      # frames; pipeline needs >= 20 silhouettes per window
FALL_MIN_LEN = 25
FALL_TAIL = 10
UPRIGHT_RATIO = 1.2    # box height must exceed 1.2 * width
MIN_TRAVEL = 20.0      # px the box centre must move across a walk segment


@dataclass
class Segment:
    video: Path
    label: str          # "walk" | "fall"
    start: int          # frame index, inclusive
    end: int            # frame index, exclusive


def parse_annotation(path):
    """Returns (fall_start, fall_end, boxes) with boxes[frame] = (x1,y1,x2,y2)."""
    lines = Path(path).read_text().split("\n")
    if "," in lines[0]:
        # one file in the dataset lacks the fall start/end header; without
        # it the ground truth is unknown, so the video is skipped
        raise ValueError(f"{path}: no fall-window header")
    fall_start, fall_end = int(lines[0]), int(lines[1])
    boxes = {}
    for line in lines[2:]:
        parts = line.strip().split(",")
        if len(parts) == 6:
            frame, _flag, x1, y1, x2, y2 = map(int, parts)
            boxes[frame] = (x1, y1, x2, y2)
    return fall_start, fall_end, boxes


def _is_upright(box):
    x1, y1, x2, y2 = box
    w, h = abs(x2 - x1), abs(y2 - y1)
    return w > 0 and h > 0 and h > UPRIGHT_RATIO * w


def _center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _walk_segments(video, boxes, before_frame):
    """Contiguous runs of upright, present, moving person before the fall."""
    frames = sorted(f for f in boxes if f < before_frame)
    segments = []
    run = []
    for f in frames:
        if _is_upright(boxes[f]) and (not run or f == run[-1] + 1):
            run.append(f)
            continue
        segments.extend(_close_run(video, run, boxes))
        run = [f] if _is_upright(boxes[f]) else []
    segments.extend(_close_run(video, run, boxes))
    return segments


def _close_run(video, run, boxes):
    if len(run) < MIN_WALK_LEN:
        return []
    xs, ys = zip(*(_center(boxes[f]) for f in run))
    travel = max(max(xs) - min(xs), max(ys) - min(ys))
    if travel < MIN_TRAVEL:
        return []  # standing still / sitting in place
    return [Segment(video, "walk", run[0], run[-1] + 1)]


def segments_for_video(video, annotation):
    fall_start, fall_end, boxes = parse_annotation(annotation)
    has_fall = 0 < fall_start < fall_end

    segments = _walk_segments(video, boxes, fall_start if has_fall else 10**9)
    if has_fall:
        end = max(fall_end, fall_start + FALL_MIN_LEN) + FALL_TAIL
        segments.append(Segment(video, "fall", fall_start, end))
    return segments


def find_segments(root=DEFAULT_ROOT):
    """Scan the dataset for scenes that ship annotations."""
    segments = []
    for ann_dir in sorted(Path(root).glob("*/*/Annotation_files")):
        video_dir = ann_dir.parent / "Videos"
        for ann in sorted(ann_dir.glob("*.txt"), key=lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)]):
            video = video_dir / (ann.stem + ".avi")
            if video.exists():
                try:
                    segments.extend(segments_for_video(video, ann))
                except ValueError as e:
                    print(f"skipping: {e}", file=sys.stderr)
    return segments


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROOT
    segments = find_segments(root)
    by_label = {}
    for seg in segments:
        by_label.setdefault(seg.label, []).append(seg)
    for label, segs in sorted(by_label.items()):
        frames = sum(s.end - s.start for s in segs)
        print(f"{label}: {len(segs)} segments, {frames} frames total")
    videos = len({s.video for s in segments})
    print(f"from {videos} videos under {root}")
