#!/usr/bin/env python3
"""Evaluate the current classifier on labelled Le2i segments.

For every walk/fall segment (see le2i.py) the headless pipeline is run over
the segment's frame range; each completed 20-silhouette window contributes
one prediction, compared against the segment's ground-truth label. Prints a
confusion matrix and per-class recall.

Usage:
    python scripts/evaluate.py [dataset_root] [--limit N] [--videos FILE]

--videos restricts evaluation to the listed videos (paths relative to the
dataset root, one per line) -- pass data/le2i_test_videos.txt to measure on
the videos held out by scripts/train_le2i.py.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from falldetect.classifier import LABELS, train_classifier
from falldetect.pipeline import median_background, run_video
from scripts.le2i import DEFAULT_ROOT, find_segments


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", nargs="?", default=DEFAULT_ROOT, type=Path)
    parser.add_argument("--limit", type=int, help="evaluate only the first N segments")
    parser.add_argument("--videos", type=Path,
                        help="only videos listed in this file (relative to root)")
    args = parser.parse_args(argv)

    segments = find_segments(args.root)
    if args.videos:
        allowed = {line.strip() for line in args.videos.read_text().splitlines() if line.strip()}
        segments = [s for s in segments if str(s.video.relative_to(args.root)) in allowed]
    if args.limit:
        segments = segments[: args.limit]
    if not segments:
        sys.exit(f"no labelled segments found under {args.root}")

    classifier = train_classifier()
    backgrounds = {}  # per-video temporal median, computed once

    confusion = Counter()  # (ground_truth, prediction) -> windows
    no_prediction = Counter()
    videos = set()

    for i, seg in enumerate(segments, 1):
        if seg.video not in backgrounds:
            backgrounds[seg.video] = median_background(seg.video)
        videos.add(seg.video)

        results = run_video(seg.video, backgrounds[seg.video],
                            start=seg.start, end=seg.end, classifier=classifier)
        for result in results:
            if result.label is None:
                no_prediction[seg.label] += 1
            else:
                confusion[(seg.label, result.label)] += 1
        print(f"\r{i}/{len(segments)} segments", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    gt_labels = sorted({gt for gt, _ in confusion} | set(no_prediction))
    print(f"\n{len(segments)} segments from {len(videos)} videos, "
          f"{sum(confusion.values())} classified windows, "
          f"{sum(no_prediction.values())} windows without keyframe silhouette\n")

    header = "gt \\ pred"
    print(f"{header:>10} " + " ".join(f"{p:>6}" for p in LABELS) + f" {'none':>6}")
    for gt in gt_labels:
        row = [confusion[(gt, p)] for p in LABELS]
        print(f"{gt:>10} " + " ".join(f"{c:>6}" for c in row) + f" {no_prediction[gt]:>6}")

    print()
    for gt in gt_labels:
        total = sum(confusion[(gt, p)] for p in LABELS)
        if total:
            recall = confusion[(gt, gt)] / total
            print(f"{gt}: recall {recall:.2%} over {total} classified windows")


if __name__ == "__main__":
    main()
