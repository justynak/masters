#!/usr/bin/env python3
"""Extract per-window features from the labelled Le2i segments.

Runs the pipeline over every walk/fall segment (see le2i.py) and stores, for
each completed 20-silhouette window:
    hog        -- the 168-dim keyframe HOG (what the classifier consumes)
    rt         -- 20-dim LLE reduction of the window's per-frame R-transforms
                  (the thesis' combined-feature ingredient)
    rt_raw     -- the raw 20x64 per-frame R-transform sequence
    label      -- 0 walk / 2 fall (indices into falldetect.classifier.LABELS)
    video      -- video identifier, for leakage-free video-level splits
    scene      -- scene name (Coffee_room_01, Home_01, ...)

Output: data/le2i_windows.npz (committed -- training and CI depend on it).

Usage:
    python scripts/extract_features.py [dataset_root]
"""

import sys
from pathlib import Path

import numpy as np
from sklearn import manifold

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from falldetect.classifier import LABELS
from falldetect.pipeline import median_background, run_video
from scripts.le2i import DEFAULT_ROOT, find_segments

OUT = REPO / "data" / "le2i_windows.npz"
WINDOW = 20


def reduce_rtransforms(rtransforms):
    """20 per-frame 64-dim R-transforms -> 20 values via 1-component LLE,
    replicating the original imageshow.py design (n_neighbors=7)."""
    lle = manifold.LocallyLinearEmbedding(
        n_neighbors=7, n_components=1, eigen_solver="auto", method="standard",
        random_state=0,
    )
    return lle.fit_transform(np.asarray(rtransforms)).ravel()


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROOT
    segments = find_segments(root)
    if not segments:
        sys.exit(f"no segments under {root}")

    hogs, rts, rt_raws, labels, videos, scenes = [], [], [], [], [], []
    backgrounds = {}
    skipped = 0

    for i, seg in enumerate(segments, 1):
        if seg.video not in backgrounds:
            backgrounds[seg.video] = median_background(seg.video)

        results = run_video(seg.video, backgrounds[seg.video],
                            start=seg.start, end=seg.end,
                            collect_rtransforms=True, collect_features=True)
        for result in results:
            if result.hog is None or len(result.rtransforms) != WINDOW:
                skipped += 1
                continue
            hogs.append(np.asarray(result.hog, np.float32))
            rts.append(reduce_rtransforms(result.rtransforms).astype(np.float32))
            rt_raws.append(np.asarray(result.rtransforms, np.float32))
            labels.append(LABELS.index(seg.label))
            videos.append(str(seg.video.relative_to(root)))
            scenes.append(seg.video.relative_to(root).parts[0])
        print(f"\r{i}/{len(segments)} segments, {len(labels)} windows",
              end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    OUT.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        OUT,
        hog=np.stack(hogs),
        rt=np.stack(rts),
        rt_raw=np.stack(rt_raws),
        label=np.array(labels, np.int64),
        video=np.array(videos),
        scene=np.array(scenes),
    )
    counts = {LABELS[i]: int((np.array(labels) == i).sum()) for i in set(labels)}
    print(f"{len(labels)} windows ({counts}), {skipped} skipped -> {OUT}")


if __name__ == "__main__":
    main()
