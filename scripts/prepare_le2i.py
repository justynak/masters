#!/usr/bin/env python3
"""One-time conversion of the Le2i dataset for OpenCV consumption.

The dataset's AVIs contain uncompressed rawvideo (bgr24) plus a broken mp3
track; opencv-python 5.0's bundled ffmpeg segfaults decoding them. This
script re-encodes every video to MJPEG (near-lossless, -q:v 2), drops the
audio, and copies the annotation files, producing a self-contained mirror:

    ~/datasets/le2i/extracted/...  ->  ~/datasets/le2i/converted/...

Idempotent: existing outputs are skipped.

Usage:
    python scripts/prepare_le2i.py [extracted_root] [converted_root]
"""

import shutil
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

EXTRACTED = Path.home() / "datasets" / "le2i" / "extracted"
CONVERTED = Path.home() / "datasets" / "le2i" / "converted"


def convert(src_root=EXTRACTED, dst_root=CONVERTED):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    videos = sorted(src_root.glob("*/*/Videos/*.avi"))
    if not videos:
        sys.exit(f"no videos found under {src_root}")

    done = skipped = 0
    for src in videos:
        dst = dst_root / src.relative_to(src_root)
        if dst.exists():
            skipped += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(src),
             "-an", "-c:v", "mjpeg", "-q:v", "2", str(dst)],
            check=True, capture_output=True,
        )
        done += 1
        print(f"\r{done + skipped}/{len(videos)} videos", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    copied = 0
    for src in src_root.glob("*/*/Annotation_files/*.txt"):
        dst = dst_root / src.relative_to(src_root)
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1

    print(f"converted {done}, skipped {skipped} existing, "
          f"copied {copied} annotation files -> {dst_root}")


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else EXTRACTED
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else CONVERTED
    convert(src, dst)
