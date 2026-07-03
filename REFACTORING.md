# Refactoring Plan

Goal: make the fall-detection app run on a modern stack (Python 3.11+,
current OpenCV/scikit-learn), remove dead code and the C++/Cython build step,
and separate the processing pipeline from the GUI so it can be tested and
reused headlessly.

## Phase 1 — Modernise the runtime (make it run at all) ✅ DONE

The code was Python 2.7 with OpenCV 2.x APIs. Ported to Python 3.14 /
OpenCV 5 / PyQt5 5.15; the Cython extension rebuilds cleanly (setuptools
instead of the removed distutils) and reproduces the C++ goldens bitwise.
All bugs listed below are fixed; additionally `QImage` construction now uses
`Format_Grayscale8` with an explicit stride, `classifier.predict` gets the
2-D input modern sklearn requires, and `import scipy` became
`import scipy.spatial.distance` (no longer auto-imported). Stale Python 2
binaries (`rTransform.so`, `*.pyc`) were removed. Original checklist:

- Port to Python 3: `print` statements, integer division (`from __future__
  import division` is already used), remove the dead `import cv` and the bogus
  `from _mysql import result`.
- Update OpenCV calls:
  - `cv2.findContours` returns 2 values in OpenCV 4 (it already unpacks 2, but
    the order/semantics changed from 2.x — verify).
  - `cv2.BackgroundSubtractorMOG` → `cv2.createBackgroundSubtractorMOG2`
    (currently unused anyway — see Phase 3).
  - `cv2.normalize(hist)` needs explicit `dst`/`norm_type` arguments.
- Fix outright bugs:
  - `imageshow.py:122` — `NULL` → `None` (crashes if reached).
  - `imageshow.py:62` — hardcoded absolute path to `data.txt`; resolve
    relative to the module instead.
  - `imageshow.py:161-162` — `self.background` is overwritten with the
    keyframe itself (double RGB2HSV conversion), so the loaded background
    image is ignored. Decide the intended behaviour (probably: keep the
    background loaded at Play time) and fix.
- Pin dependencies in `pyproject.toml` / `requirements.txt`.

## Phase 2 — Replace the C++/Cython extension with NumPy/scikit-image

The C++ exists only because a triple Python loop was too slow in Py2. The same
R-transform is a few lines on top of `skimage.transform.radon`:

```python
from skimage.transform import radon
import numpy as np

def r_transform(silhouette: np.ndarray, n_angles: int = 64) -> np.ndarray:
    thetas = np.linspace(0.0, 180.0, n_angles, endpoint=False)
    sinogram = radon(silhouette, theta=thetas)   # shape (n_offsets, n_angles)
    r = (sinogram.astype(np.float64) ** 2).sum(axis=0)
    return r / r.max()
```

- Validate against the original: run both on the silhouettes in `r.jpg` /
  recorded frames and compare curves (exact values will differ — the C++ uses
  nearest-neighbour interpolation and skips steep angles — but the shape and
  the downstream classification must match).
- Then delete `RTransform.{h,cpp}`, `rTransform.pyx`, `rTransform.so`,
  `setup.py`, `test/build/`. Keep the original C++ retrievable via git
  (`git show c55ca97:test/RTransform.cpp`).

## Phase 3 — Restructure: pipeline vs GUI

Target layout:

```
falldetect/
  features.py       # silhouette extraction, r_transform, multi-scale HOG
  keyframe.py       # keyframe selection (histogram distance)
  classifier.py     # training-data loading, k-NN fit/predict, label table
  pipeline.py       # FrameBuffer: per-frame step + every-20-frames classify
  gui/
    app.py          # PyQt widget: capture, timer, display only
    form.ui / ui_form.py
data/
  data.txt, walk.txt, run.txt, test.txt, background.png, ...
scripts/
  plot_rtransform.py   # replaces test/main.py
```

- Delete dead code: `ImageGrabber` (never called), `silhouetteDetection`
  (duplicate of `...Cropped` minus one resize), the unused `fgbg` background
  subtractor, the unused `skimage.feature.hog` import, `ui/uiform.py` vs
  `ui/ui_form.py` duplication (keep one).
- The pipeline classes must be constructible and runnable without Qt so they
  can be unit-tested and run on a video file headlessly (CLI entry point).

## Phase 4 — Resolve the feature-vector question

The app computes a combined 188-dim feature (20-dim LLE-reduced R-transform
sequence + 168-dim HOG) but classifies on HOG alone; `data.txt` is HOG-only
while `walk/run/test.txt` hold the combined vectors.

- Decide: HOG-only (matches current training data) or combined (matches the
  thesis experiments). If combined, rebuild the training matrix from
  `walk.txt`/`run.txt` (labels: rows are 113 walk / 67 run — verify counts)
  or re-extract from the source videos.
- Note (found while capturing goldens): `walk/run/test.txt` are on a
  *different feature scale* than `data.txt` — normalised [0,1] vs raw HOG
  magnitudes up to ~474k. The dumps were evidently produced by an older HOG
  variant with the (now commented-out) normalisation enabled, so they cannot
  be mixed with `data.txt` without re-extraction or rescaling.
- Replace hardcoded label counts (140/87/120) with a labelled data file
  (e.g. CSV with a label column, or one `.npz` with `X` and `y`).
- Wrap the classifier in a scikit-learn `Pipeline` with a scaler; evaluate on
  `test.txt` with a proper train/test split and report a confusion matrix, so
  refactoring regressions are measurable.

## Phase 5 — Tests and CI

Golden regression tests (Layers 1–2 of the test strategy) already exist:
`make test` builds the C++ harness (`tools/rtransform_dump.cpp`) and runs
`tests/test_golden.py` against the committed goldens in `tests/golden/`
(R-transform reference values from the original C++; k-NN predictions from
the app's training setup). `make goldens` regenerates them — only run that
deliberately. Still to add:

- Unit tests: silhouette extraction on `background.png`/synthetic images,
  `r_transform` invariance properties (translation/scale), HOG vector length,
  keyframe selection determinism.
- One end-to-end test: run the headless pipeline over a short clip (or a
  stored frame sequence) and assert the predicted label.
- GitHub Actions (or similar): lint (ruff) + tests on Python 3.11.

## Order of work

Phases 1→2 first (smallest diffs, immediately runnable), then 3 (structure),
then 4 (data/ML decisions), then 5. Phases 1–3 do not change behaviour except
for the named bug fixes; Phase 4 may change classification results and should
be measured with the Phase 5 evaluation in place.
