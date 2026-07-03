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

## Phase 2 — Replace the C++/Cython extension with NumPy ✅ DONE

The C++ existed only because a triple Python loop was too slow in Py2.

**Plan change:** the original idea was `skimage.transform.radon`, but
validation against the goldens showed it cannot reproduce the C++ output
even up to shift/scale (best correlation 0.31–0.89 across test inputs).
The thesis algorithm is not a textbook Radon transform: it samples lines
along one axis without normalising by line length, so amplitude varies
with angle (a circle produces a non-flat R-transform). Whether that
distortion carries signal the thesis relied on is unknowable without
retraining, so the safe move was a **faithful vectorised NumPy port** of
the C++ (`falldetect/features.py`), bug-for-bug including the `3.14`
approximation of pi and the float32 normalisation. Result: bitwise-identical
to the goldens on all reference inputs, ~9 ms/frame (fits the 50 ms timer).

Deleted: `RTransform.{h,cpp}`, `rTransform.pyx`, `test/setup.py`,
`tools/rtransform_dump.cpp` (harness), `test/build/`. All retrievable from
git history: `git show c55ca97:test/RTransform.cpp`,
`git show a27bea8:tools/rtransform_dump.cpp`. The R-transform goldens are
frozen. Swapping in a *correct* Radon transform remains an option for
Phase 4, where classification quality is measured.

## Phase 3 — Restructure: pipeline vs GUI ✅ DONE (2026-07-03)

Done: `falldetect/` now holds the pipeline stages (`features`, `keyframe`,
`classifier`) and a headless windowed `pipeline.Pipeline` with a CLI
(`python -m falldetect.pipeline video.avi`) and a temporal-median background
fallback. The GUI (`test/imageshow.py`) delegates to the same stage
functions. Dead code deleted: `ImageGrabber`, `silhouetteDetection`,
`getAlternativeKeyframe`/`backgroundDetection`/`processImage`, the unused
MOG subtractor, the discarded LLE computation, `ui/ui_form.py`.

Evaluation harness added on top (`scripts/`): `prepare_le2i.py` re-encodes
the Le2i fall dataset (its rawvideo AVIs crash opencv-python 5.0),
`le2i.py` derives labelled walk/fall segments from the shipped annotations
(84 fall + 102 walk segments from 107 videos), `evaluate.py` prints a
confusion matrix.

**Baseline (2026-07-03), MuHAVi-trained classifier on Le2i:** 913 windows,
fall recall 0% (fall is never predicted), walk recall 51.6%, most windows
labelled "run". The domain gap is total — this is the number Phase 4
retraining has to beat.

Remaining from the original Phase 3 scope: moving the GUI out of `test/`
into `falldetect/gui/` (cosmetic, deferred).

Original plan:

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

## Phase 4 — Resolve the feature-vector question ✅ DONE (2026-07-03)

**Decision: neither HOG-only nor the combined feature — the winning feature
is `rt_stats`**: mean+std over the window's 20 per-frame R-transforms
(128-dim), StandardScaler + kNN(5). Measured on a video-level held-out split
of the Le2i windows (`scripts/train_le2i.py`, seed 0; 74 train / 33 test
videos, 913 windows total):

| variant                | walk recall | fall recall | macro |
|------------------------|------------|-------------|-------|
| hog (thesis setup)     | 100%       | 0%          | 50.0% |
| hog+scale              | 100%       | 0%          | 50.0% |
| combined+scale (thesis)| 99.5%      | 0%          | 49.8% |
| **rt_stats (chosen)**  | **98.2%**  | **55.6%**   | **76.9%** |

The keyframe HOG carries almost no fall signal on Le2i (its hue-difference
silhouette is unreliable in realistic rooms; even a balanced random forest
reaches only 7% fall recall on it). The R-transform — which the original app
computed and then discarded — is computed from the per-frame Otsu silhouette
and separates falls well. End-to-end pipeline evaluation on the held-out
videos reproduces the offline numbers exactly (fall 15/27, walk 217/221).

Artifacts: `data/le2i_windows.npz` (all extracted window features),
`data/le2i_train.npz` (persisted training set; loaded by
`falldetect.classifier.train_classifier`), `data/le2i_test_videos.txt`
(held-out videos; pass to `scripts/evaluate.py --videos`). The legacy
MuHAVi classifier remains available as `classifier.legacy_muhavi()`.
Recall floors are pinned in `tests/test_le2i_model.py`.

Remaining ideas for later: more fall training data (fall recall is limited
by 105 fall windows), a proper (non-thesis) Radon transform, revisiting the
20-frame window vs. Le2i's ~22-frame median fall duration.

Original plan:

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
