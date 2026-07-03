# Fall Detection from Video

Master's thesis project: a PyQt5 desktop application that watches a video file or
webcam feed and classifies the behaviour of the person in it as **walk**, **run**
or **fall**, showing the label live in the GUI.

Originally built with Python 2.7, PyQt5, OpenCV 2.x, scikit-learn and a C++
extension (wrapped with Cython) for the performance-critical shape
descriptor; since ported to Python 3 and pure NumPy (see `REFACTORING.md`,
Phases 1–2).

## Running

```
make run       # sets up .venv, builds the Cython extension, starts the GUI
```

## How it works

The main application is `test/imageshow.py`. Frames are processed on a 50 ms
timer in the following pipeline:

1. **Silhouette extraction** — each grayscale frame is thresholded with Otsu's
   method, holes are filled by flood-filling from the corner and inverting, the
   external contours are found, and the first sufficiently large bounding box
   (> 40×40 px) is cropped and resized to a 128×128 binary silhouette.

2. **R-transform (per frame)** — the silhouette is fed to
   `falldetect.features.r_transform` (a faithful NumPy port of the original
   C++ implementation). It computes a discrete **Radon transform** of the
   silhouette over N = 64 angles: for each angle θ and offset ρ, the pixel
   values along the line (θ, ρ) are summed. Near-vertical angles are skipped
   for numerical stability and recovered by a second pass over the image
   rotated 90°. The **R-transform** then collapses the 2-D Radon result into
   a 1-D signature: for each angle, the sum of squared Radon values over all
   offsets, normalised by the maximum. This signature describes the *pose*
   of the silhouette and is invariant to translation and scaling — a standing
   walker and a lying fallen person produce very different curves.
   (The line sampling is deliberately kept bug-for-bug compatible with the
   thesis C++ — see the docstring in `falldetect/features.py`.)

3. **Every 20 buffered frames**, classification runs:
   - A **keyframe** is picked from frames 0/4/9/14 by minimal Canberra
     histogram distance in HSV space (i.e. the most stable-looking frame).
   - The keyframe silhouette's Canny edges are described with a custom
     multi-scale **HOG**: 8 orientation bins over 1 + 4 + 16 image partitions,
     giving a 168-dimensional vector.
   - The 20 per-frame R-transform signatures are reduced with **Locally
     Linear Embedding** (1 component) into a 20-dimensional descriptor of how
     the pose *changed over time*.
   - A **k-nearest-neighbours classifier** (k = 12), trained at startup on
     `test/data.txt` (347 samples: 140 walk, 87 run, 120 fall), predicts the
     behaviour label.

   Note: the combined HOG + R-transform feature (188-dim, as stored in
   `walk.txt`/`run.txt`/`test.txt`) is computed but the classifier currently
   uses only the 168-dim HOG vector, matching `data.txt`.

## Repository layout

| Path | Purpose |
|---|---|
| `test/imageshow.py` | Main GUI application and processing pipeline |
| `falldetect/features.py` | R-transform (NumPy port of the original C++) |
| `test/main.py` | Standalone script: R-transform of a test image, plotted |
| `test/imagegrabber.py` | Unused alternative keyframe helper (CIELAB) |
| `test/data.txt` | Training features (347×168 HOG vectors) |
| `test/walk.txt`, `run.txt`, `test.txt` | Feature dumps (188-dim, HOG + R-transform) |
| `ui/` | PyQt5 UI generated from `test/form.ui` |
| `tests/` | Golden regression tests (`make test`) |

## Testing

```
make test      # builds the C++ harness + venv, runs the golden regression tests
make goldens   # regenerates tests/golden/ -- only run deliberately
```

The tests in `tests/test_golden.py` pin the behaviour of the original code
(R-transform values, k-NN predictions) so the refactoring can be verified
against it. See `REFACTORING.md` for the plan.

## History note

The R-transform was originally C++ (`RTransform.cpp`) wrapped with Cython,
removed in Phase 2 in favour of the NumPy port (bitwise-identical output,
verified against goldens). The original sources are retrievable with
`git show c55ca97:test/RTransform.cpp`.
