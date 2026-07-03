# Fall Detection from Video

Master's thesis project: a PyQt5 desktop application that watches a video file or
webcam feed and classifies the behaviour of the person in it as **walk**, **run**
or **fall**, showing the label live in the GUI.

Built with Python 2.7, PyQt5, OpenCV 2.x, scikit-learn and a C++ extension
(wrapped with Cython) for the performance-critical shape descriptor.

## How it works

The main application is `test/imageshow.py`. Frames are processed on a 50 ms
timer in the following pipeline:

1. **Silhouette extraction** — each grayscale frame is thresholded with Otsu's
   method, holes are filled by flood-filling from the corner and inverting, the
   external contours are found, and the first sufficiently large bounding box
   (> 40×40 px) is cropped and resized to a 128×128 binary silhouette.

2. **R-transform (per frame)** — the silhouette is fed to the C++
   `RTransformer` (`test/RTransform.cpp`, exposed to Python via the Cython
   wrapper `test/rTransform.pyx`). It computes a discrete **Radon transform**
   of the silhouette over N = 64 angles: for each angle θ and offset ρ, the
   pixel values along the line (θ, ρ) are summed. Near-vertical angles are
   skipped for numerical stability and recovered by a second pass over the
   image rotated 90°. The **R-transform** then collapses the 2-D Radon result
   into a 1-D signature: for each angle, the sum of squared Radon values over
   all offsets, normalised by the maximum. This signature describes the *pose*
   of the silhouette and is invariant to translation and scaling — a standing
   walker and a lying fallen person produce very different curves.

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
| `test/RTransform.{h,cpp}` | C++ Radon/R-transform implementation |
| `test/rTransform.pyx`, `test/setup.py` | Cython wrapper and build script |
| `test/main.py` | Standalone script: R-transform of a test image, plotted |
| `test/imagegrabber.py` | Unused alternative keyframe helper (CIELAB) |
| `test/data.txt` | Training features (347×168 HOG vectors) |
| `test/walk.txt`, `run.txt`, `test.txt` | Feature dumps (188-dim, HOG + R-transform) |
| `ui/` | PyQt5 UI generated from `test/form.ui` |
| `test/build/` | Build artifacts (ignore) |

## Testing

```
make test      # builds the C++ harness + venv, runs the golden regression tests
make goldens   # regenerates tests/golden/ -- only run deliberately
```

The tests in `tests/test_golden.py` pin the behaviour of the original code
(R-transform values, k-NN predictions) so the refactoring can be verified
against it. See `REFACTORING.md` for the plan.

## Building the extension

```
cd test
python setup.py build_ext --inplace
```

**Warning:** on a case-insensitive filesystem Cython's generated `rTransform.cpp`
overwrites the hand-written `RTransform.cpp` — this happened once already
(commit `4ec0798`); the original was restored from commit `c55ca97`. Build in
`build/` or rename one of the files before regenerating.
