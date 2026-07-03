# Improvement Ideas (post-Phase 4)

Status quo (2026-07-03): `rt_stats` feature (mean+std of the window's 20
per-frame R-transforms), StandardScaler + kNN(5), trained on Le2i.
Held-out videos: **walk recall 98.2%, fall recall 55.6%, macro 76.9%**
(`scripts/evaluate.py --videos data/le2i_test_videos.txt`). Fall recall is
capped mainly by data: 105 fall windows from 84 fall events.

Ordered by expected payoff per effort. Every item is measurable with the
existing harness: rerun `scripts/extract_features.py` +
`scripts/train_le2i.py` (offline comparison) and
`scripts/evaluate.py --videos` (end-to-end on held-out videos).

## 1. Stop destroying the bounding-box aspect ratio  (~afternoon, likely big)

`falldetect.features.silhouette_cropped` resizes every crop to 128x128,
discarding the aspect ratio — the single most discriminative fall cue
(standing = tall box, fallen = wide box). The R-transform only partially
recovers orientation. Add the crop's pre-resize width/height (or h/w ratio
mean+std+min over the window) as extra features next to `rt_stats`.
~15 lines: return the ratio from `silhouette_cropped` (or alongside it),
thread it through `Pipeline`/`extract_features.py`, add a variant in
`train_le2i.py`.

## 2. Sliding windows instead of disjoint ones  (~afternoon, likely big)

Falls last ~22 frames (median) and windows are disjoint 20-frame blocks, so
many falls straddle a boundary and dilute into two mixed windows. A stride
of ~5 frames multiplies fall training windows ~4x (from 105) and guarantees
some window sits squarely on each fall. Helps training data, evaluation
granularity, and detection latency. Change the buffer handling in
`falldetect.pipeline.Pipeline.push` (drop `stride` oldest frames instead of
clearing) and regenerate features.

## 3. More fall data  (~evening of labelling, addresses the main cap)

- Coffee_room_02 (22 videos), Office (33) and Lecture_room (27) have videos
  but no annotation files in the current download. Either find the full
  official annotation distribution, or hand-label: only the fall start/end
  frame numbers are needed per video (two numbers; bboxes are only used for
  walk-purity filtering, and `scripts/le2i.py` can fall back to
  fall-window-only labelling).
- UR Fall Detection dataset is a second downloadable source (RGB stream).
- Class imbalance (78 fall vs 587 walk training windows) is the most likely
  reason fall recall sticks at ~55%.

## 4. Better per-frame silhouettes  (~day, measurable in isolation)

Otsu thresholding on grayscale assumes a bright-person-on-dark-scene
bimodal histogram. `cv2.createBackgroundSubtractorMOG2` models the actual
background per pixel; the temporal-median background
(`falldetect.pipeline.median_background`) makes a clean initialisation.
Measure silhouette quality directly against the Le2i per-frame bounding
boxes (IoU / centre distance) before and after — the same study that
explains the original method's failure (see below).

## 5. Classifier polish  (~hours, small but free)

- Tune k by cross-validation on the training split; try
  `weights="distance"`.
- For an alarm system, use `predict_proba` with a tuned threshold instead
  of argmax, trading false alarms against missed falls explicitly.
- Report precision / false-alarms-per-hour alongside recall in
  `scripts/evaluate.py` — that is the metric users of a fall alarm feel.

## 6. Cross-scene evaluation  (~hours, rigour)

The current split is by video, but every scene appears on both sides.
Train on Coffee_room + Home_01, test on Home_02 (and rotations) to measure
whether `rt_stats` generalises across rooms — exactly the test the original
MuHAVi-trained method failed. GroupShuffleSplit on `scene` instead of
`video` in `scripts/train_le2i.py`.

## 7. True Radon transform  (~day, publishable either way)

Swap the bug-for-bug thesis R-transform (line sums not normalised by line
length -> angle-dependent amplitude; see `falldetect/features.py` docstring)
for a length-normalised `skimage.transform.radon` variant, as a new feature
variant in `train_le2i.py`. The artifact may be hurting or accidentally
helping; either answer is a result. Keep the frozen goldens for the legacy
implementation.

---

# Companion experiments: proving the original method was MuHAVi-bound

For the thesis-defence-style argument, three pieces of evidence:

1. **In-domain ceiling.** kNN(12) on MuHAVi HOG (`test/data.txt`):
   training-set self-prediction 78.1% (`tests/golden/knn/data_pred.txt`);
   honest 5-fold CV **70.6% +/- 3.8%** (no video grouping possible — rows
   carry no video ids — so still slightly optimistic).
2. **Out-of-domain failure, strongest form.** Already measured: the
   MuHAVi-trained model scores 0% fall recall on Le2i, and *retrained on
   Le2i* the HOG feature still scores 0% (a balanced random-forest probe
   caps at ~7%) — the feature, not the classifier, is the bottleneck. See
   REFACTORING.md Phase 4 table.
3. **Mechanism.** Compare extracted silhouettes against Le2i's annotated
   per-frame person boxes (IoU/centre distance): the hue-difference
   keyframe silhouette (HOG's input) should rarely overlap the person,
   while the per-frame Otsu silhouette (rt_stats' input) mostly should.
   Add 3-4 side-by-side example figures. This turns "it broke" into "the
   HOG feature encodes the chroma-keyed studio, not the person".
