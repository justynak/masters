"""Feature extraction for the fall-detection pipeline.

r_transform() is a faithful NumPy port of the thesis C++ implementation
(test/RTransform.cpp, retrievable via `git show c55ca97:test/RTransform.cpp`),
replacing the Cython extension. It intentionally reproduces the original
algorithm bug-for-bug so its output matches the frozen golden files in
tests/golden/rtransform/:

* lines are sampled with nearest-neighbour stepping along one axis only and
  are NOT normalised by line length, so amplitude varies with angle (a proper
  Radon transform would give a flat R-transform for a circle; this one does
  not) -- do not "fix" this without regenerating goldens and retraining;
* pi is approximated as 3.14, as in the original;
* near-vertical angles are skipped and reconstructed from a 90-degree-rotated
  second pass (the sum2 branch of the original);
* the final normalisation happens in float32, matching the C++
  `(float) rT[i] / maxVal`.
"""

import cv2
import numpy as np


def silhouette_cropped(gray_frame):
    """Extract a 128x128 binary silhouette crop from a grayscale frame, or
    None. Port of ImageWidget.silhouetteDetectionCropped: Otsu threshold,
    flood-fill hole filling, first sufficiently large external contour."""
    _, frame = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # fill holes: flood from the corner and OR the inverse back in
    floodfill = frame.copy()
    h, w = frame.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(floodfill, mask, (0, 0), 255)
    frame = frame | cv2.bitwise_not(floodfill)

    cnts, _ = cv2.findContours(frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        if cv2.contourArea(c) < 250:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        if cw > 40 and ch > 40:
            crop = frame[y : y + ch, x : x + cw].copy()
            return cv2.resize(crop, (128, 128))

    return None


def hog_multiscale(img, bin_n=8):
    """168-dim multi-scale HOG (8 orientation bins over 1+4+16 partitions).
    Port of the HOG() function in imageshow.py."""
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n * ang / (2 * np.pi))

    hists = np.zeros(168)
    i = 0
    for parts in (1, 4, 16):
        bin_cells = np.split(bins, parts)
        mag_cells = np.split(mag, parts)
        hist = np.array(
            [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        )
        block = parts * bin_n
        hists[i : i + block] = np.hstack(hist)
        i += block

    return hists


def _round_half(v):
    """C++ ROUNDING_FACTOR semantics: round half away from zero, via floor."""
    return np.where(v >= 0, np.floor(v + 0.5), np.floor(v - 0.5)).astype(np.int64)


def r_transform(silhouette, n_angles=64):
    """R-transform of a 2-D uint8 silhouette; returns n_angles float values
    normalised to [0, 1]. Axis convention follows the original call sites:
    the array's first axis is passed to the maths as the 'cols'/width axis."""
    im = np.asarray(silhouette, dtype=np.int64)
    width, height = im.shape  # shape[0] was passed as "cols" to the C++
    N = n_angles

    xofftemp = width / 2.0 - 1
    yofftemp = height / 2.0 - 1
    xoffset = int(_round_half(np.float64(xofftemp)))
    yoffset = int(_round_half(np.float64(yofftemp)))
    D = int(_round_half(np.sqrt(np.float64(xoffset**2 + yoffset**2))))

    M = max(width, height)
    m = np.arange(M)
    d = np.arange(D)

    im_radon = np.zeros((N, D), dtype=np.uint64)

    # the original loop considers k in (N/8, 3N/8] and (5N/8, 7N/8] and fills
    # the remaining angles via the 90-degree-rotated sum2 pass
    ks = list(range(N // 8 + 1, 3 * N // 8 + 1)) + list(range(5 * N // 8 + 1, 7 * N // 8 + 1))

    for k in ks:
        theta = 2 * k * 3.14 / N
        alpha = np.tan(theta + 3.14 / 2)
        beta = _round_half(-alpha * d * np.cos(theta) + d * np.sin(theta))

        # n[d_i, m_i] = round(alpha*(m - offset) + beta[d_i])
        n1 = _round_half(alpha * (m - xoffset)[None, :] + beta[:, None])
        n2 = _round_half(alpha * (m - yoffset)[None, :] + beta[:, None])

        ny = n1 + yoffset
        valid1 = (m[None, :] < width) & (ny >= 0) & (ny < height)
        vals1 = im[np.minimum(m, width - 1)[None, :], np.clip(ny, 0, height - 1)]
        sum1 = np.where(valid1, vals1, 0).sum(axis=1)

        nx = n2 + xoffset
        valid2 = (m[None, :] < height) & (nx >= 0) & (nx < width)
        vals2 = im[np.clip(width - 1 - nx, 0, width - 1), np.minimum(m, height - 1)[None, :]]
        sum2 = np.where(valid2, vals2, 0).sum(axis=1)

        # the C++ routes the sums through a float cast; exact below 2^24
        im_radon[k] = sum1.astype(np.uint64)
        im_radon[(k + N // 4) % N] = sum2.astype(np.uint64)

    rT = (im_radon.astype(np.uint64) ** 2).sum(axis=1, dtype=np.uint64)
    max_val = rT.max()

    # (float) rT[i] / maxVal -- float32 arithmetic, stored as double
    return (rT.astype(np.float32) / np.float32(max_val)).astype(np.float64)
