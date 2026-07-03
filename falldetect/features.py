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

import numpy as np


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
