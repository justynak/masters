#!/usr/bin/env python3
# Standalone sanity script: R-transform of a test image, plotted.

import cv2
import numpy as np

import rTransform

rT = rTransform.PyRTransform()
image = cv2.imread('r.jpg', 1)
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

cols, rows = image.shape

data = rT.rTransform(image, cols, rows, 64)

print(data)

try:
    import matplotlib.pyplot as plt
except ImportError:
    print('matplotlib not installed, skipping plot')
else:
    plt.plot(data)
    plt.show()
