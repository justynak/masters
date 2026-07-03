#!/usr/bin/env python3
# Standalone sanity script: R-transform of a test image, plotted.

import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from falldetect.features import r_transform

image = cv2.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r.jpg'), 1)
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

data = r_transform(image, 64)

print(data)

try:
    import matplotlib.pyplot as plt
except ImportError:
    print('matplotlib not installed, skipping plot')
else:
    plt.plot(data)
    plt.show()
