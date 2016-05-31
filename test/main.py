import rTransform
import sys
import cv2
import cv
import numpy as np

import matplotlib.pyplot as plt
    
rT = rTransform.PyRTransform();
image = cv2.imread('r.jpg', 1)
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
cols, rows = image.shape
 
data = rT.rTransform(image, cols, rows, 64)

plt.plot(data)
plt.show()

print data
