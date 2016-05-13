#!/usr/bin/env python

import sys
import cv2
import cv

from PyQt5 import QtCore
from PyQt5.QtCore import QObject, pyqtSignal

import numpy as np
import copy


def cv_size(img):
    return tuple(img.shape[1::-1])


class ImageGrabber(QObject):
    def __init__(self, parent=None):
        super(ImageGrabber, self).__init__(parent)
        
        #init frames as empty list 
        self.frames = []
        
    def grabFrame(self, frame):
        cielabFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        self.frames.append(cielabFrame)
    
    def getKeyFrame(self):
        m,n = cv_size(self.frames[0])


        frameL0, frameA0, frameB0 = cv2.split(self.frames[0])
        frameL1, frameA1, frameB1 = cv2.split(self.frames[1])
        
        minL = np.minimum(frameL0, frameL1)
        minA = np.minimum(frameA0, frameA1)
        minB = np.minimum(frameB0, frameB1)
        
        sumL = sum(sum(minL))
        sumA = sum(sum(minA))
        sumB = sum(sum(minB))
        
        h = 1 - 1/float(3 * m * n) * (sumL + sumA + sumB)
        
        self.histogramDistanceL = (m, n, sumL, sumA, sumB, h);
        
    def getHistogramDistance(self):
        return self.histogramDistanceL
    
            
    