#!/usr/bin/env python

import sys
import time
import cv2
import cv

from PyQt5.QtCore import Qt
from PyQt5 import QtCore
from PyQt5.QtGui import *
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QGridLayout, QLabel, QLineEdit
from PyQt5.QtWidgets import QTextEdit, QWidget, QDialog, QApplication, QPushButton, QFileDialog

from ui import uiform

import numpy as np
import scipy
import copy
from imagegrabber import ImageGrabber
from _mysql import NULL

from matplotlib import pyplot as plt


class ImageWidget(QDialog):
    def __init__(self, parent=None):
        super(ImageWidget, self).__init__(parent)
        
        self.cap = cv2.VideoCapture()
        
        self.ui = uiform.Ui_Form()
        self.ui.setupUi(self)
                
        self.timer = QtCore.QTimer()      
        self.timer.timeout.connect(self.onTimerTimeout)
        
        self.ui.buttonOpen.clicked.connect(self.onButtonOpenCliked)
        self.ui.buttonPlay.clicked.connect(self.onButtonPlayClicked)
        self.ui.buttonUseCam.clicked.connect(self.onButtonUseCamClicked)
        self.background = cv2.imread('/home/justyna/workspace/qttest/test/background.png', 1)
        if self.background is None :
            print 'Dupa blada'
        
        self.fgbg = cv2.BackgroundSubtractorMOG()
        self.grabber = ImageGrabber()
        self.frames = []
    
    def reinitCam(self, arg):
        if self.cap.isOpened():
            self.cap.release()
            
        self.cap.open(arg)
    
    def setTimer(self, interval):
        self.timer.setInterval(interval)
        self.timer.start()
        
    def onButtonOpenCliked(self):
        #open the file
        filename, _ = QFileDialog.getOpenFileName()
        self.ui.lineFilename.setText(filename)
        
        self.reinitCam(filename)       
       
        
    def onButtonPlayClicked(self):    
        text = self.ui.buttonPlay.text()
        print text
        
        if text == 'Play':
            self.setTimer(50)
            self.ui.buttonPlay.setText('Stop')
        else:
            self.timer.stop()
            self.ui.buttonPlay.setText('Play')
        
    def onButtonUseCamClicked(self):
        self.reinitCam(0)
        
        
        if self.cap != NULL:
            print 'Success!'
        
    def onTimerTimeout(self):       
        retVal, frame = self.cap.read()
        
        if retVal != True :
            self.timer.stop()
            self.cap.release()
            self.ui.buttonPlay.setText('Play')
            return
        
        frame = cv2.resize(frame, (500, 300))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        
        self.frames.append(frame)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        cols, rows = frame.shape
        data = frame.data
              
        #for rgb: Format_RGB888
        image = QImage(data, rows, cols, QImage.Format_Indexed8)
        pixmap = QPixmap.fromImage(image)
        self.ui.labelImage.setPixmap(pixmap)
        
        if len(self.frames) >= 20 :
            keyframe = self.getKeyframe(self.frames)
            keyframe = cv2.cvtColor(keyframe, cv2.COLOR_RGB2HSV)
            self.background = cv2.cvtColor(keyframe, cv2.COLOR_RGB2HSV)
            
            #split channels
            hb, sb, vb = cv2.split(self.background)
            h, s, v = cv2.split(keyframe)
            
            #get the difference in hue
            h = h - hb            
            keyframe = h

            #get the silhouette and silhouette edges from keyframe
            keyframe = cv2.resize(keyframe, (500, 300))
            keyframe = cv2.GaussianBlur(keyframe, (3, 3), 0)                     
            
            silhouette = self.silhouetteDetection(keyframe)    
            silhouetteContours = cv2.Canny(silhouette, 50, 5)
            
            gradX = cv2.Sobel(silhouetteContours, cv2.CV_64F, 1, 0, ksize=5)
            gradY = cv2.Sobel(silhouetteContours, cv2.CV_64F, 0, 1, ksize=5)
            
            orientation = np.arctan2(gradY, gradX)
            
            sdeg = SDEG(orientation)
            Rt, thetas, rhos = HoughTransform(silhouette)
            
            print sdeg.shape, sdeg
            
            #rGothic = np.trapz(rTransform[], rTransform[])
            
            if not silhouette is None :            
                fig, ax = plt.subplots(1, 2, figsize=(10, 10))

                ax[0].imshow(silhouette, cmap=plt.cm.gray)
                ax[0].set_title('Input image')
                ax[0].axis('image')

                ax[1].imshow(Rt, cmap='jet', extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]])
                ax[1].set_aspect('equal', adjustable='box')
                ax[1].set_title('Hough transform')
                ax[1].set_xlabel('Angles (degrees)')
                ax[1].set_ylabel('Distance (pixels)')
                ax[1].axis('image')

                plt.show()
                
                time.sleep(.300)
                plt.close('all')
                        
            #clean the frames array
            self.frames[:] = []

    def processImage(self, frame):
        newFrame, roi = self.backgroundDetection(frame)
        return newFrame, roi
        
    def backgroundDetection(self, frame):
        blurred = cv2.GaussianBlur(frame, (15,15), 0) 
        processedFrame = blurred
        
        return processedFrame #, roi
    
    def silhouetteDetection(self, frame):
        #niezleframe = cv2.Canny(frame, 50, 50)
        ret, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Copy the thresholded image.
        floodfill = frame.copy()
 
        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = frame.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
 
        # Floodfill from point (0, 0)
        cv2.floodFill(floodfill, mask, (0,0), 255);
 
        # Invert floodfilled image
        floodfillInv = cv2.bitwise_not(floodfill)
 
        frame = frame | floodfillInv
        
        (cnts, _) = cv2.findContours(frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rois = []
        
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < 150:
                continue
            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            
            #print x, y, w, h
            #print x, y
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
            
            if w > 30 and h > 30 :
            #save roi somewhere
                y2 = y + h
                x2 = x + w
                crop = frame[y : y2 , x : x2].copy()
                crop = cv2.resize(crop, (48, 48))
                                
                return crop

        return None
            
    def getFrameHistogram(self, frame):
        hist0 =  cv2.calcHist([frame],[0], None,[256], [0,256])
        hist1 =  cv2.calcHist([frame],[1], None,[256], [0,256])
        hist2 =  cv2.calcHist([frame],[2], None,[256], [0,256])
        
        return np.array([hist0, hist1, hist2])
        
    def calculateHistogramDistance(self, frame0, frame1):
        #calculate histogram distance between frames
        hist0 = self.getFrameHistogram(frame0)
        hist1 = self.getFrameHistogram(frame1)

        m, n, _ = frame0.shape
                   
        histElement = np.zeros( (256, 3), dtype=np.int64 )
        
        for i in range(0, 256):
            histElement[i, 0] = min(hist0[0, i], hist1[0, i])
            histElement[i, 1] = min(hist0[1, i], hist1[1, i])
            histElement[i, 2] = min(hist0[2, i], hist1[2, i])
            
        return 1 - 1 / float (3 * m * n) * sum(sum(histElement))
    
    def getKeyframe(self, frames):
        frame0 = cv2.cvtColor(self.frames[0], cv2.COLOR_RGB2LAB)
        frame1 = cv2.cvtColor(self.frames[4], cv2.COLOR_RGB2LAB)
        frame2 = cv2.cvtColor(self.frames[9], cv2.COLOR_RGB2LAB)
        frame3 = cv2.cvtColor(self.frames[14], cv2.COLOR_RGB2LAB)
        #newFrame, roi = self.processImage(frame)           
          
        hd0 = self.calculateHistogramDistance(frame0, frame1)
        hd1 = self.calculateHistogramDistance(frame1, frame2)
        hd2 = self.calculateHistogramDistance(frame2, frame3)
            
        hd = np.argmin([hd0, hd1, hd2])
            
        keyframe = frame0
            
        if hd == 1 :
            keyframe = frame1
        elif hd == 2:
            keyframe = frame2
             
        keyframe = cv2.cvtColor(keyframe, cv2.COLOR_LAB2RGB)
        
        return keyframe
    
def HoughTransform(img):
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    width, height = img.shape
    diag_len = np.ceil(np.sqrt(width * width + height * height))   # max_dist
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0)
    
    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)
    
    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(img)  # (row, col) indexes to edges
    
    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
    
        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len
            accumulator[rho, t_idx] += 1
    
    
    
    return accumulator, thetas, rhos

def SDEG(frame):         
    #first stage
    windowsize0 = 24
    windowsize1 = 12
    
    hist, _ = np.histogram(frame, bins = 8)
    sdeg = np.array([])
    sdeg = np.append(sdeg, hist)
    
    #second stage
    for r in range(0, frame.shape[0] - windowsize0 + 1, windowsize0):
        for c in range(0, frame.shape[1] - windowsize0 + 1, windowsize0):
            window = frame[r : r + windowsize0, c : c + windowsize0]
            hist, _ = np.histogram(window, bins = 8)
            sdeg = np.append(sdeg, hist)   

    #third stage
    for r in range(0, frame.shape[0] - windowsize1 + 1, windowsize1):
        for c in range(0, frame.shape[1] - windowsize1 + 1, windowsize1):
            window = frame[r : r + windowsize1, c : c + windowsize1]
            hist, _ = np.histogram(window, bins = 8)
            sdeg = np.append(sdeg, hist)
            
    return sdeg

def Quantize(signal, partitions, codebook):
    indices = []
    quanta = []
    for datum in signal:
        index = 0
        while index < len(partitions) and datum > partitions[index]:
            index += 1
        indices.append(index)
        quanta.append(codebook[index])
    return indices, quanta
   
                
if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = ImageWidget()
    form.show()

    sys.exit(app.exec_())


