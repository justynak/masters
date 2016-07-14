#!/usr/bin/env python

from __future__ import division

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
from PyQt5.QtWidgets import QTextEdit, QWidget, QDialog, QApplication, QPushButton, QFileDialog, QCheckBox

from ui import uiform

import numpy as np
import scipy
import copy
from imagegrabber import ImageGrabber
from sklearn import manifold, datasets
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import hog
from matplotlib import pyplot as plt
import rTransform
from _mysql import result
import sklearn

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
        
        self.fgbg = cv2.BackgroundSubtractorMOG()
        self.grabber = ImageGrabber()
        
        self.frames = []
        self.rT = rTransform.PyRTransform()
        self.rTransforms = []
        self.lle = manifold.LocallyLinearEmbedding(n_neighbors=7, n_components=1, 
                                                    eigen_solver='auto', method='standard')     
        
        self.classifier = KNeighborsClassifier(12)
        self.behaviourLabelTable = ['walk', 'run', 'fall']
        self.ui.labelBehaviour.setText('Behaviour')
        
        # walk - 140, run -87, fall- 120
        data = np.loadtxt('/home/justyna/workspace/qttest/test/data.txt')
        train = data.astype(np.float32)
        print train.shape
        
        #print data.shape
        trainLabels = np.zeros(140, np.float32)
        trainLabels = np.append(trainLabels, np.ones(87, np.float32))
        label = np.zeros(120, np.float32)
        label.fill(2)
        trainLabels = np.append(trainLabels, label)
        #label = np.zeros(26, np.float32)
        #label.fill(3)
        #trainLabels = np.append(trainLabels, label)
        
        #print trainLabels
                
        self.classifier.fit(train, trainLabels)
    
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
        
    def onButtonPlayClicked(self):    
        text = self.ui.buttonPlay.text()
        print text
        
        if self.ui.checkboxCam.isChecked() :
            self.reinitCam(0)
        else :
            filename = self.ui.lineFilename.text()
            self.reinitCam(filename)
        
        backgroundFilename = self.ui.lineFilenameBackground.text()      
        self.background = cv2.imread(backgroundFilename, 1)
        self.background = cv2.resize(self.background, (500, 300))
        
        if self.background is None:
            print 'oops'
        
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
            print 'frame not captured'
            return
        
        frame = cv2.resize(frame, (500, 300))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)    

        grayFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        cols, rows = grayFrame.shape
        data = grayFrame.data
              
        #for rgb: Format_RGB888
        image = QImage(data, rows, cols, QImage.Format_Indexed8)
        pixmap = QPixmap.fromImage(image)
        self.ui.labelImage.setPixmap(pixmap)
        
        grayFrame = cv2.GaussianBlur(grayFrame, (3, 3), 0)                     
            
        sil = self.silhouetteDetectionCropped(grayFrame) 
        
        if sil is not None:
            x, y = sil.shape  
            self.rTransforms.append(self.rT.rTransform(sil, x, y, 64))
            self.frames.append(frame)
        
        if len(self.frames) >= 20 :
            t1 = time.time()

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
            keyframe = cv2.GaussianBlur(keyframe, (3, 3), 0)                     
            silhouette = self.silhouetteDetectionCropped(keyframe)    
                                                
            if silhouette is not None :                          
                silhouetteContours = cv2.Canny(silhouette, 50, 5)              
                sdeg = HOG(silhouetteContours, 8)
                x, y = silhouette.shape
        
                Rt = self.lle.fit_transform(self.rTransforms)
                Rt = Rt.T
                Rt = np.append(Rt, sdeg).astype(np.float32)
                                             
                arr = np.array(sdeg, np.float32)
                result = int(round(self.classifier.predict(arr)))
                
                self.ui.labelBehaviour.setText(self.behaviourLabelTable[result])
                
                #print self.behaviourLabelTable[result], time.time() - t1
                
                #ig, ax = plt.subplots(1, 2, figsize=(10, 10))
                #ax[0].imshow(silhouette, cmap=plt.cm.gray)
                #ax[0].set_title('Input image')
                #ax[0].axis('image')
                #thetas = np.linspace(0, len(sdeg), len(sdeg))
                #ax[1].plot(thetas, sdeg)
                #ax[1].set_title('SDEG transform')
                #ax[1].set_xlabel('Angles (degrees)')
                #ax[1].set_ylabel('Distance (pixels)')
                #plt.show()
                        
            #clean the frames array
            self.frames[:] = []
            self.rTransforms[:] = []

    def processImage(self, frame):
        newFrame, roi = self.backgroundDetection(frame)
        return newFrame, roi
        
    def backgroundDetection(self, frame):
        keyframe = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        self.background = cv2.cvtColor(self.background, cv2.COLOR_RGB2HSV)
        
        #split channels
        hb, sb, vb = cv2.split(self.background)
        h, s, v = cv2.split(keyframe)
        
        #get the difference in hue
        h = h - hb            
       
        return h
    
    def silhouetteDetection(self, frame):
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
            if cv2.contourArea(c) < 250:
                continue
            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            
            #print x, y, w, h
            #print x, y
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
            
            if w > 40 and h > 40:
            #save roi somewhere
                y2 = y + h
                x2 = x + w
                crop = frame[y : y2 , x : x2].copy()
                
                #check if frame is not empty (not entirely black)
                #if 100 * 100 - cv2.countNonZero(crop) < 200:
                #    print 'Empty frame'
                #    return None
                                
                return crop

        return None
    
    def silhouetteDetectionCropped(self, frame):
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
            if cv2.contourArea(c) < 250:
                continue
            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            
            #print x, y, w, h
            #print x, y
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
            
            if w > 40 and h > 40:
            #save roi somewhere
                y2 = y + h
                x2 = x + w
                crop = frame[y : y2 , x : x2].copy()
                crop = cv2.resize(crop, (128, 128))
                
                #check if frame is not empty (not entirely black)
                #if 100 * 100 - cv2.countNonZero(crop) < 200:
                #    print 'Empty frame'
                #    return None
                                
                return crop

        return None
            
    def getFrameHistogram(self, frame):
        hist =  cv2.calcHist([frame],[0], None, [8], [0,256])
        #hist1 =  cv2.calcHist([frame],[1], None,[256], [0,256])
        #hist2 =  cv2.calcHist([frame],[2], None,[256], [0,256])
        #return np.array([hist0, hist1, hist2])
        return cv2.normalize(hist).flatten()
        
    def calculateHistogramDistance(self, frame0, frame1):
        #calculate histogram distance between frames
        hist0 = self.getFrameHistogram(frame0)
        hist1 = self.getFrameHistogram(frame1)
        
        return scipy.spatial.distance.canberra(hist0, hist1)

        #m, n, _ = frame0.shape          
        #histElement = np.zeros( (256, 3), dtype=np.int64 )  
        #for i in range(0, 256):
        #    histElement[i, 0] = min(hist0[0, i], hist1[0, i])
        #    histElement[i, 1] = min(hist0[1, i], hist1[1, i])
        #    histElement[i, 2] = min(hist0[2, i], hist1[2, i]) 
        #return 1 - 1 / float (3 * m * n) * sum(sum(histElement))
    
    def getKeyframe(self, frames):
        frame0 = cv2.cvtColor(self.frames[0], cv2.COLOR_RGB2HSV)
        frame1 = cv2.cvtColor(self.frames[4], cv2.COLOR_RGB2HSV)
        frame2 = cv2.cvtColor(self.frames[9], cv2.COLOR_RGB2HSV)
        frame3 = cv2.cvtColor(self.frames[14], cv2.COLOR_RGB2HSV)
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
             
        keyframe = cv2.cvtColor(keyframe, cv2.COLOR_HSV2RGB)
        
        return keyframe
    
    def getAlternativeKeyframe(self):
        i = 0
        keyframeIndex = 0
        rmax = 0
        
        for frame in self.frames:
            difference = self.backgroundDetection(frame)
            sil = self.silhouetteDetection(difference)
            if sil is None:
                return None
            
            x, y = sil.shape
            r = x / y
            
            if (rmax < r):
                rmax = r
                keyframeIndex = i
            i = i + 1
        return self.frames[keyframeIndex]
   
def HOG(img, bin_n):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)    
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...bin_n)
    
    hists = np.zeros(168)
    parting = [1, 4, 16]
    i = 0
        
    for parts in parting:
        bin_cells = np.split(bins, parts)
        mag_cells = np.split(mag, parts)
        hist = np.array([np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)])
        #hist /= hists.sum()
        blockSize = parts * 8
        #hist = hog(img, orientations=8, pixels_per_cell=(parts, parts),
        #            cells_per_block=(1, 1), visualise=False, transform_sqrt=True)
        hists[i : i + blockSize] = np.hstack(hist)
        i += blockSize
            
    return hists


                
if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = ImageWidget()
    form.show()

    sys.exit(app.exec_())


