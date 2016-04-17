#!/usr/bin/env python

import sys
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
            keyframe = cv2.cvtColor(keyframe, cv2.COLOR_RGB2GRAY)

            #get the gradient of image
            keyframe - cv2.resize(keyframe, (500, 300))
            keyframe = cv2.GaussianBlur(keyframe, (3, 3), 0)                     
            #keyframe = cv2.Laplacian(keyframe, cv2.CV_16S, 3)
            #keyframe = cv2.convertScaleAbs(keyframe)
            #keyframe = cv2.GaussianBlur(keyframe, (3, 3), 0)                     
            #keyframe = cv2.adaptiveThreshold(keyframe, 255 ,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,11,2)
            keyframe = self.countourDetection(keyframe)
                        
            cols, rows = keyframe.shape
            data = keyframe.data
         
            imageKF = QImage(data, rows, cols, QImage.Format_Indexed8)
            pixmapKF = QPixmap.fromImage(imageKF)
            self.ui.labelROI.setPixmap(pixmapKF)
                        
            #clean the frames array
            self.frames[:] = []

    def processImage(self, frame):
        newFrame, roi = self.backgroundDetection(frame)
        return newFrame, roi
        
    def backgroundDetection(self, frame):
        blurred = cv2.GaussianBlur(frame, (15,15), 0)
                        
        #threshold = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        
        #threshold = cv2.dilate(threshold, None, iterations = 2)
        #ret,threshold = cv2.threshold(blurred, 100 ,255,cv2.THRESH_BINARY_INV)
               
        fgmask = self.fgbg.apply(blurred, learningRate=0.3) #self.fgbg.apply(blurred)
        #fgmask = cv2.dilate(fgmask, (5,5) ,iterations = 3)
        #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, (5,5))
        #fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, (5,5))
        
        processedFrame = blurred
        #roi = fgmask
        
        #roi = threshold & fgmask
        #processedFrame, roi = self.countourDetection(processedFrame)
        
        return processedFrame #, roi
    
    def countourDetection(self, frame):
        frame = cv2.Canny(frame, 50, 50)
        
        (cnts, _) = cv2.findContours(frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rois = []
        
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < 150:
                continue
            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            
            print x,y,w,h
            #print x, y
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
            
            #if w > 30 and h > 30 :
            #save roi somewhere
            #    tempframe = frame[y : y + h , x : x + 100].copy()
            #    #roi = cv2.resize(frame, (150, 150))
            #    roi = cv2.resize(frame, (0, 0), fx=0.9, fy=0.9)
             #   rois.append(tempframe)
        
        #if len(rois) > 0:
            #print 'roi exists', rois[0].shape, rois[0].dtype
            #return frame, rois[0]
        #else:
            #return frame, NULL
        return frame
            
    def getFrameHistogram(self, frame):
        hist0 =  cv2.calcHist([frame],[0],None,[256],[0,256])
        hist1 =  cv2.calcHist([frame],[1],None,[256],[0,256])
        hist2 =  cv2.calcHist([frame],[2],None,[256],[0,256])
        
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
            
if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = ImageWidget()
    form.show()

    sys.exit(app.exec_())


