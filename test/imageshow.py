#!/usr/bin/env python3
"""PyQt5 GUI for the fall-detection pipeline.

The processing stages live in the falldetect package; this module only does
video capture, display and the per-window orchestration (mirrored by the
headless falldetect.pipeline.Pipeline).
"""

import os
import sys

import cv2

from PyQt5 import QtCore
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog

# repo root on sys.path so `from ui import uiform` works when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui import uiform
from falldetect.classifier import LABELS, predict_label, train_classifier
from falldetect.features import hog_multiscale as HOG
from falldetect.features import silhouette_cropped
from falldetect.keyframe import select_keyframe

WINDOW_SIZE = 20


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

        self.frames = []
        self.background = None

        self.classifier = train_classifier()
        self.behaviourLabelTable = list(LABELS)
        self.ui.labelBehaviour.setText('Behaviour')

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

        if self.ui.checkboxCam.isChecked():
            self.reinitCam(0)
        else:
            filename = self.ui.lineFilename.text()
            self.reinitCam(filename)

        backgroundFilename = self.ui.lineFilenameBackground.text()
        self.background = cv2.imread(backgroundFilename, 1)

        if self.background is None:
            print('background image not readable:', backgroundFilename)
        else:
            self.background = cv2.resize(self.background, (500, 300))

        if text == 'Play':
            self.setTimer(50)
            self.ui.buttonPlay.setText('Stop')
        else:
            self.timer.stop()
            self.ui.buttonPlay.setText('Play')

    def onButtonUseCamClicked(self):
        self.reinitCam(0)

        if self.cap.isOpened():
            print('Success!')

    def onTimerTimeout(self):
        retVal, frame = self.cap.read()

        if retVal != True:
            self.timer.stop()
            self.cap.release()
            self.ui.buttonPlay.setText('Play')
            print('frame not captured')
            return

        frame = cv2.resize(frame, (500, 300))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)

        grayFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        cols, rows = grayFrame.shape

        image = QImage(grayFrame.tobytes(), rows, cols, rows, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(image)
        self.ui.labelImage.setPixmap(pixmap)

        grayFrame = cv2.GaussianBlur(grayFrame, (3, 3), 0)

        sil = self.silhouetteDetectionCropped(grayFrame)

        if sil is not None:
            self.frames.append(frame)

        if len(self.frames) >= WINDOW_SIZE:
            label = self.classifyWindow()
            if label is not None:
                self.ui.labelBehaviour.setText(label)
            self.frames = []

    def classifyWindow(self):
        """Keyframe -> background hue difference -> silhouette -> HOG -> k-NN.
        Mirrors falldetect.pipeline.Pipeline._classify_window."""
        if self.background is None:
            return None

        keyframe = select_keyframe(self.frames)
        keyframe = cv2.cvtColor(keyframe, cv2.COLOR_RGB2HSV)
        background = cv2.cvtColor(self.background, cv2.COLOR_RGB2HSV)

        h = cv2.split(keyframe)[0] - cv2.split(background)[0]
        h = cv2.GaussianBlur(h, (3, 3), 0)

        silhouette = self.silhouetteDetectionCropped(h)
        if silhouette is None:
            return None

        edges = cv2.Canny(silhouette, 50, 5)
        return predict_label(self.classifier, HOG(edges, 8))

    def silhouetteDetectionCropped(self, frame):
        return silhouette_cropped(frame)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = ImageWidget()
    form.show()

    sys.exit(app.exec_())
