# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '../test/form.ui'
#
# Created by: PyQt5 UI code generator 5.5.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(581, 469)
        self.labelImage = QtWidgets.QLabel(Form)
        self.labelImage.setGeometry(QtCore.QRect(20, 20, 471, 311))
        self.labelImage.setText("")
        self.labelImage.setObjectName("labelImage")
        self.lineFilename = QtWidgets.QLineEdit(Form)
        self.lineFilename.setGeometry(QtCore.QRect(20, 360, 381, 27))
        self.lineFilename.setObjectName("lineFilename")
        self.buttonOpen = QtWidgets.QPushButton(Form)
        self.buttonOpen.setGeometry(QtCore.QRect(420, 360, 81, 27))
        self.buttonOpen.setObjectName("buttonOpen")
        self.buttonPlay = QtWidgets.QPushButton(Form)
        self.buttonPlay.setGeometry(QtCore.QRect(220, 410, 99, 27))
        self.buttonPlay.setObjectName("buttonPlay")
        
        self.checkboxCam = QtWidgets.QCheckBox(Form)
        self.checkboxCam.move(420, 420)
        self.buttonPlay.setObjectName("checkboxCam")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.buttonOpen.setText(_translate("Form", "Open"))
        self.buttonPlay.setText(_translate("Form", "Play"))

