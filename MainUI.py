# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'E:\li\hjl\Lenet-5\MainUI.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(729, 476)
        self.ImageShow = QtWidgets.QLabel(Form)
        self.ImageShow.setGeometry(QtCore.QRect(90, 50, 155, 260))
        self.ImageShow.setBaseSize(QtCore.QSize(299, 522))
        self.ImageShow.setAlignment(QtCore.Qt.AlignCenter)
        self.ImageShow.setObjectName("ImageShow")
        self.OpenFileButton = QtWidgets.QPushButton(Form)
        self.OpenFileButton.setGeometry(QtCore.QRect(110, 380, 112, 34))
        self.OpenFileButton.setObjectName("OpenFileButton")
        self.ResultShow = QtWidgets.QTextBrowser(Form)
        self.ResultShow.setGeometry(QtCore.QRect(430, 230, 221, 41))
        self.ResultShow.setObjectName("ResultShow")
        self.RunButton = QtWidgets.QPushButton(Form)
        self.RunButton.setGeometry(QtCore.QRect(320, 230, 112, 41))
        self.RunButton.setObjectName("RunButton")

        self.retranslateUi(Form)
        self.OpenFileButton.clicked.connect(Form.openimage)
        self.RunButton.clicked.connect(Form.run)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Hi"))
        self.ImageShow.setText(_translate("Form", "请打开图片"))
        self.OpenFileButton.setText(_translate("Form", "打开文件"))
        self.RunButton.setText(_translate("Form", "预测结果："))

