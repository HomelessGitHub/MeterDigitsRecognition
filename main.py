# -*-coding:utf-8 -*-
from PyQt5 import QtWidgets, QtGui
import sys
from MainUI import Ui_Form
from PyQt5.QtWidgets import QFileDialog
from pre_progress import pre_progress
from inference import inference


class mywindow(QtWidgets.QWidget,Ui_Form):
    def __init__(self, imgName = ""):
        super(mywindow,self).__init__()
        self.setupUi(self)
        self.imgName = imgName
        #定义槽函数
    def openimage(self):
        self.imgName,imgType= QFileDialog.getOpenFileName(self,
                                    "打开图片",
                                    "E:\li\hjl\data",
                                    " *.jpg;;*.png;;*.jpeg;;*.bmp;;All Files (*)")
        png = QtGui.QPixmap(self.imgName).scaled(self.ImageShow.width(), self.ImageShow.height())
        self.ImageShow.setPixmap(png)
        
    def run(self):
        result = ''
        image = pre_progress(self.imgName)
        result = inference(image)
        self.ResultShow.append(result)
    
    
app = QtWidgets.QApplication(sys.argv)
window = mywindow()
window.show()
sys.exit(app.exec_())

