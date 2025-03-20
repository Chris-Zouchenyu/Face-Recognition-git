# -*- coding: utf-8 -*-

"""
Module implementing MainWindow.
"""

from PyQt5.QtCore import pyqtSlot,QThread
from PyQt5.QtWidgets import QMainWindow,QApplication,QFileDialog
from PyQt5.QtGui import QPixmap
import torch

from Ui_window2 import Ui_MainWindow
import os
import sys

class Thread_contrast(QThread):
    '''
    线程1
    '''
    def __init__(self,path1,path2):
        super().__init__()
        self.path1 = path1
        self.path2 = path2
    def run(self):
        self.detect1()

    def detect1(self,path1,path2):
        from sia.predict import con
        probability = con.contrast(path1,path2)
        return probability

class MainWindow(QMainWindow, Ui_MainWindow):
    """
    Class documentation goes here.
    """

    def __init__(self, parent=None):
        """
        Constructor

        @param parent reference to the parent widget (defaults to None)
        @type QWidget (optional)
        """
        super().__init__(parent)
        self.setupUi(self)
        self.num = 0
        self.new_num = 0
        self.file_path2 = 0

    @pyqtSlot()
    def on_pushButton_clicked(self):
        """
        Slot documentation goes here.
        """
        # 打开文件对话框，选择图片文件
        file_path, _ = QFileDialog.getOpenFileName(
            self,  # 父窗口
            "选择图片",  # 对话框标题
            "",  # 默认打开的目录（空字符串表示当前目录）
            "图片文件 (*.png *.jpg *.bmp *.jpeg)"  # 文件过滤器，只显示图片文件
        )
        self.textBrowser.append(file_path)
        from test_zcy import Face_Detect
        num = Face_Detect.face_detection(file_path)
        self.textBrowser_3.append(str(num))
        self.num = num
        new_file_name = 'img_detect\img_original.jpg'
        self.load_image1(new_file_name)  # 加载并显示图片

    def load_image1(self, image_path):
        """
        加载图片并显示在QLabel中

        @param image_path 图片文件路径
        @type str
        """
        pixmap = QPixmap(image_path)  # 创建QPixmap对象
        if not pixmap.isNull():  # 检查图片是否加载成功
            self.label_9.setPixmap(pixmap)  # 在label_9中显示图片
            self.label_9.setScaledContents(True)  # 让图片自适应QLabel的大小
        else:
            print("图片加载失败，请检查文件路径或格式。")    

    @pyqtSlot()
    def on_pushButton_2_clicked(self):
        """
        Slot documentation goes here.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,  # 父窗口
            "选择图片",  # 对话框标题
            "",  # 默认打开的目录（空字符串表示当前目录）
            "图片文件 (*.png *.jpg *.bmp *.jpeg)"  # 文件过滤器，只显示图片文件
        )
        self.textBrowser_2.append(file_path)
        self.file_path2 = file_path
        self.load_image2(file_path)  # 加载并显示图片
    def load_image2(self, image_path):
        """
        加载图片并显示在QLabel中

        @param image_path 图片文件路径
        @type str
        """
        pixmap = QPixmap(image_path)  # 创建QPixmap对象
        if not pixmap.isNull():  # 检查图片是否加载成功
            self.label_10.setPixmap(pixmap)  # 在label_10中显示图片
            self.label_10.setScaledContents(True)  # 让图片自适应QLabel的大小
        else:
            print("图片加载失败，请检查文件路径或格式。") 

    @pyqtSlot()
    def on_pushButton_3_clicked(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        # try:
        # 将路径添加到 sys.path
        img_path = r'detect\face_' + str(self.new_num) + '.jpg'
        self.load_image3(img_path)
        self.Thread1 = Thread_contrast(img_path,self.file_path2)
        probability = self.Thread1.detect1(img_path,self.file_path2)
        with torch.no_grad():
            probability = probability.numpy()
            self.textBrowser_4.append(str(probability))
        # print(probability)
        # except:
        #     print('照片地址错误，请重新输入')

    def load_image3(self, image_path):
        """
        加载图片并显示在QLabel中

        @param image_path 图片文件路径
        @type str
        """
        pixmap = QPixmap(image_path)  # 创建QPixmap对象
        if not pixmap.isNull():  # 检查图片是否加载成功
            self.label_11.setPixmap(pixmap)  # 在label_10中显示图片
            self.label_11.setScaledContents(True)  # 让图片自适应QLabel的大小
    @pyqtSlot()
    def on_pushButton_4_clicked(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        raise NotImplementedError

    @pyqtSlot(int, int)
    def on_lineEdit_cursorPositionChanged(self, p0, p1):
        """
        Slot documentation goes here.

        @param p0 DESCRIPTION
        @type int
        @param p1 DESCRIPTION
        @type int
        """
        num = self.num
        p00 = 1
        try:
            p00 = int(self.lineEdit.text())
            if p00 > num:
                print("数量过大，请重新输入")
            else:
                self.new_num = p00
                print(self.new_num)
        except ValueError:  # 如果转换失败
            p00 = 0  # 给一个默认值
            # 或者提示用户输入有效值
            print("错误：请输入一个有效的整数")
        
    
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    ui = MainWindow()
    ui.show()
    sys.exit(app.exec_())

