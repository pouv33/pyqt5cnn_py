# --coding:utf-8--
from PyQt5.QtWidgets import (QWidget, QPushButton, QLabel, QComboBox, QFileDialog)
from PyQt5.QtGui import (QPainter, QPen, QFont)
from PyQt5.QtCore import Qt
from PyQt5 import QtGui
from PyQt5.Qt import QIcon
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.Qt import QDir, QIcon
from PyQt5.QtGui import QPixmap
import sys
from PIL import Image
import os

import pyscreenshot as ImageGrab
import Run
import cv2
import numpy


def is_chinese(string):
    """
    检查整个字符串是否包含中文
    :param string: 需要检查的字符串
    :return: bool
    """
    for ch in string:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True

    return False


class MyLabel(QLabel):
    pos_xy = []

    def paintEvent(self, event):
        painter = QPainter()
        painter.begin(self)
        painter.setPen(QPen(Qt.black, 2, Qt.SolidLine))

        if len(self.pos_xy) > 1:
            point_start = self.pos_xy[0]
            for pos_tmp in self.pos_xy:
                point_end = pos_tmp

                if point_end == (-1, -1):
                    point_start = (-1, -1)
                    continue
                if point_start == (-1, -1):
                    point_start = point_end
                    continue

                painter.drawLine(point_start[0], point_start[1], point_end[0], point_end[1])
                point_start = point_end
        painter.end()

    # 记录鼠标点下的点，添加到pos_xy列表
    def mouseMoveEvent(self, event):
        pos_tmp = (event.pos().x(), event.pos().y())
        self.pos_xy.append(pos_tmp)

        self.update()

    # 鼠标释放，在pos_xy中添加断点
    def mouseReleaseEvent(self, event):
        pos_test = (-1, -1)
        self.pos_xy.append(pos_test)

    # 绘制事件
    def btn_clear_on_clicked(self):
        self.pos_xy = []
        self.setText('')
        self.update()

# 界面
class Tan(QWidget):
    def __init__(self):
        super(Tan, self).__init__()

        self.resize(700, 800)  # 外围边框大小
        self.move(550, 95)  # 设置位置
        self.setWindowTitle('基于卷积网络的手写数字识别')  # 标题
        self.setMouseTracking(False)  # False代表不按下鼠标则不追踪鼠标事件

        self.pos_xy = []  # 保存鼠标移动过的点

        # 画板文字
        self.label_draw_name = QLabel(('''<font color=#000000>手写区域：<font>'''), self)
        self.label_draw_name.setGeometry(180, 15, 70, 30)

        # 画板区域
        self.label_draw = MyLabel('', self)
        self.label_draw.setGeometry(180, 45, 450, 300)
        self.label_draw.setStyleSheet("QLabel{border:1px solid white;background-color: #FFFFFF}")
        self.label_draw.setAlignment(Qt.AlignCenter)

        # 图片结果文字
        self.label_result_name = QLabel('''<font color=#000000>识别结果：<font>''', self)
        self.label_result_name.setGeometry(180, 410, 70, 30)
        self.label_result_name.setAlignment(Qt.AlignCenter)

        # 图片结果区域
        self.label_result1 = QLabel(' ', self)
        self.label_result1.setGeometry(180, 440, 450, 300)
        self.label_result1.setStyleSheet("QLabel{border:1px solid white;background-color: #FFFFFF}")
        self.label_result1.setAlignment(Qt.AlignCenter)

        # 识别按钮，跳转到reco方法
        self.btn_recognize = QPushButton('画板识别', self)
        self.btn_recognize.setGeometry(50, 150, 100, 50)
        self.btn_recognize.clicked.connect(self.reco)

        # 随机识别按钮，跳转到mnistreco方法
        self.btn_recognize = QPushButton('随机测试', self)
        self.btn_recognize.setGeometry(50, 520, 100, 50)
        self.btn_recognize.clicked.connect(self.mnistreco)

        # 清除按钮，清空所写数字
        self.btn_clear = QPushButton("擦除画板", self)
        self.btn_clear.setGeometry(50, 600, 100, 50)
        self.btn_clear.clicked.connect(self.label_draw.btn_clear_on_clicked)

    # 识别函数
    def reco(self):
        bbox = (self.x() + 183, self.y() + 83, 1180, 470)  # 设置截屏位置
        im = ImageGrab.grab(bbox)  # 截屏
        im.save("now.png")
        img = cv2.imread("now.png")
        Run.Run(img, 1)  # 调用Run中'CNN'方法对所截图img进行处理

        self.label_result1.setPixmap(
            QtGui.QPixmap('img.png').scaled(self.label_result1.width(),
                                            self.label_result1.height()))  # 在label_result1中显示图片
        self.update()

        ''''''
        bbox = (700, 130, 1215, 900)  # 设置截屏位置
        im = ImageGrab.grab(bbox)  # 截屏
        im.save("prove.png")

    def mnistreco(self):
        Run.gather()
        img = cv2.imread("mnist.png")
        Run.Run(img, 1)  # 调用Run中'CNN'方法对所截图img进行处理

        self.label_result1.setPixmap(
            QtGui.QPixmap('img.png').scaled(self.label_result1.width(),
                                            self.label_result1.height()))  # 在label_result1中显示图片
        self.update()

