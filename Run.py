import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

import joblib
import random

def gather():
    ''''''
    # 去训练集前100数据部分
    i = random.randint(500, 800)
    # 得到训练集
    # 使用open()函数打开一个文件
    data_file = open("mnist_train.csv", 'r')
    data_list = data_file.readlines()
    data_file.close()

    # np.asfarray()是一个numpy函数，这个函数将文本字符串转换成实数，并创建这些数字的数组。
    all_values = data_list[i].split(',')
    image_array = np.asfarray(all_values[1:]).reshape((28, 28))
    plt.axis('off')
    plt.imshow(image_array, cmap='Greys', interpolation='None')
    plt.savefig('mnist.png')

    result = cv2.imread("muist.png")
    return result

def number_predict(img, model, k):
    # 图像处理
    img = img / 255.0
    # 预测
    if k == 1:
        result = model.predict_classes(img)
    else:
        result = model.predict(img)

    return result

'''
def preProccessing(imgGray):
    kernal = np.ones((3, 3))
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)
    imgDial = cv2.dilate(imgCanny, kernal, iterations=2)
    imgThres = cv2.erode(imgDial, kernal, iterations=1)
    return imgThres
'''



def Run(img, model_x):
    """
    :param img: 输入图像矩阵
    :excute   将标注好的图片存入根目录，命名为“img.png”
    """
    # 载入训练好的模型
    model = load_model('C:\Pycharm\pyqt_cnn\.cnet1.pkl')

    image = cv2.resize(img, (960, 640), interpolation=cv2.INTER_LINEAR)
    # 将这帧转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 二值化
    retval, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)
    # 放大所有轮廓
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cv2.drawContours(binary, contours, i, (255, 255, 255), 5)
    # 过滤噪声点
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        perimeter = cv2.arcLength(contours[i], False)
        if perimeter < 100:
            # print(s)
            cv2.drawContours(binary, contours, i, (0, 0, 0), 15)

    # cv2.imshow('binary_f', binary)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # print(len(contours))
    # 遍历整个图片每个轮廓
    for i in range(len(contours)):
        M = cv2.moments(contours[i])  # 找到中心点
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            x, y, w, h = cv2.boundingRect(contours[i])
            pad = max(w, h) + 10
            #  画出绿色框图
            cv2.rectangle(image, (cx - pad // 2, cy - pad // 2), (cx + pad // 2, cy + pad // 2), (0, 255, 0),
                          thickness=2)
            #  进行预测
            if cy - pad // 2 >= 0 and cx - pad // 2 >= 0:
                number_i = (binary[cy - pad // 2:cy + pad // 2, cx - pad // 2:cx + pad // 2])
                number_i = cv2.resize(number_i, (28, 28))
                if model_x == 1:
                    number_i = np.reshape(number_i, (-1, 28, 28, 1)).astype('float')
                else:
                    number_i = number_i.reshape(1, 784).astype('float32')
                # 将结果红色显示在框上
                result = number_predict(number_i, model, model_x)
                cv2.putText(image, str(result[0]), org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.2,
                            color=(0, 0, 255), thickness=2)

    cv2.imwrite('img.png', image)
    return 0
    # cv2.waitKey()
    # cv2.destroyAllWindows()
