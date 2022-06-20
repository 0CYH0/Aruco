#encoding:utf-8
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

# matplotlib模块绘制直方图
def Histogram_H(data):
    
    # 绘制直方图
    plt.hist(x = data, bins = 180, color = 'steelblue', edgecolor = 'black', label = '直方图')
    
    # 添加x轴和y轴标签
    plt.xlabel('Value of H channel')
    plt.ylabel('Quantity under this value')
    
    # 添加标题
    plt.title('H channel')
    
    # 显示图形
    plt.show()

def Histogram_S(data):
    
    # 绘制直方图
    plt.hist(x = data, bins = 256, color = 'steelblue', edgecolor = 'black', label = '直方图')
    
    # 添加x轴和y轴标签
    plt.xlabel('Value of S channel')
    plt.ylabel('Quantity under this value')
    
    # 添加标题
    plt.title('S channel')
    
    # 显示图形
    plt.show()

def Histogram_V(data):
    
    # 绘制直方图
    plt.hist(x = data, bins = 256, color = 'steelblue', edgecolor = 'black', label = '直方图')
    
    # 添加x轴和y轴标签
    plt.xlabel('Value of V channel')
    plt.ylabel('Quantity under this value')
    
    # 添加标题
    plt.title('V channel')
    
    # 显示图形
    plt.show()

cap = cv2.VideoCapture(1)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) # 格式
cap.set(cv2.CAP_PROP_FPS, 40) # 帧率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # 宽度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # ⾼度
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) # 设置取消⾃动曝光
cap.set(cv2.CAP_PROP_EXPOSURE, -3) # 设置曝光时间

while True:
    
    ret, frame = cap.read()
    
    #将BGR图像转换为HSV图像
    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)    
    #v = cv2.split(frame_HSV)
 
    """
    #使用v通道计算图像的平均值和分布
    mu = v.mean()    hsv = frame_HSV# / 255.0
    v = hsv[:, :, 2]
    print("v value :", v)
    sigma = v.std()
    print("Mean : ", mu)
    print("Stdev : ", sigma)
    """
    
    cv2.imshow('frame', frame)
    cv2.imshow('frame_HSV', frame_HSV)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        
        v = frame_HSV[:, :, 2] #获取图像v通道的值
        s = frame_HSV[:, 1, :] #获取图像s通道的值
        h = frame_HSV[0, :, :] #获取图像h通道的值
        #print("v value :", v)
        
        data_v = v.flatten() #将获取的v值又二维数组转化为一维数组
        data_s = s.flatten() #将获取的v值又二维数组转化为一维数组
        data_h = h.flatten() #将获取的v值又二维数组转化为一维数组
        print("h value :", h)
        print("s value :", s)
        print("v value :", v)
        
        cap.release()
        cv2.destroyAllWindows()
        break

Histogram_H(data_h)
Histogram_S(data_s)
Histogram_V(data_v)

