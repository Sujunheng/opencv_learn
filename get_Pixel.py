# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 15:13:03 2018

@author: 蘇
"""


import cv2
import numpy as np
#素材path: D:\work\opencv_learn\opencv-master\opencv-master\samples\data

#获取图像像素
img1_path=r'D:\work\opencv_learn\opencv-master\opencv-master\samples\data\LinuxLogo.jpg'
img2_path=r'D:\work\opencv_learn\opencv-master\opencv-master\samples\data\WindowsLogo.jpg'

img_1=cv2.imread(img1_path)
img_2=cv2.imread(img2_path)
#图片大小 =长*宽*属性通道
#print(img.size)
#像素值，利用numpy的 array.item来快速获取 n行M列 步长为Z的像素点的值
#print(img.item(0,319,2))
#img[0,2] 获取从第n行开始，加m 的像素点集合
#print(img[1,3])

#获取像素点属性  行 列 属性通道 如果没有属性通道值 表示灰度图
#图像类型
#print(img.shape,img.dtype) 

#img.itemset((10,10,2),100)
#print(img.item(10,10,2))she

#逻辑运算 按位运算 与或非
#-----------图像混合,透明化-----------
# 权重越大，透明度越低
def bitwise_and_xuanzhuan():
    img_1_alpha=cv2.imread(img1_path,0)
    img_2_alpha=cv2.imread(img2_path,0)
    alpha = 0.7
    beta = 1-alpha
    gamma = 0
    #g(x)=a*f(x)+b
    #resize第二张图为第一张图的尺寸，同一尺寸化
    h, w = img_1_alpha.shape
    img2=cv2.resize(img_2_alpha,(w,h),interpolation=cv2.INTER_AREA)
    #设置权值，使其透明化
    alpha = cv2.addWeighted(img_1_alpha,alpha,img2,beta,gamma)
    #两张图混合
    bitwise_and=cv2.bitwise_and(img_1,img2)
    cv2.namedWindow('fram',cv2.WINDOW_NORMAL)
    cv2.imshow('alpha',alpha)
    cv2.imshow('fram',bitwise_and)
#----图像旋转---反转--平移---

def _2d():
    rows,cols=img_1.shape[:2]
# 这里的第一个参数为旋转中心，第二个为旋转角度，第三个为旋转后的缩放因子

# 可以通过设置旋转中心，缩放因子，以及窗口大小来防止旋转后超出边界的问题
    M=cv2.getRotationMatrix2D((cols/2,rows/2),200,1.0)
# 第三个参数是输出图像的尺寸中心
    dst=cv2.warpAffine(img_1,M,(cols,rows))

    cv2.imshow('dst',dst)


#---图像对比度，亮度------

#函数addWeighted的原型：
#addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]]) -> dst
#src1表示需要加权的第一个数组（上述例子就是图像矩阵）
#alpha表示第一个数组的权重
#src2表示第二个数组（和第一个数组必须大小类型相同）
#beta表示第二个数组的权重
#gamma表示一个加到权重总和上的标量值
#即输出后的图片矩阵：dst = src1*alpha + src2*beta + gamma;
#属于像素混合加权

def contrast_brightness_image(src1, a, g):
    h, w, ch = src1.shape#获取shape的数值，height和width、通道 
#   新建全零图片数组src2,将height和width，类型设置为原图片的通道类型(色素全为零，输出为全黑图片)
    src2 = np.zeros([h, w, ch], src1.dtype)

#    g(x)=a*f(x)+b
#   其中α(a)调节对比度， β(b)调节亮度
#    参数f(x)表示源图像像素
#    参数g(x)表示输出图像像素
#    参数a（需要满足a>0）被称为增益（gain），常常被用来控制图像的对比度
#    参数b通常被称为偏置，常常被用来控制图像的亮度
#    
#    可以更进一步写成：
#    g(i,j)=a*f(i,j)+b
    dst = cv2.addWeighted(src1, a, src2, 1-a, g)#addWeighted函数说明如下
    cv2.namedWindow("con-bri-demo",cv2.WINDOW_NORMAL)
    cv2.imshow("con-bri-demo", dst)

dark=cv2.imread(r'dark_2.jpg')
#第一个1.2为对比度  第二个为亮度数值越大越亮
contrast_brightness_image(dark, 1.5, 30)
cv2.waitKey(0)


#cv2.imshow('alpha',alpha)
#cv2.imshow('dst',dst)

#if k == ord('q'):
#    break:

