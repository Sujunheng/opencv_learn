# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 12:03:12 2018

@author: 蘇
"""
import cv2
import numpy as np

#参考资料： csdn 像素运算
#中文翻译文档：  http://www.cnblogs.com/Undo-self-blog/p/8424220.html 10 图像上的算术运算


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