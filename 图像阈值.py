# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 11:04:27 2018

@author: 蘇
"""
#图像阈值，

# =============================================================================
# #简单阈值
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# 
# img = cv2.imread(r'D:\opencv_learn\opencv_material\dark_2.jpg',0)
# 
# gray=cv2.cvtColor(img,cv2.COLOR_BAYER_BG2GRAY)
# 
#这个函数就是 cv2.threshhold()。
#这个函数的第一个参数就是原图像，原图像应该是灰度图。
#第二个参数就是用来对像素值进行分类的阈值。
#第三个参数就是当像素值高于（有时是小于）阈值时应该被赋予的新的像素值。
#OpenCV提供了多种不同的阈值方法，这是有第四个参数来决定的
# ret,thresh1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
# ret,thresh2 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
# ret,thresh3 = cv2.threshold(gray,127,255,cv2.THRESH_TRUNC)
# ret,thresh4 = cv2.threshold(gray,127,255,cv2.THRESH_TOZERO)
# ret,thresh5 = cv2.threshold(gray,127,255,cv2.THRESH_TOZERO_INV)
# 
# titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
# images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
# 
# for i in range(6):
#     plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# 
# plt.show()
# 
# =============================================================================

#自适应阈值
# =============================================================================
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
# 
# img = cv2.imread(r'D:\opencv_learn\opencv_material\dark_2.jpg',0)
#  
# gray=cv2.cvtColor(img,cv2.COLOR_BAYER_BG2GRAY)
# 
# img = cv2.medianBlur(gray,5)
# 
# #此时的阈值是根据图像上的每一个小区域计算与其对应的阈值。
# #因此在同一幅图像上的不同区域采用的是不同的阈值，
# #从而使我们能在亮度不同的情况下得到更好的结果。
# #这种方法需要我们指定三个参数，返回值只有一个。
# #　　• Adaptive Method- 指定计算阈值的方法。
# #　　– cv2.ADPTIVE_THRESH_MEAN_C：阈值取自相邻区域的平均值
# #　　– cv2.ADPTIVE_THRESH_GAUSSIAN_C：阈值取值相邻区域的加权和，权重为一个高斯窗口。
# #　　• Block Size - 邻域大小（用来计算阈值的区域大小）。
# #　　• C - 这就是是一个常数，阈值就等于的平均值或者加权平均值减去这个常数。
# #
# 
# ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#             cv2.THRESH_BINARY,15,2)
# th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#             cv2.THRESH_BINARY,15,2)
# 
# titles = ['Original Image', 'Global Thresholding (v = 127)',
#             'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
# images = [img, th1, th2, th3]
# 
# for i in range(4):
#     plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()
# 
# 
# =============================================================================



#Otsu’ ’s 二值化

#在使用全局阈值时，我们就是随便给了一个数来做阈值，
#那我们怎么知道我们选取的这个数的好坏呢？答案就是不停的尝试。
#如果是一副双峰图像（简单来说双峰图像是指图像直方图中存在两个峰）呢？
#我们岂不是应该在两个峰之间的峰谷选一个值作为阈值？
#这就是 Otsu 二值化要做的。简单来说就是对一副双峰图像自动根据其直方图计算出一个阈值。
#（对于非双峰图像，这种方法得到的结果可能会不理想）。

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread(r'D:\opencv_learn\opencv-master\opencv-master\samples\data\ellipses.jpg',0)

gray=cv2.cvtColor(img,cv2.COLOR_BAYER_BG2GRAY)

img=gray

# global thresholding
ret1,th1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

# Otsu's thresholding
ret2,th2 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(gray,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# plot all the images and their histograms
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
          'Original Noisy Image','Histogram',"Otsu's Thresholding",
          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]

for i in range(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()




