# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 18:31:31 2019

@author: 蘇
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

path=r'D:\opencv_learn\opencv_material\light_1.jpg'
# =============================================================================
# 
# #计算并创建，显示灰度直方图
# def calcGrayHist(I):
#     # 计算灰度直方图
#     h, w = I.shape[:2]
#     grayHist = np.zeros([256], np.uint64)
#     for i in range(h):
#         for j in range(w):
#             grayHist[I[i][j]] += 1
#     return grayHist
#  
# img = cv.imread(path, 0)
# grayHist = calcGrayHist(img)
# x = np.arange(256)
# # 绘制灰度直方图
# plt.plot(x, grayHist, 'r', linewidth=2, c='black')
# plt.xlabel("gray Label")
# plt.ylabel("number of pixels")
# plt.show()
# cv.imshow("img", img)
# cv.waitKey()
# 
# =============================================================================

# =============================================================================
# #matplotlib自带方法画直方图
# def matplotlib_drawHist():
#     img = cv.imread(path, 0)
#     h, w = img.shape[:2]
#     pixelSequence = img.reshape([h * w, ])
#     numberBins = 256
#     histogram, bins, patch = plt.hist(pixelSequence, numberBins,
#                                       facecolor='black', histtype='bar')
#     plt.xlabel("gray label")
#     plt.ylabel("number of pixels")
#     plt.axis([0, 255, 0, np.max(histogram)])
#     plt.show()
#     cv.imshow("img", img)
#     cv.waitKey()
# =============================================================================

#直方图正规化
# =============================================================================
#自行实现
# img = cv.imread(path, 0)
# gray = cv.cvtColor(img,cv.COLOR_BAYER_BG2GRAY)
# # 计算原图中出现的最小灰度级和最大灰度级
# # 使用函数计算
# Imin, Imax = cv.minMaxLoc(gray)[:2]
# # 使用numpy计算
# # Imax = np.max(img)
# # Imin = np.min(img)
# Omin, Omax = 0, 255
# # 计算a和b的值
# a = float(Omax - Omin) / (Imax - Imin)
# b = Omin - a * Imin
# out = a * img + b
# out = out.astype(np.uint8)
# out = cv.cvtColor(out,cv.COLOR_GRAY2RGB)
# cv.imshow("img", img)
# cv.imshow("out", out)
# cv.waitKey()
# =============================================================================

# =============================================================================
# #直方图正规化
#借助opencv 提供的API实现
# 代码中计算原图中出现的最小灰度级和最大灰度级可以使用OpenCV提供的函数 
# 
# minVal, maxVal, minLoc, maxLoc  = cv.minMaxLoc(src[, mask])
# 
# 返回值分别为：最小值，最大值，最小值的位置索引，最大值的位置索引。
# 
# 正规化函数normalize: dst=cv.normalize(src, dst[, alpha[, beta[, norm_type[, dtype[, mask]]]]])
# 
# img = cv.imread(path, 0)
# out = np.zeros(img.shape, np.uint8)
# cv.normalize(img, out, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
# cv.imshow("img", img)
# cv.imshow("out", out)
# cv.waitKey()
# 
# =============================================================================



def img_gray_CLAHE(path):
    #自适应直方图均衡化
    img = cv2.imread(path,0)
    # 均值滤波
    #img = cv2.blur(img, (3,3))
    
    # 高斯滤波
    img = cv2.GaussianBlur(img,(3,3),0)
    
    # 中值滤波
    #img = cv2.medianBlur(img, 3)
    
    # 高斯双边滤波
#    img = cv2.bilateralFilter(img,9,75,75)
    
    #均值迁移
    #img=cv.pyrMeanShiftFiltering(image,10,50)
    
    #图像锐化（拉普拉斯算子） [0, -1, 0], [-1, 5, -1], [0, -1, 0] 这个矩阵
    
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
    img = cv2.filter2D(img, -1, kernel=kernel)
    
    src = cv2.imread(path)
    #b, g, r = cv2.split(img1)
    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(3, 3))
    # 限制对比度的自适应阈值均衡化
    dst = clahe.apply(img)
    # 使用全局直方图均衡化
    equa = cv2.equalizeHist(img)
    
    #cv2.imshow('result',equa1)
    #cv2.namedWindow("equal",cv2.WINDOW_NORMAL)
    #cv2.imshow('equal',equa)
    ## 分别显示原图，CLAHE，HE
    #cv2.namedWindow("img",cv2.WINDOW_NORMAL)
    #cv2.imshow("img", src)
#    cv2.namedWindow("dst",cv2.WINDOW_NORMAL)
#    cv2.imshow("dst", dst)
#    cv2.imwrite(r'D:\opencv_learn\opencv_material\after\dark_3_1.jpg',dst)
    name=os.path.split(path) 
    img_path=r'D:\opencv_learn\opencv_material\after\gray'
    cv2.imwrite(img_path+'_'+name[1],dst)
#    cv2.waitKey()
#    return dst

#
#img_path=r'D:\opencv_learn\opencv_material\after\dark_3.jpg'
#img_do(img_path)




def GetFileList(dir, fileList):
    newDir = dir
    if os.path.isfile(dir):
        fileList.append(dir.encode('utf-8'))
    elif os.path.isdir(dir):  
        for s in os.listdir(dir):
            #如果需要忽略某些文件夹，使用以下代码
            #if s == "xxx":
                #continue
            newDir=os.path.join(dir,s)
            GetFileList(newDir, fileList)  
    return fileList
 
list = GetFileList(r'D:\opencv_learn\opencv_material\all', [])
count=0
for i in list:
    i=str(i,encoding='utf-8')
    print(i)
    dst=img_do(i)
#    cv2.imshow('{count}'.format(count=count),dst)
#    cv2.waitKey()
    count+=1
    print(count)
















# =============================================================================
# src = cv2.imread(path)
# src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# # RGB在opencv中存储为BGR的顺序,数据结构为一个3D的numpy.array,索引的顺序是行,列,通道:
# B = src[:,:,0]
# G = src[:,:,1]
# R = src[:,:,2]
# # 灰度g=p*R+q*G+t*B（其中p=0.2989,q=0.5870,t=0.1140），于是B=(g-p*R-q*G)/t。于是我们只要保留R和G两个颜色分量，再加上灰度图g，就可以回复原来的RGB图像。
# g = src_gray[:]
# p = 0.2989; q = 0.5870; t = 0.1140
# B_new = (g-p*R-q*G)/t
# B_new = np.uint8(B_new)
# src_new = np.zeros((src.shape)).astype("uint8")
# src_new[:,:,0] = B_new
# src_new[:,:,1] = G
# src_new[:,:,2] = R
# cv2.imshow('result',src_new)
# cv2.waitKey()
# =============================================================================
