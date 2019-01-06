# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 16:17:52 2018

@author: 蘇
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

#Canny 边缘检测
#step 1 :噪声去除
#step 2 :计算图像梯度
#step 3 :非极大值抑制
#step 4 :滞后阈值

# =============================================================================
# path=r'D:\opencv_learn\opencv_material\cover_1.jpg'
#   
# img = cv2.imread(path)
# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#  
# def nothing(x):
#     pass
#  
#  
#  
# cv2.namedWindow('res',cv2.WINDOW_NORMAL)
# cv2.createTrackbar('min','res',0,100,nothing)
# cv2.createTrackbar('max','res',0,100,nothing)
# while(1):
#     maxVal=cv2.getTrackbarPos('max','res')
#     minVal=cv2.getTrackbarPos('min','res')
#     canny=cv2.Canny(img,10*minVal,10*maxVal,2)
#     cv2.imshow('res',canny)
#     if cv2.waitKey(40) & 0xFF == ord('q'):
#         break
# cv2.destroyAllWindows()
# 
# 
# =============================================================================

#图像金字塔
#略





