# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 17:26:37 2018

@author: 蘇
"""

#绘制轮廓


# =============================================================================
# import numpy as np
# import cv2
# 
# path=r'D:\opencv_learn\opencv_material\cover_1.jpg'
#  
# img = cv2.imread(path)
# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# 
# imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# ret,thresh = cv2.threshold(imgray,127,255,0)
# image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# #绘制独立轮廓，如第四个轮廓：
# img = cv2.drawContours(img, contours, -1, (0,255,0), 3)
# #但是大多数时候，下面的方法更有用：
# img = cv2.drawContours(img, contours, 3, (0,255,0), 3)
# =============================================================================

#轮廓特征

import cv2
import numpy as np

path=r'D:\opencv_learn\opencv_material\cover_1.jpg'
  
img = cv2.imread(path,0)
ret,thresh = cv2.threshold(img,127,255,0)
image,contours,hierarchy = cv2.findContours(thresh, 1, 2)

cnt = contours[0]
M = cv2.moments(cnt)
epsilon = 0.1*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)