# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 14:33:17 2018

@author: 蘇
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt


# =============================================================================
# 2D卷积
# path=r'D:\opencv_learn\opencv_material\cover_1.jpg'
# 
# img = cv2.imread(path)
# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# 
# kernel = np.ones((5,5),np.float32)/25
# dst = cv2.filter2D(img,-1,kernel)
# 
# plt.subplot(121),plt.imshow(img),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
# plt.xticks([]), plt.yticks([])
# plt.show()
# =============================================================================



# =============================================================================
# # 图像梯度
# 
# path=r'D:\opencv_learn\opencv_material\cover_1.jpg'
#  
# img = cv2.imread(path)
# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# 
# laplacian = cv2.Laplacian(img,cv2.CV_64F)
# sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
# sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
# 
# plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
# plt.title('Original'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
# plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
# plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
# plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
# plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
# 
# plt.show()
# 
# =============================================================================


path=r'D:\opencv_learn\opencv_material\cover_1.jpg'
 
img = cv2.imread(path)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


# Output dtype = cv2.CV_8U
sobelx8u = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5)

# Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
abs_sobel64f = np.absolute(sobelx64f)
sobel_8u = np.uint8(abs_sobel64f)

plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(sobelx8u,cmap = 'gray')
plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(sobel_8u,cmap = 'gray')
plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])
plt.show()