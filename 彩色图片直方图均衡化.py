# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 14:22:25 2019

@author: 蘇
"""

#彩色图片直方图均衡化
import numpy as np
import cv2
import os

#让我们考虑一个简单的灰度图像的直方图均衡。由于它只是一个通道图像，
#因此您可以查看该平面中的值，并根据图像的2D直方图对它们进行均衡。
#现在，RGB图像由三个通道组成。很自然地，你可以将它们分成三个独立的通道，
#在每个通道上应用直方图均衡，然后将它们组合在一起，对吗？好吧，不完全是！
#
#非线性直方图均衡是一个非线性过程。分别对每个通道进行通道分割和均衡是不正确的。
#均衡涉及图像的强度值，而不是颜色分量。因此，对于简单的RGB彩色图像，直方图均衡不能直接应用于通道。
#需要以这样的方式应用它，使得强度值相等而不会干扰图像的色彩平衡。
#因此，第一步是将图像的颜色空间从RGB转换为将强度值与颜色分量分开的颜色空间之一。
#一些可能的选项是HSV / HLS，YUV，YCbCr等.YCbCr是首选，
#因为它是为数字图像设计的。在强度平面Y上执行直方图均衡。现在将结果YCbCr图像转换回RGB。

def hisEqulColor(path):
    img = cv2.imread(path)
    
    ycrcb = cv2.cvtColor(img,cv2.COLOR_RGB2YCrCb)
    channels = cv2.split(ycrcb)
    y=channels[0]
    Cr=channels[1]
    Cb=channels[2]
    print (len(channels))
     
#对Y通道进行处理
#    img2=y
#对灰度通道进行处理
    img2=cv2.imread(path,0)
    #高斯模糊    
    img_gauss = cv2.GaussianBlur(img2,(3,3),0)
    #laplace算子锐化
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #锐化
    img_ruihua = cv2.filter2D(img2, -1, kernel=kernel)
    # 创建CLAHE对象
#    使用自适应直方图均衡.图像被划分为几个小块，
#    称为“tiles”(在OpenCV中默认值是8x8).然后每一个方块都是像平常一样的直方图
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(3, 3))
    # 限制对比度的自适应阈值均衡化
    dst = clahe.apply(img_ruihua)

#    cv2.merge([ynew,crnew,cbew], ycrcb)
    src=cv2.merge([dst,Cr,Cb])
    src=cv2.cvtColor(src,cv2.COLOR_YCrCb2RGB)
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.imshow('image',src )
    name=os.path.split(path) 
    img_path=r'D:\opencv_learn\opencv_material\after\color'
    cv2.imwrite(img_path+'_graymohu_'+name[1],src)
#    cv2.imwrite('D:\opencv_learn\opencv_material\dark\\dark_3\dark_3_gauss_4_3x3.jpg',src)
    




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
 
list = GetFileList(r'D:\opencv_learn\opencv_material\mohu', [])

count=0
for i in list:
    i=str(i,encoding='utf-8')
    print(i)
    dst=hisEqulColor(i)
#    cv2.imshow('{count}'.format(count=count),dst)
#    cv2.waitKey()
    count+=1
    print(count)
#cv2.imshow('im1', im)



