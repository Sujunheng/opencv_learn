# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 15:55:45 2019

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
#   读取图片
    img = cv2.imread(path)  
#   转换色彩空间 RGB2YCrcB
    ycrcb = cv2.cvtColor(img,cv2.COLOR_RGB2YCrCb)
#   色彩通道分离
    channels = cv2.split(ycrcb)
    Y=channels[0]
    Cr=channels[1]
    Cb=channels[2]
    
#    print (len(channels))    
#   对Y通道进行处理
#    img2=y
#   对灰度通道进行处理
    img2=cv2.imread(path,0)
#   高斯滤波，如果需要处理模糊图片，则注销该模糊方法
    img_gauss = cv2.GaussianBlur(img2,(3,3),0)
    
#CV_EXPORTS_W void filter2D( InputArray src, 
#                               OutputArray dst, 
#                               int ddepth,
#                               InputArray kernel, 
#                               Point anchor=Point(-1,-1),
#                               double delta=0, 
#                               int borderType=BORDER_DEFAULT )
#InputArray src: 输入图像
#
#OutputArray dst: 输出图像，和输入图像具有相同的尺寸和通道数量
#
#int ddepth: 目标图像深度，如果没写将生成与原图像深度相同的图像。原图像和目标图像支持的图像深度如下：
#
#    src.depth() = CV_8U, ddepth = -1/CV_16S/CV_32F/CV_64F
#    src.depth() = CV_16U/CV_16S, ddepth = -1/CV_32F/CV_64F
#    src.depth() = CV_32F, ddepth = -1/CV_32F/CV_64F
#    src.depth() = CV_64F, ddepth = -1/CV_64F
#    
#当ddepth输入值为-1时，目标图像和原图像深度保持一致。
#
#InputArray kernel: 卷积核（或者是相关核）,一个单通道浮点型矩阵。如果想在图像不同的通道使用不同的kernel，可以先使用split()函数将图像通道事先分开。
#
#Point anchor: 内核的基准点(anchor)，其默认值为(-1,-1)说明位于kernel的中心位置。基准点即kernel中与进行处理的像素点重合的点。
#
#double delta: 在储存目标图像前可选的添加到像素的值，默认值为0
#
#int borderType: 像素向外逼近的方法，默认值是BORDER_DEFAULT,即对全部边界进行计算。

    
    
#   图像锐化（拉普拉斯算子） [0, -1, 0], [-1, 5, -1], [0, -1, 0] 
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    img_ruihua = cv2.filter2D(img_gauss, -1, kernel=kernel)

#   创建CLAHE对象
#   Contrast Limited AHE（CLAHE）是自适应直方图均衡的变体，
#   其中对比度放大是有限的，以便减少噪声放大的这个问题。
#   'ClipLimit'是一种对比因子，可防止图像过饱和，特别是在均匀区域。
#   由于许多像素落入相同的灰度级范围内，这些区域的特征在于特定图像区块的直方图中的高峰。
#   没有剪辑限制，自适应直方图均衡技术可能产生在某些情况下比原始图像更差的结果。
#   所以在一个小区域内，直方图会限制在一个小区域（除非有噪音）。如果有噪音，它会被放大
#   “tileGridSize”图像块大小设置.然后每一个方块都是像平常一样的直方图处理

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(3, 3))
    # 限制对比度的自适应阈值均衡化
    dst = clahe.apply(img_ruihua)
    #色彩通道合并
    src=cv2.merge([dst,Cr,Cb])
    #色彩通道转换
    src=cv2.cvtColor(src,cv2.COLOR_YCrCb2RGB)
    #图片显示窗口
#    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
#    cv2.imshow('image',src )
    #保存生成图片
    name=os.path.split(path) 
    img_path=r'D:\opencv_learn\opencv_material\after\color'
    cv2.imwrite(img_path+'_graymohu_'+name[1],src)
    #测试生成图片
#    cv2.imwrite('D:\opencv_learn\opencv_material\dark\\dark_3\dark_3_gauss_4_3x3.jpg',src)
    

#   遍历文件夹下的文件
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


#   读取文件夹下的文件，循环处理图片，并保存图片
list = GetFileList(r'D:\opencv_learn\opencv_material\mohu', [])

count=0
for i in list:
    i=str(i,encoding='utf-8')
    print(i)
    dst=hisEqulColor(i)
    count+=1
    print(count)



