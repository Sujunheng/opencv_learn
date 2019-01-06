# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 16:28:54 2019

@author: 蘇
"""
#灰度图CLAHE
import numpy as np
import cv2
import os


def img_gray_CLAHE(path):
    #读取图片
    img = cv2.imread(path,0)
#   高斯滤波，如果需要处理模糊图片，则注销该模糊方法
    img = cv2.GaussianBlur(img,(3,3),0)

    #图像锐化（拉普拉斯算子） [0, -1, 0], [-1, 5, -1], [0, -1, 0]     
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) 
    img = cv2.filter2D(img, -1, kernel=kernel)
    
    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(3, 3))
    # 限制对比度的自适应阈值均衡化
    dst = clahe.apply(img)
 
#    cv2.namedWindow("dst",cv2.WINDOW_NORMAL)
#    cv2.imshow("dst", dst)
#    cv2.imwrite(r'D:\opencv_learn\opencv_material\after\dark_3_1.jpg',dst)
    name=os.path.split(path) 
    img_path=r'D:\opencv_learn\opencv_material\after\gray'
    cv2.imwrite(img_path+'_'+name[1],dst)
#    cv2.waitKey()

#测试
#img_path=r'D:\opencv_learn\opencv_material\after\dark_3.jpg'
#img_gray_CLAHE(img_path)



#遍历文件夹下的文件
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
#批量处理图片并保存
for i in list:
    i=str(i,encoding='utf-8')
    print(i)
    dst=img_gray_CLAHE(i)
#    cv2.imshow('{count}'.format(count=count),dst)
#    cv2.waitKey()
    count+=1
    print(count)