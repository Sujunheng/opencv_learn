# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 15:30:04 2019

@author: 蘇
"""

#格式转换
from PIL import Image
import os

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
list = GetFileList(r'D:\opencv_learn\opencv_material\public', [])

for i in list:
    i=str(i,encoding='utf-8')
    print(i)
    name=os.path.split(i)
    name_type=name[1].split('.')
    img = Image.open(i)
    jpg_pic=name_type[0]+'.png'
    path='D:/opencv_learn/opencv_material/public_png/'
    img.save(path+jpg_pic)
