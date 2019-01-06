# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 17:14:55 2018

@author: 蘇
"""

import numpy as np
import cv2

cap = cv2.VideoCapture(1)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
e1 = cv2.getTickCount()
# your code execution

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame,1)
    # Our operations on the frame come here
        #颜色空间转换函数 cv2.cvtColor()
#        https://blog.csdn.net/a1809032425/article/details/82156145
        color = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    
        # Display the resulting frame
        cv2.imshow('frame',color)
    
    #   调用摄像头读取图像数据，以及设置图像信息，使用
    #   cap.set( propId ， value ) 
    #   cap.get( propId )
        one=cap.get(1)
        two=cap.get(2)
        three=cap.get(3)
        four=cap.get(4)
        five=cap.get(5)
        six=cap.get(6)
        k = cv2.waitKey(1) & 0xFF

        cv2.cvtColor(frame,cv2.COLOR_BAYER_BG2GRAY)
        # write the flipped frame
        out.write(frame)
    
        ##每帧数据延时 1ms，延时不能为 0，否则读取的结果会是静态帧
    
        if  k == ord('s'):
            print(one,two,three,four,five,six)
        elif k == ord('q'):
            break
    else:
        break
    
# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()*1000
print(time)