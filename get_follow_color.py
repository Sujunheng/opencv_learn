# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 13:55:09 2018

@author: 蘇
"""
import cv2
import numpy as np



cap = cv2.VideoCapture('D:/work/opencv_learn/output.avi')

while(True):
    ret, frame = cap.read()
    
    if ret == True:

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lower_red = np.array([100, 43, 46])
        
        upper_red = np.array([124, 255, 255])
    
        mask = cv2.inRange(hsv, lower_red, upper_red) #lower20===>0,upper200==>0

        dst = cv2.bitwise_and(frame,frame,mask=mask)#提取特定颜色
        
        cv2.imshow('mask',mask)
        
        cv2.imshow('color',dst)
        
        
    
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break

    else:
        break
cap.release()
cv2.destroyAllWindows()