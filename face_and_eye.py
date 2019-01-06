# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 18:23:31 2018

@author: 蘇
"""

import numpy as np
import cv2


#path_face='D:/work/opencv_learn/haarcascade_frontalface_default.xml'
#path_eye='D:/work/opencv_learn/haarcascade_eye.xml'
#
#face_cascade = cv2.CascadeClassifier(path_face)
#eye_cascade = cv2.CascadeClassifier(path_eye)
#
#cap = cv2.VideoCapture('D:/work/opencv_learn/output.avi')
#
#while(cap.isOpened()):
#    ret, frame = cap.read()
#
#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#    
#    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#    for (x,y,w,h) in faces:
#        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
#        roi_gray = gray[y:y+h, x:x+w]
#        roi_color = frame[y:y+h, x:x+w]
#        eyes = eye_cascade.detectMultiScale(roi_gray)
#        for (ex,ey,ew,eh) in eyes:
#            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
#            
#    cv2.imshow('frame',frame)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
#
#
#cap.release()
#cv2.destroyAllWindows()
#

# OpenCV 创建一个面部和眼部检测器 
path_face='D:/work/opencv_learn/haarcascade_frontalface_default.xml'
path_eye='D:/work/opencv_learn/haarcascade_eye.xml'
#我们要加载需要的 XML 分类器。然后以灰度格式加载输入图像或者是视频
face_cascade = cv2.CascadeClassifier(path_face)
eye_cascade = cv2.CascadeClassifier(path_eye)

#调用摄像头
cap = cv2.VideoCapture(1)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
#fps = videoCapture.get(cv2.cv.CV_CAP_PROP_FPS)
out = cv2.VideoWriter('output.avi',fourcc, 15.0, (640,480))
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
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x,y,w,h) in faces:
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    
        # Display the resulting frame
        cv2.imshow('frame',frame)
    
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
    
        # write the flipped frame
        out.write(frame)
    
        ##每帧数据延时 1ms，延时不能为 0，否则读取的结果会是静态帧
    
        if  k == ord('s'):
            print(one,two,three,four,five,six)
        elif k == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
e2 = cv2.getTickCount()
time = (e2 - e1)/ cv2.getTickFrequency()*1000
print(time)