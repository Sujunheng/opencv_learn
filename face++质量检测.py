# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 15:26:29 2019

@author: 蘇
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 18:10:07 2018

@author: 蘇
"""
import os
import urllib
import requests as req
import json
import base64
from io import BytesIO


def get_face(path):
    compare_url = "https://api-cn.faceplusplus.com/facepp/v3/detect"
    
    Api_key = "bdUfoto5F_ecfilBNsWoRxknmip09BoO"
    
    secret_key = "t6r0rjvX0wXBtR1NMnyW6sn-Gaxd-sK_"
        
    attributes='gender,age,smiling,headpose,facequality,blur,eyestatus,emotion,ethnicity,beauty,mouthstatus,eyegaze,skinstatus'
    #读取图片并用base64编码
    with open(path,'rb') as f:# 将这个图片保存在内存
        
        base64_img = base64.b64encode(f.read())
    
    data = {"api_key": Api_key,\
            "api_secret": secret_key,\
            "image_base64":base64_img,\
            "return_attributes":attributes,
            }
      
    req_rst = req.post(compare_url, data=data)
    #对返回的byte类型数据解码 utf-8
    content=req_rst.content.decode('utf-8')
    #用json格式的方式加载内容
    content_json = json.loads(content)
    
    return content_json

def get_facequality(path):
    erro_list=[]
    try:
        face_info=get_face(path).get('faces')
        #face_info为list
        face_token=str(face_info[0].get('face_token'))
        face_attributes=face_info[0].get('attributes')
        #人脸框
        face_bbox=face_info[0].get('face_rectangle')
        bbox_width=str(face_bbox.get('width'))
        bbox_top=str(face_bbox.get('top'))
        bbox_left=str(face_bbox.get('left'))
        bbox_height=str(face_bbox.get('height'))
        #年龄，性别
        age=str(face_attributes['age'].get('value'))
        gender=str(face_attributes['gender'].get('value'))
        #人脸质量检测
        headpose=face_attributes['headpose']
        #人脸偏转
        headpose_yaw_angel=str(headpose.get('yaw_angle'))
        headpose_pitch_angle=str(headpose.get('pitch_angle'))
        headpose_roll_angle=str(headpose.get('roll_angle'))
        
        #人脸模糊判断
        blur=face_attributes['blur'].get('blurness')
        blurness_threshold=str(blur['threshold'])
        blurness_value=str(blur['value'])
        
        gaussianblur=face_attributes['blur'].get('gaussianblur')
        gaussianblur_threshold=str(gaussianblur['threshold'])
        gaussianblur_value=str(gaussianblur['value'])
        
        motionblur=face_attributes['blur'].get('motionblur')
        motionblur_threshold=str(motionblur['threshold'])
        motionblur_value=str(motionblur['value'])
        #人脸质量评分
        facequality=face_attributes['facequality']
        facequality_threshold=str(facequality['threshold'])
        facequality_value=str(facequality['value'])
        
#        name=os.path.split(path)
#        wfpath=name[0]+'\\all.txt'
#        with open(wfpath,'a+') as f:
    #        'url'+' '+'face_token'+' '+'bbox'+' '+'bbox_width'+' '+'bbox_top'+' '\
    #                +'bbox_left'+' '+'bbox_height'+' '+'age'+' '+'gender'+' '+'headpose_pitch_angle'+' '\
    #                +'headpose_roll_angle'+' '+'headpose_yaw_angel'+' '\
    #                +'blurness_threshold'+' '+'blurness_value'+' '\
    #                +'gaussianblur_threshold'+' '+'gaussianblur_value'+' '\
    #                +'motionblur_threshold'+' '+'motionblur_value'+' '\
    #                +'facequality_threshold'+' '+'facequality_value'+' '
#            f.write(name[1]+' '\
#                    +face_token+' '\
#                    +bbox_width+','+bbox_top+','+bbox_left+','+bbox_height+' '\
#                    +age+' '\
#                    +gender+' '\
#                    +headpose_pitch_angle+','+headpose_roll_angle+','+headpose_yaw_angel+' '\
#                    +blurness_threshold+' '+blurness_value+' '\
#                    +gaussianblur_threshold+' '+gaussianblur_value+' '\
#                    +motionblur_threshold+' '+motionblur_value+' '\
#                    +facequality_threshold+' '+facequality_value+'\n')
        return facequality
    except:
        erro_list.append(path)
        return erro_list


#测试    
# =============================================================================
# result_path 结果保存路径
# 检测图片Url
# url='http://inno-pd-photo-roads-show.oss-cn-hangzhou.aliyuncs.com/2743/2743_2743-2153-192.168.12.100_01_20181209153030090_FACE_SNAP_slice_20181210234951455291.jpg'
# 
# a=get_face(url)
# b=get_facequality(a,path)
# =============================================================================

#循环遍历
#需要检测的图片url path
#url_path=r'yashili_url.txt'
#保存检测结果  path
#result_path=r'yashili_result.txt'

#with open(url_path,'r')as f_url:
#    url_all=f_url.readlines()
#for each in url_all:
#    i=each.split()[0]
#    get_facequality(i,result_path)

#
#light_1_path=r'D:\opencv_learn\opencv_material\after\after_onlymohu_m_1.jpg'
light_dst_path=r'D:\opencv_learn\opencv_material\after\after_onlymohu_m_1.jpg'
#
#
#light_1=get_facequality(light_1_path)
a=get_face(light_dst_path)
        
#def GetFileList(dir, fileList):
#    newDir = dir
#    if os.path.isfile(dir):
#        fileList.append(dir.encode('utf-8'))
#    elif os.path.isdir(dir):  
#        for s in os.listdir(dir):
#            #如果需要忽略某些文件夹，使用以下代码
#            #if s == "xxx":
#                #continue
#            newDir=os.path.join(dir,s)
#            GetFileList(newDir, fileList)  
#    return fileList
# 
#list = GetFileList(r'D:\opencv_learn\opencv_material\after', [])
#count=0
#for i in list:
#    i=str(i,encoding='utf-8')
#    i=r'{i}'.format(i=i)
#    print(i)
#    dst=get_facequality(i)
#    count+=1
#    print(count)


