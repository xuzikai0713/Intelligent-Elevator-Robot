import subprocess
import time
import requests
import base64
import cv2   
import numpy as np
import copy
import random
import math
import RPi.GPIO as GPIO
import time
import threading
import setup
import setDirection
import take_photo
import Dianji_zhuan
import mathc_img
import Cut
import brightness1
import Analyze
import Action
import BrightAnallyze
from PIL import Image  
from PIL import ImageStat
P_SERVO = 22 
fPWM = 50  
a = 10
b = 2
IN1 = 11
IN2 = 13  
IN3 = 15
IN21 = 12
IN22 = 16 
IN23 = 18
c=[0]
c2=[0]
def mathc_img(image,Target,value):
    img_rgb = cv2.imread(image) #读取拍摄图片
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY) #转化为灰度图
    template = cv2.imread(Target,0) #读取模板图片
    w, h = template.shape[::-1] #提取模板图片的高和宽
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED) #模板匹配
    threshold = value #设置阈值
    loc = np.where( res >= threshold) #逐像素匹配
    
    print(w)
    print(h)
    print(loc[0][0])
    
    print(loc[1][0])
    print(w)
    print(h)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (7,249,151), 2) #在图中标注匹配成功的图片（用作调试时直观比对）
    #cv2.resizeWindow('findCorners', 20, 30) 
    #cv2.imshow('Detected',img_rgb)
    #cv2.waitKey(0)
    cv2.imwrite('detected6-test4.jpg',img_rgb)
    #cv2.destroyAllWindows()
    print('done4')
    return loc[0][0],loc[1][0],w,h #返回坐标
