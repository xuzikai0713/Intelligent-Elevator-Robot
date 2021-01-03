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
def Cut(x,y,w,h,im):    
    img = im
    img = cv2.imread(img) #读取图片
    #cv2.imshow('original', img)
    #x = 331
    #y = 217
    #w = 210
    #h = 194
    roi = (x,y,w,h)
    print(roi)
    if roi != (0, 0, 0, 0):
        crop = img[x:x+w, y:y+h] #裁剪图片
        #cv2.imshow('crop', crop)
        cv2.imwrite('crop.jpg', crop) #写入图片
        print('Saved!')
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()