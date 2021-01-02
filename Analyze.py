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
def Analyze(a):   
    floor =a
    img = "123.jpg"
    Target = ('target%s'%(floor)+'.jpg')
    value = 0.8
    print('done')
    x,y,w,h = mathc_img(img,Target,value)
    print(x)
    print(y)
    print(w)
    print(h)
    img = '123.jpg'
    Cut(x,y,w,h,img)
    bright = brightness1('crop.jpg')
    print(bright)
    M=(557-x)/6.66,N=(1010-y)/6.66
    return M,N,bright 