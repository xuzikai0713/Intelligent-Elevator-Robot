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
def Dianji_zhuan(a,IN_1,IN_2,IN_3,c):
    b = int(a)*25
    if(b>c[0]):
        GPIO.output(IN_2, GPIO.LOW)
    else:
        GPIO.output(IN_2, GPIO.HIGH)
    GPIO.output(IN_3, GPIO.LOW)
    GPIO.output(IN_1, 0)
    for i in range(1,abs(c[0]-b)):
        GPIO.output(IN_1, 0)
        time.sleep(0.001)
        GPIO.output(IN_1, 1)
        time.sleep(0.001)
    c[0] = b