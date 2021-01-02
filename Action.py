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
def Action(X,Y):
    thread1 = threading.Thread(target = Dianji_zhuan,
                                args = (X,IN1,IN2,IN3,c))
    thread2 = threading.Thread(target = Dianji_zhuan,
                                args = (Y,IN21,IN22,IN23,c2))
    thread1.start()
    thread2.start()
    setDirection(80)
    setDirection(140)
    setDirection(80)