# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 16:30:25 2017

@author: AbhishekBhat
"""

import cv2
#import matplotlib.pyplot as plt


cam= cv2.VideoCapture(0)

s,image = cam.read()

# Convert image from brg2rgb
image2 = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#plt.imshow(image2)

cv2.imwrite('/home/pi/Desktop/test_usb.jpg',image)
cam.release()