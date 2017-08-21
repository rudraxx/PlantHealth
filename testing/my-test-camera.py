#!/home/pi/.virtualenvs/cv/bin/python3.4


### /usr/bin/env python

from picamera.array import PiRGBArray
from picamera import PiCamera

import time
import datetime

import cv2

import os


#import platform
#print('Python Version:',platform.python_version())
bucket_location = 's3://orig-rgb-images'


# init the camera
camera = PiCamera()
rawCapture =PiRGBArray(camera)


try:
	# Allow camera to warm up
	time.sleep(2)

	# Grab an image
	camera.capture(rawCapture,format='bgr')
	image = rawCapture.array
	#rgbImage = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

	file_name = datetime.datetime.now().strftime("%Y-%m-%d-%H.%M.%S.jpg")
	full_image_path = "/home/pi/Desktop/testImages/"+file_name
	
	# Save the file to the disk
	cv2.imwrite(full_image_path,image)

	# upload file to aws s3 bucket using s3cmd
	cmd  = '/usr/bin/s3cmd put FILE' + ' ' +full_image_path + ' ' + bucket_location
	os.system(cmd)
	print('s3cmd command {} executed'.format(cmd))

finally:
	camera.close() 
