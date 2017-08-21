#!/home/pi/.virtualenvs/cv/bin/python3.4

from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import time

with PiCamera() as camera:
	# Set the awb mode off
	camera.awb_mode = 'off'

	# Set image resolution
	camera.resolution = (1024,768)
	
	awb_values = [0.5,0.6,0.7,0.8 ,0.9, 1.0 , 1.1,1.2,1.3,1.4,1.5]

	for i in awb_values:
		for j in awb_values:
			my_gains = (i,j)
			# Set awb gains
			camera.awb_gains = my_gains
			print(camera.awb_gains)
	
			# Annotate text - using camera values to confirm setting
			camera.annotate_text=str(camera.awb_gains)
	
			print('Capturing image with awb gains = ',i,' , ',j, '\n' )
			camera.start_preview()
			time.sleep(1)
			filename = './awb_images/out_' + str(i) + '_' + str(j) + '.jpg'
		
			camera.capture(filename)
 


