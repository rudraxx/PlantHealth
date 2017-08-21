#!/home/pi/.virtualenvs/cv/bin/python3.4




### /usr/bin/env python

# Imports for PiCamera stuff and ndvi calculation
from picamera.array import PiRGBArray
from picamera import PiCamera

import time
import datetime

import cv2
import os
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np


# Imports for email smtp 
import smtplib

from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart



class SmartDrip:
    def __init__(self,controller_id):
        self.id = controller_id
        self.from_address = 'sdrip03@gmail.com'
        self.password = 'detroit2016'
        self.images_folder = "/home/pi/Desktop/testImages/"
        self.bucket_location = 's3://orig-rgb-images'
        self.width = 1640#3280#1280
        self.height = 922#2464#720

        

    def capture_image(self):
        print('\n Capturing Image from Rpi camera...\n')
        # init the camera
        camera = PiCamera()
        camera.resolution = (self.width,self.height)
        rawCapture =PiRGBArray(camera)
                
        try:
            # Allow camera to warm up
            time.sleep(2)

            # Grab an image
            # Note: the camera resolution is provided as width,height to rpi.
            # But cv2 capture output is height,width.
            camera.capture(rawCapture,format='bgr')
            image_bgr = rawCapture.array

        finally:
            camera.close()

        print(' \nCapture_image method complete. \n')
        return image_bgr

    def calculate_ndvi(self,image_bgr):
        print('\nCalculating NDVI from image... \n')
	        
        # Create a copy of the image
        img = np.copy(image_bgr)
        
        # Split image into components
        b,g,nir = cv2.split(img)
        
        # Calculate denominator
        den = (nir.astype(np.float) + b.astype(np.float))
        den[den==0]=0.001 # Preventing divide by zero
        
        # Calculate numerator        
        num = nir.astype(np.float) - b.astype(np.float)        
        
        # Calculate the ndvi of the image:        
        ndvi =  (num / den)                
        ndvi[ndvi<0]=0
        ndvi[ndvi>1]=0
        ndvi = (ndvi*255).round().astype(np.uint8)
        
        print('\n NDVI complete. \n')
        return ndvi
    
    def ndvi_scaling(self, ndvi_image):
        print('\nScaling NDVI image to 0-255 and changing datatype to UINT8.\n')
        ndvi2 = np.copy(ndvi_image)
        
        # Ensure the ndvi is a uint8 image with 0-255 values
        # If any value is > 255, that probably means it was a divide by 0.
#        ndvi2[ndvi2<0]=0
#        ndvi2[ndvi2>1]=0
#        ndvi2 = (ndvi2*255).round().astype(np.uint8)
        ndvi3 = cv2.applyColorMap(ndvi2,cv2.COLORMAP_JET)
        print('\n NDVI scaling complete... \n')
        return ndvi3 

    
    def histogram_image(self,image):
    
        im2 = np.copy(image)        
        im_height,im_width,channels = im2.shape
    #    color = ['b','g','r']
        color = [ (255,0,0) , (0,255,0), (0,0,255)]

        num_bins = 256
        bins = np.arange(0,num_bins,1,dtype=np.uint8).reshape(num_bins,1)
    
        '''
        Create empty image to print histogram data.
        the first value is high, as a lot of pixels are present in the image.
        If this is kept low, main part of the histogram gets
        flattened out after normalizing .
        '''
        h = 255*np.ones((10000,num_bins,3))
    
        for ch, col in enumerate(color):
            hist_item = cv2.calcHist([im2],[ch],None,[num_bins],[0,num_bins])
#            hist_item = cv2.normalize(hist_item,0,255,cv2.NORM_MINMAX)
            hist = (np.round(hist_item)).astype(np.int32)
            pts  = np.column_stack((bins,hist))
            cv2.polylines(h,[pts],False,col,1,8)
    
#        print('\n im2 histocopy shape: \n',im2.shape[0:2])
        h=np.flipud(h)
        # NOTE: resize needs (cols,rows), while shape gives (rows,cols)
        h = cv2.resize(h,(im_width,im_height))
#        print('\n h image shape: \n',h.shape)
        
        return h    
        
        
    def display_combined(self,orig_image=None,im2=None,im3=None,hist_image=None,ndvi_image=None,ndvi_scaled=None):
        '''
        Combine 6 images for display purpose
        '''
#        print('im1  shape: ',orig_image.shape)
#        print('im4 shape: ',hist_image.shape)
#    
        height,width,cc = orig_image.shape
#        # Resize te 4th image.
#        im5 = cv2.resize(hist_image,(width,height))
#        print('im5 shape: ',im5.shape)
    #    combined = np.zeros(2*height,2*width,3,).astype(np.uint8)

        combined = np.zeros((2*height,3*width,3),dtype=np.uint8)
        
        combined[0:height,0:width]        = orig_image
        combined[0:height,width:2*width]  = cv2.cvtColor(im2,cv2.COLOR_GRAY2BGR)
        combined[0:height,2*width:3*width] = cv2.cvtColor(ndvi_image,cv2.COLOR_GRAY2BGR)
    
        combined[height:2*height,0:width] = cv2.cvtColor(im3,cv2.COLOR_GRAY2BGR)        
    #    combined[height:2*height,width:2*width]  = cv2.cvtColor(im4,cv2.COLOR_GRAY2BGR)
        combined[height:2*height,width:2*width]  = hist_image
        combined[height:2*height,2*width:3*width]  = ndvi_scaled    
        print('Combining image done..')

        return combined
    
    def label_image(self,image,text,text_color=(0,0,0)):
        '''
        label the image
        '''
        return cv2.putText(image,text,(0,70),cv2.FONT_HERSHEY_SIMPLEX,3,text_color,3)
    
# Saving and Sending data methods.
    def save_images_to_rpi(self,image_bgr,ndvi=np.empty(1)):
        
        print('\n Saving Images. \n')
        full_image_path = ""
        file_name = datetime.datetime.now().strftime("%Y-%m-%d-%H.%M.%S.png")
        full_image_path = self.images_folder + file_name
	
        # Save the file to the disk
        cv2.imwrite(full_image_path,image_bgr)        
        ndvi_file_path = self.images_folder + "ndvi_"+ file_name 
#        if (ndvi.shape[0]>1):
#            ## Convert image from brg2rgb for plotting
#            image_rgb = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)
#            nn_nir,nn_green,nn_blue = cv2.split(image_rgb)

##            ndvi = self.calculate_ndvi(image_brg)
#            # Convert to 0-255.
#            ndvi2 = ndvi*255.0
#            # Set negative values to 0
#            ndvi2[ndvi2<0]=0.0
#                        
#            # Threshold the image after converting to uint8. cv2 needs images to be uint8
#            ndvi3 = np.uint8(ndvi2)
#            ret,ndvi4 = cv2.threshold(ndvi3,19,255,cv2.THRESH_BINARY)
#
##            plt.figure(1,figsize=(6,6))
##            plt.imshow(ndvi3)
##            plt.savefig(ndvi_file_path)
#
#
###            cv2.imwrite(ndvi_file_path,ndvi3)
##            cv2.imwrite(ndvi_file_path,cv2.applyColorMap(ndvi3,cv2.COLORMAP_JET))
#            
#            # Set up the figure
#            fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(16,10))
#            plot_images = [image_rgb,nn_nir,nn_blue,ndvi]
#            titles = ['original','nir','blue','ndvi']
#            for idx,ax in enumerate(axes.flat):
#                im = ax.imshow(plot_images[idx])#, vmin=0, vmax=1)
#                ax.set_title(titles[idx])
##            
##            fig.subplots_adjust(right=0.8)
##            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
##            fig.colorbar(im, cax=cbar_ax)
##            
##            #plt.show()
#            fig.suptitle('NDVI Image', fontsize=14, fontweight='bold')
#            fig.savefig(ndvi_file_path)
#                
##            cv2.imwrite(ndvi_file_path,self.calculate_ndvi(image_brg) * 255)
#            cv2.imwrite(ndvi_file_path,ndvi)
        
        print('\n Files saved. \n')    
        return ndvi_file_path         
        
        
    def uploadToAWS_S3_bucket(self,full_image_path):
        # upload file to aws s3 bucket using s3cmd
        cmd  = '/usr/bin/s3cmd put FILE' + ' ' +full_image_path + ' ' + self.bucket_location
        os.system(cmd)
        print('s3cmd command {} executed'.format(cmd))
    
        
    def send_image_email(self,to_address,msgBody,imageFileName):
        
        print('\n Sending email ... \n')
        # Set up the msg headers            
        msg = MIMEMultipart()
        msg['Subject']  = 'Farm image from controller_' + str(self.id)
        msg['From']     = self.from_address
        msg['To']       = to_address
        
        # Attach the message text
        msg.attach(MIMEText(msgBody))
        
        # Attach the image to send
        img_data = open(imageFileName,'rb').read()
        image = MIMEImage(img_data,name = imageFileName)
        msg.attach(image)
        
        # Set up the smtp server        
        s = smtplib.SMTP(host='smtp.gmail.com',port=587)
        s.starttls()
        s.login(self.from_address,self.password)
        s.sendmail(from_addr= self.from_address,
                   to_addrs= to_address,
                   msg=msg.as_string())
        s.quit()
        print('\n Email sending complete... \n')
        
        
# Main function
send_address = 'rudraxx@gmail.com'
mufi =  SmartDrip(11)

# Step 1: Capture image
raw_image = mufi.capture_image()
print('\n Raw_image shape : \n' , raw_image.shape)

#image_path = '/home/pi/Desktop/testImages/cam3.jpg'
#image_path = 'cam4.jpg'
#raw_image = cv2.imread(image_path)
##print(raw_image.shape)

#Step 2: Calculate NDVI
ndvi_image = mufi.calculate_ndvi(raw_image)
print('\n NDVI image shape: \n' , ndvi_image.shape)

#Step 2: Calculate scaled NDVI image
ndvi_scaled = mufi.ndvi_scaling(ndvi_image)
print('\n NDVI scaled image shape: \n' , ndvi_scaled.shape)

# Step 3: Save original and ndvi image
ndvi_file_path = mufi.save_images_to_rpi(raw_image,ndvi_scaled)
print('\nSaving NDVI image to : \n' + ndvi_file_path)

# Step 4: Get the histogram of the raw image 
hist_image = mufi.histogram_image(raw_image)
print('\n Histogram image shape : \n' , hist_image.shape)

# Step 5: # Create a composite image
b,g,r = cv2.split(raw_image)

# Name the images    
mufi.label_image(raw_image,'Original',(255,0,0))
mufi.label_image(b,'Blue',(255,0,0))
mufi.label_image(r,'NIR',(255,0,0))
mufi.label_image(hist_image ,'Histogram')
mufi.label_image(ndvi_image,'NDVI',(255,255,255))
mufi.label_image(ndvi_scaled,'NDVI-scaled',(255,255,0))

combined = mufi.display_combined(raw_image,b,r,hist_image,ndvi_image,ndvi_scaled)

# Save combined image
cv2.imwrite(ndvi_file_path,combined)



#Step 6: Send ndvi to gmail
mufi.send_image_email(send_address,"Sending ndvi image",ndvi_file_path)

#SendMail('farm.png','rudraxx@gmail.com')
    
#if __name__ == '__main__':
#    # Fill info...
#    usr='example@sender.ex'
#    psw='password'
#    fromaddr= usr
#    toaddr='example@recevier.ex'
#    SendAnEmail( usr, psw, fromaddr, toaddr)        
        
#'''
#19th July 2017 - Abhishek Bhat
#Class and main file to do various tasks needed for smartdrip
#
#1) Good Tutorial on using simple message transfer protocol (SMTP):
#    http://naelshiab.com/tutorial-send-email-python/
#
#
#'''
