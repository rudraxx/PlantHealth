# Script to send emails from python
#https://stackoverflow.com/questions/13070038/attachment-image-to-send-by-mail-using-python

import smtplib

# Import the email modules you will need
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart


def SendImagebyMail(ImgFile):

	img_data = open(ImgFile)
# Open plain text file containing only ascii 

'''
with open(textfile) as fp:

	# Create message
	msg = MIMEText(fp.read())
'''

msg = MIMEText("HelloFarm")

# me== senders email
# you == recipient's email address

msg['Subject'] = 'The content of message' #%s' % textfile
msg['From'] = 'sdrip03@gmail.com'
msg['To']   = 'rudraxx@gmail.com'

# Send the message via our smtp server

s = smtplib.SMTP('localhost')

print s

s.quit()


