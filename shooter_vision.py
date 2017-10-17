'''
Simply display the contents of the webcam with optional mirroring using OpenCV 
via the new Pythonic cv2 interface.  Press <esc> to quit.
'''

import cv2
import numpy as np
from networktables import NetworkTables
import sys
import logging
from time import sleep
logging.basicConfig(level=logging.DEBUG)

if len(sys.argv) != 2:
    print("Error: specify an IP to connect to!")
    exit(0)

ip = sys.argv[1]
'''while (not NetworkTables.isConnected()):
	NetworkTables.initialize(server=ip)
	sleep(3)
	pass
'''
NetworkTables.initialize(server=ip)


sd = NetworkTables.getTable("data/vision")
#dd = NetworkTables.getTable("data/Drive")

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
 
# Filter by Area.
params.filterByArea = True
params.minArea = 200
 
# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.1
 
# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.87
 
# Filter by Inertia
params.filterByInertia = False
params.maxInertiaRatio = 0.2

detector = cv2.SimpleBlobDetector(params)

kernel = np.ones((5, 5), np.uint8)

# hsv colour ranges
lower_green_hsv = np.array([162,122,197], dtype=np.uint8)
upper_green_hsv = np.array([255,255,180], dtype=np.uint8)

# rgb colour ranges 234, 0, 96
#lower_green_rgb = np.array([96, 0, 234], dtype=np.uint8)
#upper_green_rgb = np.array([124, 120, 255], dtype=np.uint8)
lower_green_rgb = np.array([30-0, 110-0, 0-0], dtype=np.uint8)
upper_green_rgb = np.array([255+0, 255+0, 80+0], dtype=np.uint8)


oldError1 = 0
errorSum1 = 0
counter1 = 0

# set the camera
cam = cv2.VideoCapture(0)
while cv2.waitKey(1) != 1048603:
	# get image from camera
	ret, img = cam.read()
        mask_rgb = cv2.inRange(img, lower_green_rgb, upper_green_rgb)

        contours,_ = cv2.findContours(mask_rgb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 0, 255), 1)

        targets = []
	selectedTargets = []
        targetContours = []
        for i in range(len(contours)):
            if cv2.contourArea(contours[i]) > 100:
                rect1 = cv2.boundingRect(contours[i])
                if rect1[3] < 250:
                    targets.append(rect1)
                    targetContours.append(contours[i])
                        
                    selectedTargets = []
                    selectedContours = []
                    if targetContours and len(targetContours)>=2:
                        area = 0
                        for a in range(len(targetContours)):
                            for b in range(len(targetContours)):
                                if a != b and cv2.contourArea(targetContours[a]) + cv2.contourArea(targetContours[b]) >= area:
                                    rect1 = cv2.boundingRect(targetContours[a])
                                    rect2 = cv2.boundingRect(targetContours[b])
                                    if (abs((rect1[0]+rect1[2]/2)-(rect2[0]+rect2[2]/2)) < 30):
                                        selectedTargets = []
                                        selectedContours = []
                                        selectedTargets.append(rect1)
                                        selectedTargets.append(rect2)
                                        selectedContours.append(targetContours[a])
                                        selectedContours.append(targetContours[b])
                                        area = cv2.contourArea(targetContours[a]) + cv2.contourArea(targetContours[b])		


                        cv2.drawContours(img, targetContours, -1, (255, 0, 0), 2)
                        cv2.drawContours(img, selectedContours, -1, (0, 255, 0), 2)

	if selectedTargets:
		print "t"
		print selectedTargets
		t1 = selectedTargets[0]
		t2 = selectedTargets[1]
		print t1[1]-t2[1]

		offset = (320 - (0.5*(t1[0] + t2[0]) + 0.25*(t1[2] + t2[2])))/320
		
			
		#angle_error = 1.65 -  abs(t1[0] - t2[0])/((t1[3] + t2[3])*0.5)
		#print abs(t1[0] - t2[0])/((t1[3] + t2[3])*0.5)
		angle_error = 0 #dd.getNumber('gyro_angle', 0)	
		#PID CONTROL
		# 0.875, 16, 6.5
		P_turn = 0.5
		D_turn = 12
		I_turn = 2.5
		
		diff = (P_turn*offset) + ((offset-oldError1)*D_turn/320) + I_turn*errorSum1/320
		oldError1 = offset
		
		counter1 += 1
		errorSum1 += offset
		if counter1 >= 1000:
			errorSum1 = 0
			counter1 = 0

		if diff > 0.4:
			diff = 0.4
		if diff < -0.4:
			diff = -0.4		
	
		zero = 0.1
		if diff > 0:
			zero = 0.1
		else:
			zero = -0.1

		print t1[1]+t2[1]*0.5

		sd.putBoolean('S_found', True)
		sd.putNumber('S_left', zero+diff)
		sd.putNumber('S_right', -zero+-diff)
		sd.putNumber('S_dist', t1[1]+t2[1]*0.5)
	else:
		errorSum1 = 0
		print "f"
		sd.putNumber('S_left', 0)
                sd.putNumber('S_right',0)
		sd.putBoolean('S_found', False)

	# display regular web cam
	#cv2.imshow('regular', img1)
	# display the processed image
	#cv2.imshow('processed', mask_rgb1)

	
cv2.destroyAllWindows()
