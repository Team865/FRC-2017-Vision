'''
Simply display the contents of the webcam with optional mirroring using OpenCV 
via the new Pythonic cv2 interface.  Press <esc> to quit.
'''

import cv2
import numpy as np
from networktables import NetworkTables
import sys
import logging
logging.basicConfig(level=logging.DEBUG)

if len(sys.argv) != 2:
    print(sys.argv)
    print("Error: specify an IP to connect to!")
    exit(0)

ip = sys.argv[1]
NetworkTables.initialize(server=ip)
sd = NetworkTables.getTable("data/vision")
dd = NetworkTables.getTable("data/Drive")

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
upper_green_hsv = np.array([180,255,255], dtype=np.uint8)

# rgb colour ranges 234, 0, 96
#lower_green_rgb = np.array([96, 0, 234], dtype=np.uint8)
#upper_green_rgb = np.array([124, 120, 255], dtype=np.uint8)
lower_green_rgb = np.array([103, 112, 60], dtype=np.uint8)
upper_green_rgb = np.array([181, 218, 198], dtype=np.uint8)


# set the camera
cam = cv2.VideoCapture(0)
while cv2.waitKey(1) != 1048603:
	# get image from camera
	ret_val, img = cam.read()
	
	# get the hsv colour image
	#hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV_FULL)
	#hsl = cv2.cvtColor(hsv,cv2.COLOR_BGR2HLS)

	# masked image
	#mask_hsl = cv2.inRange(hsl, lower_green_hsv, upper_green_hsv)
	#rgb = cv2.bitwise_and(hsv, hsv, mask=mask_hsl)
	mask_rgb = cv2.inRange(img, lower_green_rgb, upper_green_rgb)

	# dilated image
	dilation = mask_rgb.copy()#cv2.dilate(mask, kernel, iterations=1)

	#res = cv2.bitwise_and(img, img, mask=mask)

	# blobs
	#keypoints = detector.detect(dilation)
	#if keypoints:
	#	print keypoints[0].pt
	
	# image showing the blobs
	#im_with_keypoints = cv2.drawKeypoints(dilation, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	contours,_ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(img, contours, -1, (0, 0, 255), 1)

	targets = []
	targetContours = []
	for i in range(len(contours)):
		if cv2.contourArea(contours[i]) > 200:
			rect1 = cv2.boundingRect(contours[i])		
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

		distance = (0.5*(t1[1] + t2[1])+120)/480
		offset = (320 - 0.5*(t1[0] + t2[0]))/320
		
		offset = offset*distance		
		#angle_error = 1.65 -  abs(t1[0] - t2[0])/((t1[3] + t2[3])*0.5)
		#print abs(t1[0] - t2[0])/((t1[3] + t2[3])*0.5)
		angle_error = 0 #dd.getNumber('gyro_angle', 0)	
		#PID CONTROL
		P = -0.65
		P_turn = 0.3
		P_angle = 1
		speed = P*distance
		diff = P_turn*offset# + P_angle*angle_error
		sd.putNumber('distance', P*distance)
                sd.putNumber('offset',P_turn*offset)
                sd.putNumber('angle', P_angle*angle_error)
	
		sd.putBoolean('found', True)
		sd.putNumber('left', speed - diff)
		sd.putNumber('right',speed + diff)
	else:
		print "f"
		sd.putNumber('left', 0)
                sd.putNumber('right',0)
		sd.putBoolean('found', False)

	# display regular web cam
	#cv2.imshow('regular', img)
	# display the processed image
	#cv2.imshow('processed', mask_rgb)

	
cv2.destroyAllWindows()
