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

oldError = 0

# hsv colour ranges
lower_green_hsv = np.array([162,122,197], dtype=np.uint8)
upper_green_hsv = np.array([180,255,255], dtype=np.uint8)

# rgb colour ranges 234, 0, 96
#lower_green_rgb = np.array([96, 0, 234], dtype=np.uint8)
#upper_green_rgb = np.array([124, 120, 255], dtype=np.uint8)
lower_green_rgb = np.array([25-0, 80-0, 0-0], dtype=np.uint8)
upper_green_rgb = np.array([180+0, 255+0, 70+0], dtype=np.uint8)

# set the camera
cam = cv2.VideoCapture(0)
while cv2.waitKey(1) != 1048603:
	# get image from camera
	ret_val, img = cam.read()
	
	# get the hsv colour image
	#hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV_FULL)
	#hsl = cv2.cvtColor(hsv,cv2.COLOR_BGR2HLS)
	mask_rgb = cv2.inRange(img, lower_green_rgb, upper_green_rgb)

	# dilated image
	#dilation = mask_rgb.copy()#cv2.dilate(mask, kernel, iterations=1)

	#res = cv2.bitwise_and(img, img, mask=mask)

	# blobs
	#keypoints = detector.detect(dilation)
	#if keypoints:
	#	print keypoints[0].pt
	
	# image showing the blobs
	#im_with_keypoints = cv2.drawKeypoints(dilation, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	contours,_ = cv2.findContours(mask_rgb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(img, contours, -1, (0, 0, 255), 1)

	targets = []
	targetContours = []
	for i in range(len(contours)):
		if cv2.contourArea(contours[i]) > 100:
			rect1 = cv2.boundingRect(contours[i])
			if(rect1[3] < 250):	
				a = cv2.contourArea(contours[i])
                        	hull = cv2.convexHull(contours[i])
                        	hull_area = cv2.contourArea(hull)
                        	solidity = float(a)/hull_area
				if(solidity > 0.7):
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
					if (abs((rect1[1]+rect1[3]) - (rect2[1]+rect2[3])) < 30):
                                            areaRatio = (rect1[2]*rect1[3])/(rect2[2]*rect2[3])
                                            if (areaRatio < 1.3) and (areaRatio > 0.7):
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
		t1 = map(float, selectedTargets[0])
		t2 = map(float, selectedTargets[1])

		distance = 1.0-((t1[1]+t2[1])/2.0)/480.0
		offset = ((((t1[0]+t1[2]/2.0) + (t2[0]+t2[2]/2.0))/2.0)-320.0)/320.0
		
		offset = offset*distance
	#PID CONTROL
		P = -1
		P_turn = 0.75
		D_turn = 4
		speed = P*distance
		diff = P_turn*offset + D_turn*(offset-oldError)
		oldError = offset
		sd.putNumber('D_dist', P*distance)
                sd.putNumber('D_offset',P_turn*offset)
	
		sd.putBoolean('D_found', True)
		sd.putNumber('D_left', speed - diff)
		sd.putNumber('D_right',speed + diff)
	else:
		oldError = 0
		print "f"
		sd.putNumber('D_left', 0)
                sd.putNumber('D_right',0)
		sd.putBoolean('D_found', False)

	# display regular web cam
	#cv2.imshow('regular', img)
	# display the processed image
	#cv2.imshow('processed', mask_rgb)

	
cv2.destroyAllWindows()
