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
    #print("Error: specify an IP to connect to!")
    exit(0)

ip = sys.argv[1]
'''while (not NetworkTables.isConnected()):
	NetworkTables.initialize(server=ip)
	sleep(3)
	pass
'''

while(True):
	NetworkTables.initialize(server=ip)
	sleep(1)	
	ad = NetworkTables.getTable("data/auto")
	
	#print(ad.getString("I AM HERE", defaultValue=""))
	if ad.getString("I AM HERE", defaultValue="") ==  "true":
		break	
	NetworkTables.shutdown()

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
upper_green_rgb = np.array([255+0, 255+0, 70+0], dtype=np.uint8)

lower_green_rgb1 = np.array([90-0, 230-0, 0-0], dtype=np.uint8)
upper_green_rgb1 = np.array([255+0, 255+0, 235+0], dtype=np.uint8)

oldError1 = 0
errorSum1 = 0
counter1 = 0


# set the camera
cam = cv2.VideoCapture(0)
cam1 = cv2.VideoCapture(1)
while cv2.waitKey(1) != 1048603:
	ret, img = cam.read()
        mask_rgb = cv2.inRange(img, lower_green_rgb, upper_green_rgb)

        contours,_ = cv2.findContours(mask_rgb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 0, 255), 1)

        targets = []
	selectedTargets = []
        selectedContours = []

        targetContours = []
        for i in range(len(contours)):
            if cv2.contourArea(contours[i]) > 100:
                rect1 = cv2.boundingRect(contours[i])
                if rect1[3] < 250:
                    a = cv2.contourArea(contours[i])
                    hull = cv2.convexHull(contours[i])
                    hull_area = cv2.contourArea(hull)
                    solidity = float(a)/hull_area
                    if solidity > 0.7:
                        targets.append(rect1)
                        targetContours.append(contours[i])
                        
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
		t1 = selectedTargets[0]
		t2 = selectedTargets[1]

		distance = float(((float(t1[1])+float(t1[3])+float(t2[1])+float(t2[3]))/2.0)/480.0)
		offset = float(((((float(t1[0])+float(t1[2])/2.0) + (float(t2[0])+float(t2[2])/2.0))/2.0)-320.0)/320.0)		
		offset = offset*distance	
	
		#PID CONTROL
		P = -1
		P_turn = 0.75
		D_turn = 4
		speed = P*distance
		diff = P_turn*offset + D_turn*(offset-oldError)
		oldError = offset
	
		sd.putBoolean('D_found', True)
		sd.putNumber('D_left', speed + diff)
		sd.putNumber('D_right',speed - diff)
	else:
		oldError = 0
		print "f"
		sd.putNumber('D_left', 0)
                sd.putNumber('D_right',0)
		sd.putBoolean('D_found', False)


#===============================================================
	
	# get image from camera
	ret_val1, img1 = cam1.read()

	mask_rgb1 = cv2.inRange(img1, lower_green_rgb1, upper_green_rgb1)

        contours1,_1 = cv2.findContours(mask_rgb1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img1, contours1, -1, (0, 0, 255), 1)

	targets1 = []
	selectedTargets1 = []
	selectedContours1 = []
	selectedTargets = []
	targetContours1 = []
	for i in range(len(contours1)):
		if cv2.contourArea(contours1[i]) > 100:
			rect1 = cv2.boundingRect(contours1[i])
			if (rect1[2] < 200) and (rect1[3] < 100):
                                if(rect1[2] > 40) and (rect1[3] > 7):
               			    targets1.append(rect1)
                       		    targetContours1.append(contours1[i])
	
	
	if targetContours1 and len(targetContours1)>=2:
		area1 = 0
		for a in range(len(targetContours1)):
			for b in range(len(targetContours1)):
				if a != b and cv2.contourArea(targetContours1[a]) + cv2.contourArea(targetContours1[b]) >= area1:
					rect1 = cv2.boundingRect(targetContours1[a])
					rect2 = cv2.boundingRect(targetContours1[b])
                                    	if (abs((rect1[0]+rect1[2]/2)-(rect2[0]+rect2[2]/2)) < 15):
                                                if abs((rect1[1]+rect1[3]/2)-(rect2[1]+rect2[3]/2)) < 125:
						    areaRatio = (rect1[2]*rect1[3])/(rect2[2]*rect2[3])
                                                    if (areaRatio < 2) and (areaRatio > 0.5):
                                                        selectedTargets1 = []
						        selectedContours1 = []
						        selectedTargets1.append(rect1)
						        selectedTargets1.append(rect2)
						        selectedContours1.append(targetContours1[a])
						        selectedContours1.append(targetContours1[b])
						        area1 = cv2.contourArea(targetContours1[a]) + cv2.contourArea(targetContours1[b])		


		cv2.drawContours(img1, targetContours1, -1, (255, 0, 0), 2)
		cv2.drawContours(img1, selectedContours1, -1, (0, 255, 0), 2)

	if selectedTargets1:
		print "t"
		#print selectedTargets1
		t1 = selectedTargets1[0]
		t2 = selectedTargets1[1]

		#offset = (320 - (0.5*(t1[0] + t2[0]) + 0.25*(t1[2] + t2[2])))/320
		offset = (((float(t1[0])+float(t2[0]))/2.0)+((float(t1[2])+float(t2[2]))/4.0)-320)/320.0
			
		#angle_error = 1.65 -  abs(t1[0] - t2[0])/((t1[3] + t2[3])*0.5)
		print offset
		angle_error = 0 #dd.getNumber('gyro_angle', 0)	
		#PID CONTROL
		# 0.875, 16, 6.5
		P_turn = 0.5
		D_turn = 10.0
		I_turn = 0.0
		
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
	
		zero = 0.04
		if diff > 0:
			zero = 0.04
		else:
			zero = -0.04

		sd.putBoolean('S_found', True)
		sd.putNumber('S_left', -zero+-diff)
		sd.putNumber('S_right', zero+diff)
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

	# display regular web cam
	#cv2.imshow('regular', img)
	# display the processed image
	#cv2.imshow('processed', mask_rgb)
	
cv2.destroyAllWindows()
