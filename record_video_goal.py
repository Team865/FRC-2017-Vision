#imports
import cv2
import numpy as np

cap = cv2.VideoCapture(1)

#output and the video name
fourcc = cv2.cv.FOURCC(*"MJPG")
out = cv2.VideoWriter('output_goal.avi', fourcc, 20.0, (640,480))


while (cap.isOpened()):
	#camera input
	ret, frame = cap.read()
	if ret:
		
		frame = cv2.flip(frame,0)
		
		#writing to the video filev 
		out.write(frame)

cap.release()
out.release()
