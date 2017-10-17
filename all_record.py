import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap1= cv2.VideoCapture(1)


fourcc = cv2.cv.CV_FOURCC(*'MJPG')

out = cv2.VideoWriter('/home/ubuntu/src/output_gear.avi', fourcc, 20.0, (640, 480))
out1 = cv2.VideoWriter('/home/ubuntu/src/output_goal.avi', fourcc, 20.0, (640, 480))

while(cap.isOpened() and cap1.isOpened()):
	ret, frame = cap.read()
	ret1, frame1 = cap1.read()
	
	if ret and ret1:
		frame = cv2.flip(frame,0)
		
		frame1 = cv2.flip(frame1, 0)				

		out.write(frame)
		
		out1.write(frame1)

cap.release()
out.release()

out1.release()
cap.release()
