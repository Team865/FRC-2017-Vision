import cv2
import numpy as np

file = 'output_gear_boiler.avi'
cap = cv2.VideoCapture(file)



cv2.namedWindow('video')
counter = 0

while(cap.isOpened()):
    
    open, frame = cap.read()


    cv2.imshow('video', frame)

    k = cv2.waitKey(1)
    if k%256 == 32:
	modfile = file.replace(' ', '')[:-4]
	filename = modfile + str(counter) + '.png'
        cv2.imwrite(filename, frame)
        counter += 1

    if k%256 == 27:
        break

cap.release()

cv2.destroyAllWindows()