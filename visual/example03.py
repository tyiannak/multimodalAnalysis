"""!
@brief Example 3
@details: Real-time color detection using a simple color thresholding
@author Theodoros Giannakopoulos {tyiannak@gmail.com}
"""
import cv2
import numpy as np
cap = cv2.VideoCapture(0)
while 1:
    _, frame = cap.read() # Get frames
    frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5) # resize
    h, w = frame.shape[0], frame.shape[1]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Convert BGR to HSV
    # define range of BLUE color in HSV
    lower_blue = np.array([105, 50, 50])
    upper_blue = np.array([135, 255, 255])
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    cv2.moveWindow('mask', 0, h)
    cv2.moveWindow('res', w, h)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()