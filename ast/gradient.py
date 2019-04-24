
import matplotlib
matplotlib.use('Qt5Agg')


import numpy as np
import cv2
from matplotlib import pyplot as plt


frame = cv2.resize(cv2.imread("IMG_20160523_095200.jpg"),(512,512))

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

lower_red = np.array([30, 120, 50])
upper_red = np.array([255, 255, 180])

mask = cv2.inRange(hsv, lower_red, upper_red)
res = cv2.bitwise_and(frame, frame, mask=mask)

laplacian = cv2.Laplacian(frame, cv2.CV_64F)
sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)

cv2.imshow('Original', frame)
cv2.imshow('Mask', mask)
cv2.imshow('laplacian', laplacian)
cv2.imshow('sobelx', sobelx)
cv2.imshow('sobely', sobely)

k = cv2.waitKey()
if k == 27:
    cv2.destroyAllWindows()


