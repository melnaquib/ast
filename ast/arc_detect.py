

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt

import numpy as np
import cv2

inputImage = cv2.imread("../test/data/IMG_20160523_095300.jpg")
inputImage = cv2.resize(inputImage,(512,512))
inputImageGray = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(inputImageGray,150,200,apertureSize = 3)
minLineLength = 5
maxLineGap = 1
lines = cv2.HoughLinesP(edges,cv2.HOUGH_PROBABILISTIC, np.pi/180, 30, minLineLength,maxLineGap)
for x in range(0, len(lines)):
    for x1,y1,x2,y2 in lines[x]:
        #cv2.line(inputImage,(x1,y1),(x2,y2),(0,128,0),2, cv2.LINE_AA)
        pts = np.array([[x1, y1 ], [x2 , y2]], np.int32)
        cv2.polylines(inputImage, [pts], True, (0,255,0))

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(inputImage,"Tracks Detected", (500, 250), font, 0.5, 255)
cv2.imshow("Trolley_Problem_Result", inputImage)
cv2.imshow('edge', edges)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
