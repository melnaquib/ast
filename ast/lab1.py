


import matplotlib
matplotlib.use('Qt5Agg')

from matplotlib import pyplot as plt
import numpy as np
import cv2

img = cv2.imread('../test/data/IMG_20160523_095533.jpg')

scale = 0.125
img = cv2.resize(img, (int(img.shape[0] * scale), int(img.shape[1] * scale)))

# imgc = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
# imgc = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
imgc = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)


image_sum = imgc[:,:,1] - imgc[:,:,2]


Result = np.zeros((image_sum.shape[0],image_sum.shape[1],3))

for i in range(image_sum.shape[0]):
    for j in range(image_sum.shape[1]):
        if image_sum[i,j] < 110:
            Result[i, j, 0] = 255

        else:
            Result[i, j, 2] = 255



# blurred = cv2.GaussianBlur(imgc[:,:,2], (5, 5), 0)
#
#
# kernel = np.array([[-1,-1,-1],
#                    [-1, 9,-1],
#                    [-1,-1,-1]])
# sharpened = cv2.filter2D(imgc[:,:,2], -1, kernel)
#
# kernel = np.ones((5,5),np.float32)/25
# smoothed = cv2.filter2D(sharpened,-1,kernel)


# blurred = np.log10(imgc[:,:,2])
# print(blurred)

cv2.imshow(" ", img)
cv2.imshow("0", imgc[:,:,0])
cv2.imshow("1", imgc[:,:,1])
cv2.imshow("2", imgc[:,:,2])
cv2.imshow("image_sum", image_sum)
cv2.imshow(" Result ", Result)

# cv2.imshow(" blurred ", blurred)
# cv2.imshow(" sharpened ", sharpened)
# cv2.imshow(" smoothed ", smoothed)


plt.figure(3)
c = image_sum
plt.hist(c.ravel(),256,[0,256])
plt.title("channel L")

plt.show()




cv2.waitKey()
