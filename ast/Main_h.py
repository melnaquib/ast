
import matplotlib
matplotlib.use('Qt5Agg')

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import utils
import cv2
import numpy as np
from PIL import Image, ImageDraw



# load the image and convert it from BGR to RGB so that
# we can dispaly it with matplotlib
image = cv2.imread("../test/data/IMG_20160523_095300.jpg",cv2.IMREAD_COLOR)
# img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img = image
output = image.copy()

# show our image

# plt.figure()
# plt.axis("off")
# plt.imshow(img)

# reshape the image to be a list of pixels
# img = img.reshape((img.shape[0] * img.shape[1], 3))
#
# # cluster the pixel intensities
#
# clt = KMeans(n_clusters = 3)
# clt.fit(img)
#
# # build a histogram of clusters and then create a figure
# # representing the number of pixels labeled to each color
# hist = utils.centroid_histogram(clt)
# bar = utils.plot_colors(hist, clt.cluster_centers_)
#
# # show our color bart
# plt.figure()
# plt.axis("off")
# plt.imshow(bar)
# plt.show()



# circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1.2, 100)
#
#
#
#
# if circles is not None:
#     # convert the (x, y) coordinates and radius of the circles to integers
#     circles = np.round(circles[0, :]).astype("int")
#
#     # loop over the (x, y) coordinates and radius of the circles
#     for (x, y, r) in circles:
#         # draw the circle in the output image, then draw a rectangle
#         # corresponding to the center of the circle
#         cv2.circle(output, (x, y), r, (0, 255, 0), 4)
#         cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
#
#     # show the output image
#
#     plt.figure()
#     plt.axis("off")
#     plt.imshow(np.hstack([image, output]))
#     plt.show()
#
    # cv2.imshow("output", np.hstack([image, output]))
    # cv2.waitKey(0)


mask = np.zeros(image.shape)
print(mask.shape)
cx , cy = 2365, 1695
r = 1370




for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
        if ((i - cy)**2 + (j - cx)**2) <= r**2:
            mask[i,j,:] = 1


plt.figure(1)
output1 = image*mask
cv2.circle(output1, (cx,cy), r, (0, 255, 0), 20)
plt.imshow(output1)
plt.title("output1")

plt.figure(2)
cv2.circle(output, (cx,cy), r, (0, 255, 0), 20)

plt.imshow(output)
plt.title("output")

image_gray = cv2.imread("IMG_20160523_095300.jpg",0)

plt.figure(3)
plt.imshow(image_gray)

plt.title("gray")





print(img.shape)
print(output1.shape)

img = output1.reshape((output1.shape[0] * output1.shape[1], 3))

# cluster the pixel intensities

clt = KMeans(n_clusters = 4)
clt.fit(img)


hist = utils.centroid_histogram(clt)
bar = utils.plot_colors(hist, clt.cluster_centers_)



# show our color bart

plt.figure(4)
plt.axis("off")
plt.imshow(bar)






plt.show()



# cv2.imshow("img",output)
# # cv2.imshow("img",img)
# k = cv2.waitKey()
# if k == 27:
#     cv2.destroyAllWindows()






