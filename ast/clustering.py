


import numpy as np
import cv2

img = cv2.imread('../test/data/IMG_20160523_095300.jpg')

img_cy = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
img_cy[:,:,0] = 0.5

# (channel_y, channel_u, channel_v) = cv2.split(img_cy)
# channel_y = 0.5
# channel_y = (np.ones(shape=channel_u.shape))*0.5
#
# im = cv2.equalizeHist(channel_y,channel_u,channel_v)

cv2.imshow('im',cv2.resize(img_cy,(512,512)))

Z = img.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 10
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img_cy.shape))

cv2.imshow('res2',cv2.resize(res2,(512,512)))
cv2.waitKey(0)
cv2.destroyAllWindows()



