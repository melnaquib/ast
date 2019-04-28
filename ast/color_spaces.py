


import matplotlib
matplotlib.use('Qt5Agg')

from matplotlib import pyplot as plt
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

img_LAB = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)


# img_HLS = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)


# img_HLS_H = img_HLS.copy()
# img_HLS_L = img_HLS.copy()
# img_HLS_S = img_HLS.copy()


# img_HLS_H[:,:1:2] = 0
# img_HLS_L[:,:0] = 0
# img_HLS_L[:,:2] = 0
# img_HLS_S[:,:0:1] = 0

img_LAB_L = img_LAB.copy()
img_LAB_A = img_LAB.copy()
img_LAB_B = img_LAB.copy()

img_LAB[:,:,0] = 0.5

img_LAB_L[:,:,1] = 0
img_LAB_L[:,:,2] = 0
img_LAB_A[:,:,2] = 0.0
img_LAB_A[:,:,0] = 0.0
img_LAB_B[:,:,0:2] = 0.0


########################################################################################################################

image__ = cv2.resize(img_LAB_B,(int(img_LAB_B.shape[0]/8),int(img_LAB_B.shape[1]/8)))

Result = np.zeros(image__.shape)

for i in range(image__.shape[0]):
    for j in range(image__.shape[1]):
        if image__[i,j,2] < 113:
            pass
        elif image__[i,j,2] < 126:
            Result[i, j, 0] = 255
        elif image__[i, j, 2] < 149:
            Result[i, j, 1] = 255
        else:
            Result[i, j, 2] = 255




########################################################################################################################

Z = img_LAB_B.reshape((-1,3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img_cy.shape))

cv2.imshow('res2',cv2.resize(res2,(512,512)))


cv2.imshow('img',cv2.resize(img,(512,512)))
cv2.imshow('img_cy',cv2.resize(img_cy,(512,512)))
cv2.imshow('img_LAB',cv2.resize(img_LAB,(512,512)))

cv2.imshow('img_LAB_A',cv2.resize(img_LAB_A,(512,512)))
cv2.imshow('img_LAB_B',cv2.resize(img_LAB_B,(512,512)))
cv2.imshow('img_LAB_L',cv2.resize(img_LAB_L,(512,512)))

cv2.imshow('Result',Result)


# cv2.imshow('img_HLS_H',cv2.resize(img_HLS_H,(512,512)))
# cv2.imshow('img_HLS_L',cv2.resize(img_HLS_L,(512,512)))
# cv2.imshow('img_HLS_S',cv2.resize(img_HLS_S,(512,512)))

k = cv2.waitKey()



plt.figure(1)
chan_A = img_LAB[:,:,1]
plt.hist(chan_A.ravel(),256,[0,256])
plt.title("channel A")


plt.figure(2)
chan_B = img_LAB[:,:,2]
plt.hist(chan_B.ravel(),256,[0,256])
plt.title("channel B")



plt.figure(3)
chan_B = img_LAB_L[:,:,0]
plt.hist(chan_B.ravel(),256,[0,256])
plt.title("channel L")

plt.show()


if k == 27:
    cv2.destroyAllWindows()








