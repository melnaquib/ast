


import cv2
import os
import numpy as np
import os

class CropLayer(object):
    def __init__(self, params, blobs):
        # initialize our starting and ending (x, y)-coordinates of
        # the crop
        self.startX = 0
        self.startY = 0
        self.endX = 0
        self.endY = 0

    def getMemoryShapes(self, inputs):
        # the crop layer will receive two inputs -- we need to crop
        # the first input blob to match the shape of the second one,
        # keeping the batch size and number of channels
        (inputShape, targetShape) = (inputs[0], inputs[1])
        (batchSize, numChannels) = (inputShape[0], inputShape[1])
        (H, W) = (targetShape[2], targetShape[3])

        # compute the starting and ending crop coordinates
        self.startX = int((inputShape[3] - targetShape[3]) / 2)
        self.startY = int((inputShape[2] - targetShape[2]) / 2)
        self.endX = self.startX + W
        self.endY = self.startY + H

        # return the shape of the volume (we'll perform the actual
        # crop during the forward pass
        return [[batchSize, numChannels, H, W]]

    def forward(self, inputs):
        # use the derived (x, y)-coordinates to perform the crop
        return [inputs[0][:, :, self.startY:self.endY,
                self.startX:self.endX]]


# load our serialized edge detector from disk

path = "holistically-nested-edge-detection/hed_model"

print("[INFO] loading edge detector...")
protoPath = os.path.sep.join([path,"deploy.prototxt"])
modelPath = os.path.sep.join([path,"hed_pretrained_bsds.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# register our new layer with the model
cv2.dnn_registerLayer("Crop", CropLayer)


image_name = "IMG_20160523_095300.jpg"
image_path = "../test/data/"+image_name

# load the input image and grab its dimensions
image = cv2.resize(cv2.imread(image_path),(512,512))
(H, W) = image.shape[:2]


imgc = cv2.cvtColor(image,cv2.COLOR_BGR2HLS)

image_sum = imgc[:,:,1] - imgc[:,:,2]

image_sum = cv2.cvtColor(image_sum,cv2.COLOR_GRAY2BGR)

img_LAB = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
b = img_LAB[:,:,2]

b = cv2.cvtColor(b,cv2.COLOR_GRAY2BGR)

img_yuv = cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
# img_cy[:,:,0] = 0.5


(channel_y, channel_u, channel_v) = cv2.split(img_yuv)

img_cy = cv2.equalizeHist(channel_v,channel_u)

cv2.imshow('image',image)
cv2.imshow('img_cy',img_cy)


k = cv2.waitKey()
if k == 27:
    cv2.destroyAllWindows()

# convert the image to grayscale, blur it, and perform Canny
# edge detection
print("[INFO] performing Canny edge detection...")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
canny = cv2.Canny(blurred, 30, 150)

blurred1 = cv2.GaussianBlur(image, (3, 3), 0)


kernel = np.ones((5,5),np.float32)/25
smoothed = cv2.filter2D(image,-1,kernel)


blur = cv2.bilateralFilter(image,9,75,75)



# construct a blob out of the input image for the Holistically-Nested
# Edge Detector
blob = cv2.dnn.blobFromImage(blurred1, scalefactor=1.0, size=(W, H),
                             mean=(104.00698793, 116.66876762, 122.67891434),
                             swapRB=False, crop=False)




blob_sum = cv2.dnn.blobFromImage(image_sum, scalefactor=1.0, size=(W, H),
                             mean=(104.00698793, 116.66876762, 122.67891434),
                             swapRB=False, crop=False)

blob_cy = cv2.dnn.blobFromImage(img_cy, scalefactor=1.0, size=(W, H),
                             mean=(104.00698793, 116.66876762, 122.67891434),
                             swapRB=False, crop=False)



blob_smoothed = cv2.dnn.blobFromImage(smoothed, scalefactor=1.0, size=(W, H),
                             mean=(104.00698793, 116.66876762, 122.67891434),
                             swapRB=False, crop=False)


blob_blur = cv2.dnn.blobFromImage(blur, scalefactor=1.0, size=(W, H),
                             mean=(104.00698793, 116.66876762, 122.67891434),
                             swapRB=False, crop=False)


# mean=(104.00698793, 116.66876762, 122.67891434)
# print(blob.shape)
# print(blob[0,1,:,:])

# cv2.circle(output1, (cx,cy), r, (0, 255, 0), 20)

# set the blob as the input to the network and perform a forward pass
# to compute the edges
print("[INFO] performing holistically-nested edge detection...")
net.setInput(blob)
hed = net.forward()
hed = cv2.resize(hed[0, 0], (W, H))
hed = (255 * hed).astype("uint8")


print("[INFO] performing holistically-nested edge detection...")
net.setInput(blob_sum)
hed_b = net.forward()
hed_b = cv2.resize(hed_b[0, 0], (W, H))
hed_b = (255 * hed_b).astype("uint8")


# net.setInput(blob_cy)
# hed_cy = net.forward()
# hed_cy = cv2.resize(hed_cy[0, 0], (W, H))
# hed_cy = (255 * hed_cy).astype("uint8")


net.setInput(blob_smoothed)
hed_smoothed = net.forward()
hed_smoothed = cv2.resize(hed_smoothed[0, 0], (W, H))
hed_smoothed = (255 * hed_smoothed).astype("uint8")






net.setInput(blob_blur)
hed_blur = net.forward()
hed_blur = cv2.resize(hed_blur[0, 0], (W, H))
hed_blur = (255 * hed_blur).astype("uint8")

# print(hed.shape)
# print(hed)



DL_canny = cv2.Canny(hed, 30, 150)


kernel = np.array([[-1,-1,-1],
                   [-1, 9,-1],
                   [-1,-1,-1]])
sharpened = cv2.filter2D(hed, -1, kernel)







##############################################################
# houghman

anti_circles = cv2.HoughCircles(DL_canny,cv2.HOUGH_GRADIENT,1,20,
                            param1=80,param2=30,minRadius=0,maxRadius=35)


cimg = cv2.cvtColor(hed,cv2.COLOR_GRAY2BGR)
cimg2 = cv2.cvtColor(hed,cv2.COLOR_GRAY2BGR)

centers_for_anti = []
for i in anti_circles[0]:
    centers_for_anti.append((i[0],i[1]))
    cv2.circle(cimg, (i[0],i[1]),2,(0,0,255),3)
    cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
    # cv2.circle(cimg2, (i[0], i[1]), 2, (0, 0, 255), 3)



circles = cv2.HoughCircles(DL_canny,cv2.HOUGH_GRADIENT,1,20,
                            param1=60,param2=30,minRadius=35,maxRadius=150)


def dist(p1,p2):
    d = []
    for p in p2:
        d.append(((p1[0] - p[0])**2 + (p1[1] - p[1])**2)**0.5)
    return min(d),p2[d.index(min(d))]


circles = np.uint16(np.around(circles))
distances = {}
for i in circles[0,:]:
    # draw the outer circle
    p = (i[0],i[1])

    d,p2 = dist(p, centers_for_anti)
    if d <= 15:
        cv2.circle(cimg2, p, i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(cimg2, p, 2, (0, 0, 255), 3)
    if p2 in distances:
        if d < distances[p2][0]:
            distances[p2] = (d,i[2])
    else:
        distances[p2] = (d,i[2])


print(distances)


for key in distances:

    if distances[key][0] <= 15:
        cv2.circle(cimg,key,distances[key][1],(0,255,0),2)
        cv2.circle(image, key, distances[key][1], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(cimg,key,2,(0,0,255),3)
        cv2.circle(image, key, 2, (0, 0, 255), 3)


######################################################################





# show the output edge detection results for Canny and
# Holistically-Nested Edge Detection
cv2.imshow("Input", image)
cv2.imshow("Canny", canny)
cv2.imshow("HED", hed)
cv2.imshow("DL_canny", DL_canny)
cv2.imshow("sharpened", sharpened)
cv2.imshow('detected circles',cimg)
cv2.imshow('detected circles222',cimg2)

cv2.imshow('hed_blur',hed_blur)
cv2.imshow('hed_smoothed',hed_smoothed)


cv2.imshow('blur',blur)
cv2.imshow('smoothed',smoothed)


cv2.imshow('hed_b',hed_b)
# cv2.imshow('hed_cy',hed_cy)


# cv2.imwrite("output/smoothed_"+image_path,smoothed)


# cv2.imwrite("output/"+image_name,image)
#
# cv2.imwrite("output/DL_"+image_name,cimg)
#
# cv2.imwrite("output/DL_canny_"+image_name,DL_canny)
#
# cv2.imwrite("output/HED_canny_"+image_name,hed)


k = cv2.waitKey()
if k == 27:
    cv2.destroyAllWindows()





