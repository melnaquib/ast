

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


image_name = "IMG_20160523_095452.jpg"
image_path = "../test/data/"+image_name

# load the input image and grab its dimensions
image = cv2.imread(image_path)

# image = cv2.resize(image,(int(image.shape[0]/4),int(image.shape[1]/4)))
image = cv2.resize(image,(int(image.shape[1]/6),int(image.shape[0]/6)))


image_rgb = cv2.imread(image_path,0)
image_rgb = cv2.resize(image_rgb,(int(image_rgb.shape[1]/6),int(image_rgb.shape[0]/6)))
image2 = image.copy()


# image_rgb = cv2.resize(image_rgb,(512,512))

(H, W) = image.shape[:2]

blurred1 = cv2.GaussianBlur(image, (3, 3), 0)

B_mean = np.mean(blurred1[:,:,0])
G_mean = np.mean(blurred1[:,:,1])
R_mean = np.mean(blurred1[:,:,2])

# print(B_mean,G_mean,R_mean)

blob_blur = cv2.dnn.blobFromImage(blurred1, scalefactor=1.0, size=(W, H),
                             mean=(B_mean,G_mean,R_mean),
                             swapRB=False, crop=False)


blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H),
                             mean=(B_mean,G_mean,R_mean),
                             swapRB=False, crop=False)

net.setInput(blob_blur)
hed_blur = net.forward()
hed_blur = cv2.resize(hed_blur[0, 0], (W, H))
hed_blur = (255 * hed_blur).astype("uint8")




net.setInput(blob)
hed = net.forward()
hed = cv2.resize(hed[0, 0], (W, H))
hed = (255 * hed).astype("uint8")


dl_canny = cv2.Canny(hed, 30, 150)

dl_canny1 = dl_canny.copy()

canny = cv2.Canny(image_rgb, 40, 80)


def get_dist(p1,p2):

    cx_n, cy_n = p2[0], p2[1]
    cy,cx = p1[0],p1[1]
    dist = ((cx - cx_n) ** 2 + (cy - cy_n) ** 2)
    dist.sort()
    return dist


# non_zero = np.nonzero(dl_canny)

print(image_rgb.shape)
circles = cv2.HoughCircles(canny,cv2.HOUGH_GRADIENT,1,500,
                            param1=100,param2=90,minRadius=100,maxRadius=2500)

# print(circles)
print(image2.shape)
for i in circles[0]:
    cv2.circle(dl_canny, (i[0], i[1]), i[2], (0, 0, 0), 80)

# print(non_zero)
# dist = get_dist(non_zero,(dl_canny.shape[0]/2,dl_canny.shape[1]/2))

# cv2.line(dl_canny,(0,256),(512,256),(255,255,255),3)

cv2.imshow('canny', cv2.resize(canny,(512,512)))
cv2.imshow('image2', cv2.resize(image2,(512,512)))
cv2.imshow('dl_canny', cv2.resize(dl_canny,(512,512)))
cv2.imshow('dl_canny1', cv2.resize(dl_canny1,(512,512)))
# cv2.imshow('hed_blur', cv2.resize(hed_blur,(512,512)))
# cv2.imshow('hed', cv2.resize(hed,(512,512)))


# cv2.imwrite("../test/data/test.jpg",image2)

k = cv2.waitKey()
if k == 27:
    cv2.destroyAllWindows()






