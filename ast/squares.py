

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


image_name = "IMG_20160523_095227.jpg"
image_path = "../test/data/"+image_name

# load the input image and grab its dimensions
image = cv2.imread(image_path)

image = cv2.resize(image,(int(image.shape[1]/6),int(image.shape[0]/6)))

# original_image = image.copy()

image_rgb = cv2.imread(image_path,0)


image_rgb = cv2.resize(image_rgb,(int(image_rgb.shape[1]/6),int(image_rgb.shape[0]/6)))

img_LAB = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
img_LAB[:,:,0:2] = 0.0
check_image = img_LAB

canny = cv2.Canny(image_rgb, 40, 80)


(H, W) = image.shape[:2]

blur = cv2.bilateralFilter(image,9,75,75)


blurred1 = cv2.GaussianBlur(image, (3, 3), 0)




blob_blur = cv2.dnn.blobFromImage(blur, scalefactor=1.0, size=(W, H),
                             mean=(104.00698793, 116.66876762, 122.67891434),
                             swapRB=False, crop=False)



blob = cv2.dnn.blobFromImage(blurred1, scalefactor=1.0, size=(W, H),
                             mean=(104.00698793, 116.66876762, 122.67891434),
                             swapRB=False, crop=False)



net.setInput(blob)
hed = net.forward()
hed = cv2.resize(hed[0, 0], (W, H))
hed = (255 * hed).astype("uint8")



net.setInput(blob_blur)
hed_blur = net.forward()
hed_blur = cv2.resize(hed_blur[0, 0], (W, H))
hed_blur = (255 * hed_blur).astype("uint8")



dl_canny = cv2.Canny(hed, 0, 50)



def remove_the_plate(image_to_detect,image_to_remove,thickness = 60):

    plate = cv2.HoughCircles(image_to_detect,cv2.HOUGH_GRADIENT,1,500,
                            param1=100,param2=90,minRadius=100,maxRadius=2500)

    for i in plate[0]:
        cv2.circle(image_to_remove, (i[0], i[1]), i[2], (0, 0, 0), thickness=thickness)

    return image_to_remove

dl_canny = remove_the_plate(canny,dl_canny)


pill_circles = cv2.HoughCircles(hed_blur,cv2.HOUGH_GRADIENT,1,40,
                            param1=60,param2=30,minRadius=0,maxRadius=20)




def get_pills(pill_circles,image,draw_pills = True):

    pill_centers = []
    for i in pill_circles[0]:
        pill_centers.append((int(i[0]), int(i[1]),int(i[2])))
        if draw_pills:
            cv2.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 3)
            cv2.circle(image, (i[0], i[1]),2,(0,255,0),3)

    return pill_centers


pill_centers = get_pills(pill_circles,image)


def get_distances(pill_centers):

    distances = {}
    count = 0
    for pill in pill_centers:
        d = []
        # print(pill)
        cx, cy, r = pill[0],pill[1],pill[2]
        for pill1 in pill_centers:
            cx_n, cy_n , _ = pill1[0],pill1[1],pill1[2]
            temp_D = (((cx - cx_n) ** 2 + (cy - cy_n) ** 2) ** 0.5)

            if temp_D == 0:
                temp_D = 1000
            d.append(temp_D)
        distances[count] = ((cx,cy),pill_centers[d.index(min(d))],int((min(d))),r)
        count+=1
    return distances


distances = get_distances(pill_centers)


def badding_pills(img,center,r,thickness=20):
    cx, cy = center
    c1 = (cx - (r+thickness), cy - (r+thickness))
    c2 = (cx + r+thickness, cy + r+thickness)
    img[c1[1]:c2[1],c1[0]:c2[0]] = 0
    return img


def get_square(center,d,img,r):
    cx,cy = center
    c1 = (cx-d,cy-d)
    c2 = (cx+d,cy+d)
    img = badding_pills(img, center, r)
    return img[c1[1]:c2[1], c1[0]:c2[0]]






def get_dist(p1,p2):

    cx_n, cy_n = p2[0], p2[1]
    cy,cx = p1[0],p1[1]
    dist = ((cx - cx_n) ** 2 + (cy - cy_n) ** 2)
    dist.sort()
    return dist



def get_the_wanted_circule(distances,image,check_image,sensitivity = 60):

    for i in range(len(distances)):
        center = distances[i][0]
        img = get_square(center, distances[i][2], dl_canny, distances[i][3])

        non_zero = np.nonzero(img)

        # dist,distance = get_dist(non_zero,(img.shape[0]/2,img.shape[1]/2))


        dist = get_dist(non_zero,(img.shape[0]/2,img.shape[1]/2))


        if len(dist) > sensitivity:

            d = int((dist[sensitivity])**0.5)
            check_dist = d - 10
            cx, cy = center
            stx = cx - check_dist
            endx = cx - 20
            sty = cy - check_dist
            endy = cy + check_dist
            src = check_image[sty:endy,stx:endx,2]
            # cv2.rectangle(image,(sty,endy),(stx,endx),(0,255,255),3)
            m = np.mean(src)
            print(m)
            if m < 136:
                cv2.circle(image, distances[i][0],d,(0,255,0),3)

print(distances)
get_the_wanted_circule(distances,image,check_image)




cv2.imshow('image',image)
cv2.imshow('hed_blur',hed_blur)
cv2.imshow('hed',hed)
cv2.imshow('DL_canny', dl_canny)
cv2.imshow('check_image', check_image)

k = cv2.waitKey()
if k == 27:
    cv2.destroyAllWindows()



