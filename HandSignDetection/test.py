import math
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5","Model/labels.txt")

offset = 20
imgSize = 300

labels = ['A','B','C']

while True:
    success, img = cap.read()
    hands = detector.findHands(img, draw=False)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]

        aspectRatio = h/w

        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((300-wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize
            prediction, index = classifier.getPrediction(img)
            print(prediction,index)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((300 - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow("imageCrop", imgCrop)
        cv2.imshow("imageWhite", imgWhite)

    cv2.imshow("Image", img)
    cv2.waitKey(1)


