import cv2
import os
import numpy as np
path = "E:/li/hjl/data/image/"
names = os.listdir(path)
for name in names:
    kernel = np.ones((3,3),np.uint8)
    image = cv2.imread(path + name)
    name = name[:-4]
    img1 = cv2.erode(image,kernel)
    img2 = cv2.dilate(image,kernel) 
    img3 = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel) 
    img4 = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    cv2.imshow('123',img1)