# our custom transformer to detect only skin
import matplotlib.pyplot as plt
import cv2
import numpy as np

def custom_skin_detector(img):
    img = np.array(img)
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #skin color range for h in hsv
    HSV_mask = cv2.inRange(img_HSV[:,:,0], np.array((0)), np.array((17)))
    HSV_mask = cv2.morphologyEx(HSV_mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))

    #converting from gbr to YCbCr color space
    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    #putting all values of y to 0
    img_YCrCb[:,:,0] = 0

    #skin color range for ycrcb color space 
    YCrCb_mask = cv2.inRange(img_YCrCb, np.array((0, 135, 85)), np.array((255,180,135))) 
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    #global mask made from YCrCb mask and hsv mask
    global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
    global_mask=cv2.medianBlur(global_mask,3)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))
    global_img = cv2.bitwise_and(img,img,mask=global_mask)
    global_img = cv2.cvtColor(global_img,cv2.COLOR_BGR2RGB)
    return global_img