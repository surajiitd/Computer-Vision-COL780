
import cv2
import glob
import os
import time
import numpy as np
 
fgbg = cv2.createBackgroundSubtractorMOG2(history = 800)
datadir = "./data/moving_bg/input/"
x=os.listdir(datadir)
x.sort()

r=[]
kernel = np.ones((3,3), np.uint8)
kernel_open = np.ones((3,3),np.uint8)
cnt = 0
for filename in x:
    print(filename)
    r.append(filename)
    i = cv2.imread(datadir+filename)
    gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    #gray = cv2.medianBlur(gray,5)


    fgmask = fgbg.apply(gray)



    _,thresh = cv2.threshold(fgmask, 10, 255, cv2.THRESH_BINARY)
    #dilate = cv2.dilate(thresh, kernel_dilate, iterations=2)
    #erode = cv2.erode(dilate,kernel_erode,iterations=2)
    # gradient = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel,iterations = 1)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel,iterations = 3 )
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_open,iterations = 3)
    
    cnt+=1
    if(cnt >= 800):
        cv2.imwrite("./output4/"+filename, opening)

