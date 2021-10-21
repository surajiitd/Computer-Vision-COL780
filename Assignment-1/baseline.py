
import cv2
import glob
import os
import time
import numpy as np
 
fgbg = cv2.createBackgroundSubtractorMOG2()
datadir = "./data/baseline/input/"
x=os.listdir(datadir)
x.sort()

r=[]
kernal = np.ones((3,3), np.uint8)
kernal_open = np.ones((3,3),np.uint8)
cnt = 0
fgmask = cv2.imread(datadir+x[0])
for filename in x:
    print(filename)
    r.append(filename)
    i = cv2.imread(datadir+filename)
    frame = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    #gray = cv2.GaussianBlur(gray, (21,21), 0)
    frame = cv2.medianBlur(frame,5)
    # fgmask = fgbg.apply(gray)
    if(cnt > 0):
        fgmask = fgbg.apply(image=frame)
    else:
        fgmask = fgbg.apply(image=frame)
    _,thresh = cv2.threshold(fgmask, 10, 255, cv2.THRESH_BINARY)
    #dilate = cv2.dilate(thresh, kernal_dilate, iterations=2)
    #erode = cv2.erode(dilate,kernal_erode,iterations=2)
    
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernal,iterations = 4 )
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernal_open,iterations = 4)
    
    cnt+=1
    if(cnt >= 470):
        cv2.imwrite("./output1/"+filename, opening)



