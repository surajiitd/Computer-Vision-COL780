
import cv2
import glob
import os
import time
import numpy as np
 
# fgbg = cv2.createBackgroundSubtractorMOG2()
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = False)

datadir = "./data/illumination/input/"
x=os.listdir(datadir)
x.sort()

kernel = np.ones((3,3), np.uint8)
kernel_open = np.ones((3,3),np.uint8)

gt_dir = datadir+ "../groundtruth/"
gt_list = os.listdir(gt_dir)
gt_img = cv2.imread(gt_dir + gt_list[0])
gt_size = gt_img.shape
cnt = 0
fgmask = gt_img
for filename in x:
    print(filename)
    # r.append(filename)        
    frame = cv2.imread(datadir+filename)

    #-----Converting image to LAB Color model----------------------------------- 
    lab= cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    # cv2.imshow("lab",lab)

    #-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)
    # cv2.imshow('l_channel', l)
    # cv2.imshow('a_channel', a)
    # cv2.imshow('b_channel', b)

    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    # cv2.imshow('CLAHE output', cl)


    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))
    # cv2.imshow('limg', limg)

    #-----Converting image from LAB Color model to RGB model--------------------
    frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


    ############END

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame = cv2.GaussianBlur(frame, (5,5), 0)
    # frame = cv2.medianBlur(frame,5)
    # frame = cv2.equalizeHist(frame)
    
    if(cnt > 0):
        fgmask = fgbg.apply(image=frame,fgmask=fgmask,learningRate=.47)
    else:
        fgmask = fgbg.apply(image=frame,learningRate=.47)
    # fgmask2 = fgbg2.apply(gray)
    # fgmask = cv2.medianBlur(fgmask,3)
    _,thresh = cv2.threshold(fgmask, 10, 255, cv2.THRESH_BINARY)
    # _,thresh2 = cv2.threshold(fgmask2, 200, 255, cv2.THRESH_BINARY)    #dilate = cv2.dilate(thresh, kernel_dilate, iterations=2)
    #erode = cv2.erode(dilate,kernel_erode,iterations=2)

    # gradient = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel,iterations = 1)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel,iterations = 2 )
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_open,iterations = 4)
    

    # closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel,iterations = 6 )
    
    output = opening

    cnt+=1
    if(cnt >= 57):
        output = cv2.resize(output,(gt_size[1], gt_size[0]))
        cv2.imwrite("./output2/"+filename, output)
    
