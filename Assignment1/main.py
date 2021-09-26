""" This is the skeleton code for main.py
You need to complete the required functions. You may create addition source files and use them by importing here.
"""

import os
import cv2
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Get mIOU of video sequences')
    parser.add_argument('-i', '--inp_path', type=str, default='input', required=True, \
                                                        help="Path for the input images folder")
    parser.add_argument('-o', '--out_path', type=str, default='result', required=True, \
                                                        help="Path for the predicted masks folder")
    parser.add_argument('-c', '--category', type=str, default='b', required=True, \
                                                        help="Scene category. One of baseline, illumination, jitter, dynamic scenes, ptz (b/i/j/m/p)")
    parser.add_argument('-e', '--eval_frames', type=str, default='eval_frames.txt', required=True, \
                                                        help="Path to the eval_frames.txt file")
    args = parser.parse_args()
    return args


def baseline_bgs(args):
    fgbg = cv2.createBackgroundSubtractorMOG2()
    datadir = args.inp_path  #"./data/baseline/input/"
    x=os.listdir(datadir)
    x.sort()


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))  #[[1,1,1],[1,1,1],[1,1,1] ]
    
    eval_file = open(args.eval_frames + "eval_frames.txt")
    eval_list = list(eval_file.read().split(" "))
    eval_list = [int(n) for n in eval_list]
    
    cnt = 0
    fgmask = cv2.imread(datadir+x[0])
    for filename in x:
        i = cv2.imread(datadir+filename)
        frame = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
       
        frame = cv2.medianBlur(frame,5)

        if(cnt > 0):
            fgmask = fgbg.apply(image=frame)
        else:
            fgmask = fgbg.apply(image=frame)
        _,thresh = cv2.threshold(fgmask, 10, 255, cv2.THRESH_BINARY)

        
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel,iterations = 4 )
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel,iterations = 4)
        
        cnt+=1
        if(cnt >= eval_list[0]):
            cv2.imwrite(args.out_path + filename, opening)


def illumination_bgs(args):
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = False)

    datadir = args.inp_path  # "./data/illumination/input/"
    x=os.listdir(datadir)
    x.sort()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

    gt_dir = datadir + "../groundtruth/"
    gt_list = os.listdir(gt_dir)
    gt_img = cv2.imread(gt_dir + gt_list[0])
    gt_size = gt_img.shape

    eval_file = open(args.eval_frames + "eval_frames.txt")
    eval_list = list(eval_file.read().split(" "))
    eval_list = [int(n) for n in eval_list]

    cnt = 0
    fgmask = [[]]

    for filename in x: 
        frame = cv2.imread(datadir+filename)

        #-----Converting image to LAB Color model----------------------------------- 
        lab= cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        #-----Splitting the LAB image to different channels-------------------------
        l, a, b = cv2.split(lab)

        #-----Applying CLAHE to L-channel-------------------------------------------
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)


        #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
        limg = cv2.merge((cl,a,b))

        #-----Converting image from LAB Color model to RGB model--------------------
        frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if(cnt > 0):
            fgmask = fgbg.apply(image=frame,fgmask=fgmask,learningRate=.47)
        else:
            fgmask = fgbg.apply(image=frame,learningRate=.47)

        _,thresh = cv2.threshold(fgmask, 10, 255, cv2.THRESH_BINARY)

        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel,iterations = 2 )
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel,iterations = 4)
        
        output = opening

        cnt+=1
        if(cnt >= eval_list[0]):
            output = cv2.resize(output,(gt_size[1], gt_size[0]))
            cv2.imwrite(args.out_path+filename, output)
        


def jitter_bgs(args):
    #TODO complete this function
    pass


def dynamic_bgs(args):
    fgbg = cv2.createBackgroundSubtractorMOG2(history = 800)
    datadir = args.inp_path # "./data/moving_bg/input/"
    x=os.listdir(datadir)
    x.sort()

    kernel = np.ones((3,3), np.uint8)
    kernel_open = np.ones((3,3),np.uint8)
    cnt = 0

    eval_file = open(args.eval_frames + "eval_frames.txt")
    eval_list = list(eval_file.read().split(" "))
    eval_list = [int(n) for n in eval_list]


    for filename in x:
        i = cv2.imread(datadir+filename)
        gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        #gray = cv2.medianBlur(gray,5)


        fgmask = fgbg.apply(gray)

        _,thresh = cv2.threshold(fgmask, 10, 255, cv2.THRESH_BINARY)
    
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel,iterations = 3 )
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_open,iterations = 3)
        
        cnt+=1
        if(cnt >= eval_list[0]):
            cv2.imwrite(args.out_path + filename, opening)



def ptz_bgs(args):
    #TODO: (Optional) complete this function
    pass


def main(args):
    if args.category not in "bijmp":
        raise ValueError("category should be one of b/i/j/m/p - Found: %s"%args.category)
    FUNCTION_MAPPER = {
            "b": baseline_bgs,
            "i": illumination_bgs,
            "j": jitter_bgs,
            "m": dynamic_bgs,
            "p": ptz_bgs
        }

    FUNCTION_MAPPER[args.category](args)

if __name__ == "__main__":
    args = parse_args()
    main(args)
