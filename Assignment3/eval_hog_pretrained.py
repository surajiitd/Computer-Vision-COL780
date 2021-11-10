"""
command to execute:

val: python3 eval_hog_pretrained.py --root "./pedestrian_detection_dataset" --test "./pedestrian_detection_dataset/PennFudanPed_val.json" --out "./jsons/val_predict_hog_pretrained.json"

detections:
val: python3 eval_detections.py --pred "./jsons/val_predict_hog_pretrained.json" --gt "./pedestrian_detection_dataset/PennFudanPed_val.json"
"""

from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import copy
import os
import json


def new_coordinates(original_size, new_size, original_coordinate):
    #here we are getting input in the x1,y1 and x2,y2 format
    scale = np.flipud(np.divide(new_size, original_size))
    new_top_left_corner = np.divide(original_coordinate, scale )
    #print("there ")
    #print(new_top_left_corner)
    return new_top_left_corner 

def getScaledBbox(original_size, new_size, original_coordinate):
    scale = np.flipud(np.divide(new_size, original_size))
    x,y,w,h = original_coordinate
    topLeft = (x,y)
    bottomRight = (x + w, y + h)
    newTopLeft = np.divide(topLeft,scale)
    newbottomRight = np.divide(bottomRight,scale)
    finalBbox = [newTopLeft[0],newTopLeft[1],newbottomRight[0]-newTopLeft[0],newbottomRight[1]-newTopLeft[1]]
    return finalBbox
    
   
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-r", "--root", type=str, required=True, help="path to the dataset root directory")
ap.add_argument("-t", "--test", type=str, required=True, help="path to test json")
ap.add_argument("-o", "--out", type=str, required=True, help="path to output json")
args = vars(ap.parse_args())

imgFolderPath = os.path.join(args['root'],"PennFudanPed/PNGImages")      
generatedOutputFilePath = "./output/hog"   
generatedOutputJsonPath = args['out']  
validationJsonPath = args['test']  

imgFiles = os.listdir(imgFolderPath)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

f = open(validationJsonPath)

data = json.load(f)

valData=[]
for i in data['images']:
    imageName = i['file_name'].split('/')[2]
    valData.append(imageName)

outputArray=[]
imgid = 0
for imagePath in valData:
    
    cocoDicttemp={
        "image_id":-1,
        "category_id":1,
        "bbox":[-1,-1,-1,-1],
        "score":-1
        }
    
    print(imagePath)

    image = cv2.imread(imgFolderPath+"/"+imagePath)
    fullimg = image.copy()
    originalShape = image.shape[:2]
    
    if image.shape[1] > 400: 
        image = cv2.resize(image,(400,300))
    
    newShape = image.shape[:2]
  
    orig = image.copy()
    

    (rects, weights) = hog.detectMultiScale(image, winStride=(4,4),padding=(8, 8), scale=1.05)
    
    modRects=[]
    for (x, y, w, h) in rects:
        modRects.append([x, y, x + w, y + h])
    modRects = np.array(modRects)
    
    pick = non_max_suppression(modRects, probs=None, overlapThresh=0.65)
    
    if(len(pick)!=0):
        nPick = pick.tolist()
    else:
        nPick = []
        
    if(len(modRects)!=0):
        nmodRects = modRects.tolist()
    else:
        nmodRects = []
    
    for r in range(len(nmodRects)):
        if nmodRects[r] in nPick:
            #print(rects[r])
            cocodict = copy.deepcopy(cocoDicttemp)
            cocodict["image_id"] = imgid
            cocodict["bbox"] = getScaledBbox(originalShape, newShape, rects[r].tolist())
            cocodict["score"] = weights[r][0]
            outputArray.append(cocodict)
            
        
    for (xA, yA, xB, yB) in pick:
        
        xAn,yAn = new_coordinates(originalShape, newShape, (xA,yA))
        xBn,yBn = new_coordinates(originalShape, newShape, (xB,yB))
        cv2.rectangle(fullimg, (int(xAn),int(yAn)), (int(xBn),int(yBn)), (0, 255, 0), 2)

    cv2.imwrite(generatedOutputFilePath+"/"+imagePath, fullimg)
    imgid+=1

print(len(outputArray))    
with open(generatedOutputJsonPath, "w") as output:
    output.write(json.dumps(outputArray))