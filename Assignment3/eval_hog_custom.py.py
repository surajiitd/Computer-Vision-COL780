"""
command to execute:

val: python3 eval_hog_custom.py.py --root "./pedestrian_detection_dataset" --test "./pedestrian_detection_dataset/PennFudanPed_val.json" --out "./jsons/val_predict_hog_custom.json" --model "./trained_hog_svm_model.p"

detections:
val: python3 eval_detections.py --pred "./jsons/val_predict_hog_custom.json" --gt "./pedestrian_detection_dataset/PennFudanPed_val.json"
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
from sklearn.svm import LinearSVC
from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from skimage import color
import pickle
import sys



def new_coordinates(original_size, new_size, original_coordinate):
    #here we are getting input in the x1,y1 and x2,y2 format
    scale = np.flipud(np.divide(new_size, original_size))
    new_top_left_corner = np.divide(original_coordinate, scale )
    return new_top_left_corner 

def getScaledBbox(original_size, new_size, original_coordinate):
    #here we are getting input in the x1,y1 and x2,y2 format
    scale = np.flipud(np.divide(new_size, original_size))
    x1,y1,x2,y2 = original_coordinate
    topLeft = (x1,y1)
    bottomRight = (x2, y2)
    newTopLeft = np.divide(topLeft,scale)
    newbottomRight = np.divide(bottomRight,scale)
    finalBbox = [newTopLeft[0],newTopLeft[1],newbottomRight[0]-newTopLeft[0],newbottomRight[1]-newTopLeft[1]]
    return finalBbox

def transform(img):
    transformedImage = cv2.resize(img,(64,128))
    hogFeature = hog(transformedImage,orientations=9,pixels_per_cell=(8,8),visualize=False,cells_per_block=(3,3))
    return hogFeature

def makePrediction(img,model):
    bins=9
    cellsize=(8,8)
    blocks=(3,3)
    feature = hog(img, orientations=bins,pixels_per_cell=cellsize,visualize=False,cells_per_block=blocks)
    feature = feature.reshape(1, -1)
    pred = model.predict(feature)
    return pred,feature

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-r", "--root", type=str, required=True, help="path to the dataset root directory")
ap.add_argument("-t", "--test", type=str, required=True, help="path to test json")
ap.add_argument("-o", "--out", type=str, required=True, help="path to output json")
ap.add_argument("-m", "--model", type=str, required=True, help="path to trained SVM model")

args = vars(ap.parse_args())

imgFolderPath = os.path.join(args['root'],"PennFudanPed/PNGImages")
generatedOutputJsonPath = args['out']  #"D:\\IIT DELHI CLASSES\\COMPUTER VISION\\Assignments\\Assignment 3\\pedestrian_detection"
positiveImagePath = "./positive_images_train"  #D:\\IIT DELHI CLASSES\\COMPUTER VISION\\Assignments\\Assignment 3\\positiveSamplesTrain"
negativeImagePath = "./negative_images_train"  #"D:\\IIT DELHI CLASSES\\COMPUTER VISION\\Assignments\\Assignment 3\\negativeSamplesNewTrain"
generatedOutputFilePath = "./output/hog_svm"  #"D:\\IIT DELHI CLASSES\\COMPUTER VISION\\Assignments\\Assignment 3\\generatedOutputHogSvm"
validationJsonPath = args['test']  #'D:\\IIT DELHI CLASSES\\COMPUTER VISION\\Assignments\\Assignment 3\\pedestrian_detection\\PennFudanPed_val.json'
trainedModelPath = args['model']

print("Do you want to build a model?")
print("yes - if the model has to be built(do load the data in the same directory)")
print("no - otherwise")

decision = input()
if(decision == "yes"):
    positiveImgFiles = os.listdir(positiveImagePath)
    negativeImgFiles = os.listdir(negativeImagePath)
    trainDataInput=[]
    trainDataOutput=[]
    
    print("Selecting Positive examples from the folder")
    for posFile in positiveImgFiles:
        #print(posFile)
        positiveImg = cv2.imread(positiveImagePath+"/"+posFile,0)
        hogFeature = transform(positiveImg)
        trainDataInput.append(hogFeature)
        trainDataOutput.append(1)
    
    print("Selecting Negative examples from the folder")    
    for negFile in negativeImgFiles:
        #print(negFile)
        negativeImage = cv2.imread(negativeImagePath+"/"+negFile,0)
        hogFeature  = transform(negativeImage)
        trainDataInput.append(hogFeature)
        trainDataOutput.append(0)
        
    trainDataInput = np.float64(trainDataInput)
    trainDataOutput = np.array(trainDataOutput)
    
    print('Total length of the train data is :',len(trainDataInput))
    print("SVM Model is being trained")
    model = LinearSVC()
    model.fit(trainDataInput,trainDataOutput)
    modelFileName = trainedModelPath
    pickle.dump(model, open(modelFileName, 'wb'))
    #pickle.dumps(model)

elif decision == "no":
    model = pickle.load(open(trainedModelPath, 'rb'))

else:
    print("invalid selection")
    sys.exit()

f = open(validationJsonPath)

data = json.load(f)

valData=[]
for i in data['images']:
    imageName = i['file_name'].split('/')[2]
    valData.append(imageName)

outputArray=[]
imgid = 0

for imgName in valData:
    
    cocoDicttemp={
        "image_id":-1,
        "category_id":1,
        "bbox":[-1,-1,-1,-1],
        "score":-1
        }
    
    print(imgName)
    
    image = cv2.imread(imgFolderPath+"/"+imgName)
    
    fullimg = image.copy()
    cropping = image.copy()
    originalShape = fullimg.shape[:2]
    
    image = cv2.resize(image,(400,256))
    
    newShape = image.shape[:2]
    
    size = (64,128)
    stride = (9,9)
    downscale = 1.25
    detections = []
    scale = 0
    
    gaussianPyramids = pyramid_gaussian(image,downscale = 1.25)
    
    for scaled in gaussianPyramids:
        
        if scaled.shape[0] < size[1]:
            break
        
        if scaled.shape[1] < size[0]:
            break
        
        imcopy = scaled.copy()

        for y in range(0, imcopy.shape[0], stride[1]):
            for x in range(0, imcopy.shape[1], stride[0]):
                window = imcopy[y: y + size[1], x: x + size[0]]
            
                if window.shape[0] != size[1]:
                    continue
                
                if window.shape[1] != size[0]:
                    continue
                
                window = color.rgb2gray(window)
    
                pred,fd = makePrediction(window,model)
                
                if pred == 1:
                    
                    confidence = model.decision_function(fd)   
                    if confidence > 0.5:
                        
                        xN = int(x * (pow(downscale,scale)))
                        yN = int(y * (pow(downscale,scale)))
                        s = confidence
                        wN = int(size[0] * (pow(downscale,scale)))
                        hN = int(size[1] * (pow(downscale,scale)))
                        detections.append([xN,yN,s,wN,hN])
     
        scale += 1
        
        
    rects=[]
    sc=[]
    for (x, y,s, w, h) in detections:
        rects.append([x, y, x + w, y + h])
        sc.append(s[0])
    
    rects = np.array(rects)
    sc = np.array(sc)
    
    pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.3)
    objCount = 0 
    
    npick = pick.tolist()
    nrects = rects.tolist()

    for r in range(len(detections)):
        if nrects[r] in npick:
            
            cocodict = copy.deepcopy(cocoDicttemp)
            cocodict["image_id"] = imgid
            cocodict["bbox"] = getScaledBbox(originalShape, newShape, rects[r].tolist())
            cocodict["score"] = sc[r]
            outputArray.append(cocodict)
            
    for(x1, y1, x2, y2) in pick:
        objCount+=1
        xAn,yAn = new_coordinates(originalShape, newShape, (x1,y1))
        xBn,yBn = new_coordinates(originalShape, newShape, (x2,y2))
        cv2.rectangle(fullimg, (int(xAn),int(yAn)), (int(xBn),int(yBn)), (0, 0, 255), 2)
    
    cv2.imwrite(generatedOutputFilePath+"/"+imgName, fullimg)
    imgid+=1

      
with open(generatedOutputJsonPath, "w") as output:
    output.write(json.dumps(outputArray))