import cv2
import numpy as np 
import math
import os
import copy
import time

def findParams(xPoints,yPoints,width,height):
    x, y = np.meshgrid(xPoints, yPoints)
    o = np.full((width,height), 1)  
    z = np.full((width,height), 0)  
    firstRow = np.stack((x, z, y, z, o, z), axis=2)
    secondRow = np.stack((z, x, z, y, z, o), axis=2)
    return (firstRow,secondRow)
    
def findAffineJacobian(height,width):
    xPoints = np.array(range(height))
    yPoints = np.array(range(width))
    jacobian = np.stack(findParams(xPoints,yPoints,width,height), axis=2)
    return jacobian


def getWarp(p):
    return np.array([[1+p[0], p[2], p[4]], [p[1], 1+p[3], p[5]]])
    

def normalLucasKanade(image, template, region, p, threshold,maximumIteration = 100):
    
    template = template[region[0][1]:region[1][1], region[0][0]:region[1][0]]
    rows, cols = template.shape
    previousP = p
    warp_mat = getWarp(previousP)
    points=np.array([[5,5,1],[5,10,1],[10,10,1],[10,5,1]])
    transform_prev = np.matmul(warp_mat,points.transpose())
    deltaPNorm = 99999
    
    it = 0
    while (deltaPNorm >= threshold):
        xGradient = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        yGradient = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
        
        warp = getWarp(previousP)            
        warpedImage = cv2.warpAffine(image,warp, (image.shape[1],image.shape[0]),flags=cv2.INTER_CUBIC)
        warpedImage = warpedImage[region[0][1]:region[1][1], region[0][0]:region[1][0]]
            
        error = template.astype(int) - warpedImage.astype(int)
        
        warpXGradient = cv2.warpAffine(xGradient, warp, (image.shape[1],image.shape[0]),flags=cv2.INTER_CUBIC+cv2.WARP_INVERSE_MAP)
        warpXGradient = warpXGradient[region[0][1]:region[1][1], region[0][0]:region[1][0]]
        warpYGradient = cv2.warpAffine(yGradient, warp, (image.shape[1],image.shape[0]),flags=cv2.INTER_CUBIC+cv2.WARP_INVERSE_MAP)
        warpYGradient = warpYGradient[region[0][1]:region[1][1], region[0][0]:region[1][0]]


        affineJacobian = findAffineJacobian(cols, rows)

        
        gradient = np.stack((warpXGradient, warpYGradient), axis=2)
        gradient = np.expand_dims((gradient), axis=2)
        
        descent = np.matmul(gradient, affineJacobian)
        order = (0, 1, 3, 2)
        hessian = np.matmul(np.transpose(descent, order),descent)
        hessian = hessian.sum((0,1))

        error = error.reshape((rows, cols, 1, 1))
        update = (np.transpose(descent, order) * error)
        update = update.sum((0,1))

        deltaP = np.matmul(np.linalg.pinv(hessian), update).reshape((-1))
        
        previousP += deltaP
     
        
        if iter == 1:
            first_p = p
            
        deltaPNorm = np.linalg.norm(deltaP)
        it += 1
        if it > maximumIteration:
            print("hi")
            return previousP
        
    return previousP

def pyramidLucasKanade(image,template,region,layers,threshold,p_prev):
    
    region = (region * (math.pow(0.5,layers) )).astype(int)
    
    for i in range(layers):
        image = cv2.pyrDown(image)
        template = cv2.pyrDown(template)
    
    p = np.zeros(6)

    
    for i in range(layers+1):
        p = normalLucasKanade(image,template,region,p,threshold)
        image = cv2.pyrUp(image)
        template = cv2.pyrUp(template)
        region = (region*2).astype(int)
    return p

imageFiles=os.listdir("../data/Bolt/img/")
imageFiles.sort()

template = cv2.imread("../data/Bolt/img/"+imageFiles[0])
height, width, _ = template.shape
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template_original = copy.deepcopy(template)
region = np.array([[336, 165], [362,226]])   # Bolt
topLeft = np.array([region[0][0], region[0][1], 1])
bottomRight = np.array([region[1][0], region[1][1], 1])


threshold = 0.05
layers = 1
p= np.zeros(6)
file1 = open('pred_bolt.txt', 'w')

for img in range(1,len(imageFiles)):
    print("Processing on frame : ",img+1)
    image = cv2.imread("../data/Bolt/img/"+imageFiles[img])
    image_copy = copy.deepcopy(image)
    image_copy_2 = copy.deepcopy(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    p_prev = p
    #p = normalLucasKanade(image, template, roi, p_prev ,threshold)
    p = pyramidLucasKanade(image,template,region,layers,threshold,p_prev)

    w = getWarp(p)  
    
    topLeftNew = (w @ topLeft).astype(int)
    bottomRightNew = (w @ bottomRight).astype(int)
  
    cv2.rectangle(image_copy, tuple(topLeftNew), tuple(bottomRightNew), (0, 0, 255), 1)
    
    cv2.imshow('Tracked Image', image_copy)
    cv2.waitKey(1)
    cv2.imwrite( "./lk_predicted/bolt/" +imageFiles[img], image_copy)
    s = str(topLeftNew[0])+","+str(topLeftNew[1])+","+str(abs(topLeftNew[0]-bottomRightNew[0]))+","+str(abs(topLeftNew[1]-bottomRightNew[1]))
    file1.write(s+'\n')
    

file1.close()  
cv2.destroyAllWindows()