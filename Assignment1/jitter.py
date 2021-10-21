import cv2
import os
import numpy as np

def lineSmooth(trajectory):
    """function to smoothen the curve for x,y cordinate and angle coordinate"""
    smoothTrajectory = np.copy(trajectory)
    kernalLength = 101
    for i in range(3):
        line = np.copy(smoothTrajectory[:,i])
        kernal = np.ones(kernalLength)/kernalLength
        padded = np.lib.pad(line, (kernalLength//2, kernalLength//2), 'edge')
        smoothedLine = np.convolve(padded,kernal, mode='same')
        smoothedLine = smoothedLine[kernalLength//2:-(kernalLength//2)]
        smoothTrajectory[:,i] = smoothedLine
    return smoothTrajectory


datadir = "./data/jitter/input/"
imageDirectory =os.listdir(datadir)
imageDirectory.sort()
previousFrame = cv2.imread(datadir+imageDirectory[0])
previousFrameGray = cv2.cvtColor(previousFrame, cv2.COLOR_BGR2GRAY)
frameTransforms = np.zeros((len(imageDirectory), 3), np.float32)

for i in range(1,len(imageDirectory)):
    # Using Harris Corner detector to determine feature points and track it using differential point tracking
    previousFramePoints = cv2.goodFeaturesToTrack(previousFrameGray,maxCorners=200,qualityLevel=0.01,minDistance=30,blockSize=3)
    currentFrame = cv2.imread(datadir+imageDirectory[i])
    currentFrameGray = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2GRAY)
    currentFramePoints, status, err = cv2.calcOpticalFlowPyrLK(previousFrameGray, currentFrameGray, previousFramePoints, None)
    points = np.where(status==1)
    points = points[0]
    previousFramePoints = previousFramePoints[points]
    currentFramePoints = currentFramePoints[points]

    motion = cv2.estimateAffinePartial2D(previousFramePoints, currentFramePoints)[0]
    dx = motion[0,2]
    dy = motion[1,2]

    da = np.arctan2(motion[1,0],motion[0,0])
    frameTransforms[i-1] = [dx,dy,da]
    previousFrameGray = currentFrameGray
    print("Stabilizing Frame: " + str(i) + " of " + str(len(imageDirectory)))

trajectory = np.cumsum(frameTransforms, axis=0)

smoothTrajectory = lineSmooth(trajectory)
difference = smoothTrajectory - trajectory
SmoothTransforms = frameTransforms + difference

fgbg = cv2.createBackgroundSubtractorMOG2()
kernal = np.ones((3,3), np.uint8)
count = 0

for i in range(len(imageDirectory)):
    frame = cv2.imread(datadir+imageDirectory[i])
    # performing geometric transformations
    dx = SmoothTransforms[i,0]
    dy = SmoothTransforms[i,1]
    da = SmoothTransforms[i,2]
    transformationMatrix = np.zeros((2,3), np.float32)
    transformationMatrix[0,0] = np.cos(da)
    transformationMatrix[0,1] = -np.sin(da)
    transformationMatrix[1,0] = np.sin(da)
    transformationMatrix[1,1] = np.cos(da)
    transformationMatrix[0,2] = dx
    transformationMatrix[1,2] = dy
    stabilizedFrame = cv2.warpAffine(frame, transformationMatrix, (frame.shape[1],frame.shape[0]))
    
    # Scaling the image by 0.4%
    Transformation = cv2.getRotationMatrix2D((frame.shape[1]/2, frame.shape[0]/2), 0, 1.004)
    stabilizedFrame = cv2.warpAffine(stabilizedFrame, Transformation, (stabilizedFrame.shape[1], stabilizedFrame.shape[0]))
    
    # performing background subtraction
    gray = cv2.cvtColor(stabilizedFrame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray,7)
    fgmask = fgbg.apply(gray)
    _,thresh = cv2.threshold(fgmask, 10, 255, cv2.THRESH_BINARY)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernal,iterations = 6)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernal,iterations = 6)
    count+=1
    if(count>=800):
        cv2.imwrite("./output3/"+imageDirectory[i], opening)