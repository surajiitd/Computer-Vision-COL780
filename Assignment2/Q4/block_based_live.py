import cv2
import numpy as np
import os
# import iou_code

def template_matching(image,template):
    # convert both the image and template to grayscale
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    #print("performing template matching...")
    #methods:- cv2.TM_SQDIFF(take minLoc)--------- cv2.TM_CCORR_NORMED(take maxLoc)
    result = cv2.matchTemplate(imageGray, templateGray, cv2.TM_CCORR_NORMED)  
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)

    # determine the starting and ending (x, y)-coordinates of the
    # bounding box
    (startX, startY) = maxLoc
    endX = startX + template.shape[1]
    endY = startY + template.shape[0]

    # draw the bounding box on the image
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
    return image,(startX,startY,endX,endY)
    
global startingPoint, endPoint

# Selecting template from the image
def selectRectangle(event, x, y, flags, param):
    global startingPoint, endPoint
    if event == cv2.EVENT_LBUTTONDOWN:
        startingPoint = [x,y]
    elif event == cv2.EVENT_LBUTTONUP:
        endPoint = [x,y]
        cv2.rectangle(image, (startingPoint[0], startingPoint[1]), (endPoint[0], endPoint[1]),  (255,255,255), 2)
        cv2.imshow("Mark", image)
        cv2.waitKey(0)


vid = cv2.VideoCapture(0)
  
while(True):
    ret, image = vid.read()
    cv2.imshow('frame', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.imshow("Sample image",image)
# img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
cv2.namedWindow("Mark")
cv2.setMouseCallback("Mark", selectRectangle)
cv2.imshow("Mark", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#crop sample_image to get the template
template = image[startingPoint[1]:endPoint[1],startingPoint[0]:endPoint[0]]
cv2.imshow('template', template)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('template.png',template)

vid = cv2.VideoCapture(0)
while True:
    ret,image = vid.read()
    image ,(px1,py1,px2,py2) = template_matching(image,template)
    cv2.imshow("Tracked image",image)
    # iou += iou_code.intersection_over_union(x1,y1,x2,y2,px1,py1,px2,py2)
    # cv2.imwrite("./track_predicted/bolt/"+filename,img)
    # n+=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# print("iou score = ",calc_IOU( ))
vid.release()