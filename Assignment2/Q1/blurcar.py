import cv2
import numpy as np
import os

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
	cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 3)
	return image,(startX,startY,endX,endY)
	

if __name__=="__main__":
	gt = np.genfromtxt("../data/BlurCar2/groundtruth_rect.txt",delimiter='\t')
	coord = gt[0]
	img_path = "../data/BlurCar2/img/"
	imgname = "0001.jpg"
	x,y,w,h = int(coord[0]),int(coord[1]),int(coord[2]),int(coord[3])

	image = cv2.imread(img_path+imgname)
	template = image[y:y+h,x:x+w]

	cv2.imwrite("./template.png",template)
	filenames = os.listdir(img_path)
	filenames.sort()

	iou = 0.0
	n = 0
	file1 = open('./pred_blurcar.txt', 'w')
	for i,filename in enumerate(filenames):
		print(filename)
		coord = gt[i]
		x1,y1,x2,y2 = int(coord[0]),int(coord[1]),int(coord[0])+int(coord[2]),int(coord[1])+int(coord[3])

		img = cv2.imread(img_path+filename)
		img ,(px1,py1,px2,py2) = template_matching(img,template)

		s = str(px1)+","+str(py1)+","+str(template.shape[1])+","+str(template.shape[0])
		file1.write(s+'\n')
		cv2.imwrite("./block_based_predicted/blurcar/"+filename,img)
		n+=1
	
