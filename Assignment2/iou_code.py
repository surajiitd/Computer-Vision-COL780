import cv2
import numpy as np

def calc_Intersection_length(a0, a1, b0, b1):
    if a0 >= b0 and a1 <= b1: # Contained
        intersection = a1 - a0
    elif a0 < b0 and a1 > b1: # Contains
        intersection = b1 - b0
    elif a0 < b0 and a1 > b0: # Intersects right
        intersection = a1 - b0
    elif a1 > b1 and a0 < b1: # Intersects left
        intersection = b1 - a0
    else: # No intersection (either side)
        intersection = 0

    return intersection

def intersection_over_union(x1,y1,x2,y2,px1,py1,px2,py2):

	inters_width = calc_Intersection_length(x1, x2, px1, px2)        
	inters_height = calc_Intersection_length(y1, y2, py1, py2)


	#print(inters_width,inters_height)
	inters_area = inters_width * inters_height
	#print("intersection area = ",inters_area)

	gt_rect_area = (x2-x1)*(y2-y1)
	pred_rect_area = (px2-px1)*(py2-py1)
	union_area = gt_rect_area + pred_rect_area - inters_area
	#print("union area = ",union_area)
	return inters_area/union_area

def calc_IOU(gt_path,pred_templ_path,delim = ','):
	iou = 0.

	gt = np.genfromtxt(gt_path,delimiter=delim)
	pred_templ = np.genfromtxt(pred_templ_path,delimiter=',')
	# N = len(gt)
	gt = gt[1:]
	N = len(gt) if len(gt) < len(pred_templ) else len(pred_templ)
	for i in range(N):
		coord = gt[i]
		x1,y1,x2,y2 = int(coord[0]),int(coord[1]),int(coord[0])+int(coord[2]),int(coord[1])+int(coord[3])
		coord = pred_templ[i]
		px1,py1,px2,py2 = int(coord[0]),int(coord[1]),int(coord[0])+int(coord[2]),int(coord[1])+int(coord[3])
		iou += intersection_over_union(x1,y1,x2,y2,px1,py1,px2,py2)
	iou /= (N-1)
	return iou
#Q1
print("\n*******************Q1:***********************\n")
print("blurcar iou = " + str( calc_IOU("./data/BlurCar2/groundtruth_rect.txt", "./Q1/pred_blurcar.txt",delim = '\t')))
print("bolt iou = " + str( calc_IOU("./data/Bolt/groundtruth_rect.txt", "./Q1/pred_bolt.txt")) )
print("liquor iou = " + str( calc_IOU("./data/Liquor/groundtruth_rect.txt", "./Q1/pred_liquor.txt")))


#Q1
print("\n\n*******************Q2:***********************\n")
print("blurcar iou = " + str( calc_IOU("./data/BlurCar2/groundtruth_rect.txt", "./Q2/pred_blurcar.txt",delim = '\t')))
print("bolt iou = " + str( calc_IOU("./data/Bolt/groundtruth_rect.txt", "./Q2/pred_bolt.txt")))
print("liquor iou = " + str( calc_IOU("./data/Liquor/groundtruth_rect.txt", "./Q2/pred_liquor.txt")))



#Q1
print("\n\n*******************Q3:***********************\n")
print("blurcar iou = " + str( calc_IOU("./data/BlurCar2/groundtruth_rect.txt", "./Q3/pred_blurcar.txt",delim = '\t')))
print("bolt iou = " + str( calc_IOU("./data/Bolt/groundtruth_rect.txt", "./Q3/pred_bolt.txt")))
print("liquor iou = " + str( calc_IOU("./data/Liquor/groundtruth_rect.txt", "./Q3/pred_liquor.txt")))


