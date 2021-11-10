"""
command to execute:
full: python eval_faster_rcnn.py --root "./pedestrian_detection_dataset" --test "./pedestrian_detection_dataset/PennFudanPed_full.json" --out "./jsons/full_predict_frcnn.json" --model "frcnn-mobilenet"
val: python eval_faster_rcnn.py --root "./pedestrian_detection_dataset" --test "./pedestrian_detection_dataset/PennFudanPed_val.json" --out "./jsons/val_predict_frcnn.json" --model "frcnn-resnet"

detections:
full:python3 eval_detections.py --pred "./jsons/full_predict_frcnn.json" --gt "./pedestrian_detection_dataset/PennFudanPed_full.json" 
val: python3 eval_detections.py --pred "./jsons/val_predict_frcnn.json" --gt "./pedestrian_detection_dataset/PennFudanPed_val.json"
"""

from torchvision.models import detection
from torch.utils.data import DataLoader
import numpy as np
import argparse
import torch
import cv2
import os
from time import time
import json
from my_classes import Dataset


#Helper Functions
def read_test_filenames(test_json):
	f = open(test_json)
	data = json.load(f)

	imgpath_to_id = {}
	for img_dict in data['images']:
	    imgpath_to_id[ img_dict['file_name']] = img_dict['id']
	return imgpath_to_id

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-r", "--root", type=str, required=True, help="path to the dataset root directory")
ap.add_argument("-t", "--test", type=str, required=True, help="path to test json")
ap.add_argument("-o", "--out", type=str, required=True, help="path to output json")
ap.add_argument("-m", "--model", type=str, default="frcnn-mobilenet",
	choices=["frcnn-resnet", "frcnn-mobilenet"], help="name of the object detection model")
args = vars(ap.parse_args())

# set the device we will be using to run the model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = {
	"frcnn-resnet": detection.fasterrcnn_resnet50_fpn,
	"frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn
}
# load the model and set it to evaluation mode
model = MODELS[args['model']](pretrained=True, progress=True, num_classes=91, pretrained_backbone=True).to(DEVICE)
model.eval()


dataset_root_dir = args['root']
test_json = args["test"]    
imgpath_to_id = read_test_filenames(test_json)
imgpaths = list(imgpath_to_id.keys())

# Generators
test_set = Dataset(imgpaths)
test_generator = DataLoader(test_set, batch_size= 5, shuffle= False, num_workers= 5)
    
thresh_confidence = 0.99

startTime = time()
output_path = "./output/frcnn/"
coco_format_output = []


for batch in test_generator:

	for img_path in batch:
		print(img_path)
		image = cv2.imread(os.path.join(dataset_root_dir,img_path))
		orig = image.copy()
		# convert the image from BGR to RGB channel ordering and change the
		# image from channels last to channels first ordering
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image.transpose((2, 0, 1))
		# add the batch dimension, scale the raw pixel intensities to the
		# range [0, 1], and convert the image to a floating point tensor
		image = np.expand_dims(image, axis=0)
		image = image / 255.0
		image = torch.FloatTensor(image)
		# send the input to the device and pass the it through the network to
		# get the detections and predictions
		image = image.to(DEVICE)
		detections = model(image)[0]

		# loop over the detections
		for i in range(0, len(detections["boxes"])):
			# extract the confidence (i.e., probability) associated with the
			# prediction
			confidence = detections["scores"][i].item()

			# filter out weak detections by ensuring the confidence is
			# greater than the minimum confidence
			if detections['labels'][i] == 1 and confidence > thresh_confidence:
				
				# compute the (x, y)-coordinates of the bounding box for the object
				box = detections["boxes"][i].detach().cpu().numpy()
				(startX, startY, endX, endY) = box.astype("int")
				label = "{:.2f}%".format(confidence * 100)
				# draw the bounding box and label on the image
				cv2.rectangle(orig, (startX, startY), (endX, endY),(0,0,250), 1)
				y = startY - 6 if startY - 6 > 6 else startY + 6
				cv2.putText(orig, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,250), 1)
				op_dict = {}
				op_dict['image_id'] = int(imgpath_to_id[img_path])
				op_dict['category_id'] = 1
				op_dict['bbox'] = [float(startX), float(startY), float(endX-startX), float(endY-startY)]
				op_dict['score'] = float(confidence)
				coco_format_output.append(op_dict)
		# write the output image
		cv2.imwrite(os.path.join(output_path,img_path),orig)
	
print("time taken = {:.2f} seconds for {} images".format(time()-startTime, len(imgpaths)))
with open(args["out"],"w") as op:
	op.write(json.dumps(coco_format_output))
