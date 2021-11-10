
# Evaluation code for Pedestrian Detection

## Instructions
pip install pycocotools

## Evaluation script
    python eval_detections.py --gt <path to ground truth annotations json> --pred <path to detections json>

The script `eval_detections.py` takes in ground truth annotations and predicted detections for the evaluation dataset and computes the following metrics:
1. Average Precision, computed over 10 IOU thresholds in the range 0.5:0.05:0.95
2. Average Recall computed at 1 detection per image.
3. Average Recall comptued at 10 detections per image.

`PennFudanPed_train.json`:  Contains COCO annotations for a randomly generated train split of the PennFudan dataset. 

`PennFudanPed_val.json`:  Contains COCO annotations for the corresponding validation split of the PennFudan dataset. 

The above script should be run for detections obtained using all the three methods mentioned below:
1. Pretrained HoG
2. Custom HoG trained using SVM on HoG features
3. Pretrained Faster RCNN 

## Required files
You need to create the following scripts for generating output annotations using the three methods:
1. `eval_hog_pretrained.py`
    This script should use a pretrained HoG detector to make predictions on the provided test set, and store the detections in COCO format in the output file. 

        python eval_hog_pretrained.py --root <path to dataset root directory> -- test <path to test json> --out <path to output json>

2. `eval_hog_custom.py`
   This script should use the custom HoG detector to make predictions on the provided test set, and store the detections in COCO format in the output file. 
    
        python eval_hog_custom.py --root <path to dataset root directory> -- test <path to test json> --out <path to output json> --model <path to trained SVM model>

3. `eval_faster_rcnn.py`
   This script should use a pretrained Faster RCNN model to make predictions on the provided test set, and store the detections in COCO format in the output file. 
    
        python eval_faster_rcnn.py --root <path to dataset root directory> -- test <path to test json> --out <path to output json> --model <path to pretrained Faster RCNN weights file>

Along with the above mentioned scripts, all training scripts and other utility code should also be present in the zipped folder submitted. 


   
