# Pedestrian Detection

In eval_hog_custom.py, the program will ask to train the model or to use the pretrained model (yes/no). 

If "yes", then place the positive and negative samples from google drive into the "positive_images_train" and "negative_images_train" directories respectively.

If no, we have already placed the pretrained model("`trained_hog_svm_model.p`") in the same directory.

Our code is expecting "--root" path as the directory containing PennFudanPed/ and input json files

### _Commands to execute_

1. python3 eval_hog_pretrained.py --root "./pedestrian_detection_dataset" --test "./pedestrian_detection_dataset/PennFudanPed_val.json" --out "./jsons/val_predict_hog_pretrained.json"

evaluation:
python3 eval_detections.py --pred "./jsons/val_predict_hog_pretrained.json" --gt "./pedestrian_detection_dataset/PennFudanPed_val.json"



2. python3 eval_hog_custom.py.py --root "./pedestrian_detection_dataset" --test "./pedestrian_detection_dataset/PennFudanPed_val.json" --out "./jsons/val_predict_hog_custom.json" --model "./trained_hog_svm_model.p"

evaluation:
python3 eval_detections.py --pred "./jsons/val_predict_hog_custom.json" --gt "./pedestrian_detection_dataset/PennFudanPed_val.json"



3. 
val: python eval_faster_rcnn.py --root "./pedestrian_detection_dataset" --test "./pedestrian_detection_dataset/PennFudanPed_val.json" --out "./jsons/val_predict_frcnn.json" --model "frcnn-resnet"

full: python eval_faster_rcnn.py --root "./pedestrian_detection_dataset" --test "./pedestrian_detection_dataset/PennFudanPed_full.json" --out "./jsons/full_predict_frcnn.json" --model "frcnn-mobilenet"


evaluation:
val: python3 eval_detections.py --pred "./jsons/val_predict_frcnn.json" --gt "./pedestrian_detection_dataset/PennFudanPed_val.json"
full: python3 eval_detections.py --pred "./jsons/full_predict_frcnn.json" --gt "./pedestrian_detection_dataset/PennFudanPed_full.json"
