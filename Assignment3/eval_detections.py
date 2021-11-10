from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse

parser = argparse.ArgumentParser(description='Evaluation script')
parser.add_argument('--gt', type=str, help='path to ground truth annotations')
parser.add_argument('--pred', type=str, help='path to predicted detections')


args = parser.parse_args()

anno = COCO(args.gt)  # init annotations api
pred = anno.loadRes(args.pred)  # init predictions api
is_coco = True
test_indices = range(len(anno.imgs))

eval = COCOeval(anno, pred, 'bbox')

if is_coco:
    eval.params.imgIds = test_indices

eval.evaluate()
eval.accumulate()
eval.summarize()

AP = eval.stats[0]
AR_at_1 = eval.stats[6]
AR_at_10 = eval.stats[7]
print('\nAverage Precision = ', AP)
print('Average Recall @ 1 detection per image = ', AR_at_1)
print('Average Recall @ 10 detections per image = ', AR_at_10)

print("precsion: ",eval.eval['precision'].shape)
print("recall: ",eval.eval['recall'].shape)
# print("recall: ",eval.eval['recall'])
# print(eval.stats)
