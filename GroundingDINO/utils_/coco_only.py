from pycocotools.coco import COCO
from dotenv import load_dotenv
import os

load_dotenv()
coco_root = os.getenv('COCODATA')
working_root = os.getenv('WORKING')

coco = COCO(coco_root + "/annotations/instances_train2017.json")


coco.dataset["categories"] = [{
        "id": 91,
        "name": "object",
        "supercategory": "object"
    }]

for i, ann in enumerate(coco.dataset["annotations"]):
    ann["category_id"] = 91         # object 클래스로 바꾸기

coco.createIndex()
print(len(coco.dataset["annotations"]))
import json
with open(working_root + "/annotations/labels_only.json", "w") as f:
    json.dump(coco.dataset, f, indent=2)