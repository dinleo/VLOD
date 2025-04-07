from copy import deepcopy
from pycocotools.coco import COCO
from dotenv import load_dotenv
import os

load_dotenv()
coco_root = os.getenv('COCODATA')


# 기존 COCO 객체: coco (type: pycocotools.coco.COCO)
# 가장 높은 annotation ID 찾기
coco = COCO(coco_root + "/annotations/instances_train2017.json")

max_ann_id = max(ann["id"] for ann in coco.dataset["annotations"])
coco.dataset["categories"].append({
        "id": 91,
        "name": "object",
        "supercategory": "object"
    })

# 어노테이션 추가
new = []
for i, ann in enumerate(coco.dataset["annotations"]):
    new_ann = deepcopy(ann)
    new_ann["id"] = max_ann_id + i + 1  # 새로운 유일한 ID 부여
    new_ann["category_id"] = 91         # object 클래스로 바꾸기
    new.append(new_ann)

coco.dataset["annotations"].extend(new)

# 내부 인덱스 재생성
coco.createIndex()
print(len(coco.dataset["annotations"]))
import json
with open(coco_root + "/annotations/labels.json", "w") as f:
    json.dump(coco.dataset, f, indent=2)

