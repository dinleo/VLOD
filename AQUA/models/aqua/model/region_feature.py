from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from rpns import GeneralizedRCNN

# 1. config load
cfg = get_cfg()
cfg.merge_from_file("faster_rcnn_R_50_FPN_3x.yaml")

# 2. model 생성 (full model → RPN 꺼내기 위함)
model = GeneralizedRCNN(cfg)
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

# 3. RPN 모듈 추출
rpn = model.proposal_generator
