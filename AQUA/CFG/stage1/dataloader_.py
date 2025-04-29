from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.evaluation import COCOEvaluator

from datasets import DetrDatasetMapper
from datasets.builtin_meta import COCO_CATEGORIES
from datasets.datasets_utils import build_dataset, build_sub_dataset

thing_classes = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]

dataloader = OmegaConf.create()
dataloader.train = L(build_detection_train_loader)(
    dataset=L(build_dataset)(
        name="train",
        meta_name="coco",
        json_file="inputs/data/annotations/labels.json",
        image_root="inputs/data/train2017",
    ),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        augmentation_with_crop=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(400, 500, 600),
                sample_style="choice",
            ),
            L(T.RandomCrop)(
                crop_type="absolute_range",
                crop_size=(384, 600),
            ),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        is_train=True,
        mask_on=False, # segmentation
        img_format="RGB",
        categories_names=thing_classes,
    ),
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(build_dataset)(
        name="test",
        meta_name="coco",
        json_file="inputs/data/annotations/labels.json",
        image_root="inputs/data/train2017",
        filter_empty=False,
    ),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.ResizeShortestEdge)(
                short_edge_length=800,
                max_size=1333,
            ),
        ],
        augmentation_with_crop=None,
        is_train=False,
        mask_on=False,
        img_format="RGB",
        categories_names=thing_classes,
    ),
    num_workers=4,
)

dataloader.test_sub = L(build_detection_test_loader)(
    dataset=L(build_sub_dataset)(
        original_name="test",
        n=100,
    ),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.ResizeShortestEdge)(
                short_edge_length=800,
                max_size=1333,
            ),
        ],
        augmentation_with_crop=None,
        is_train=False,
        mask_on=False,
        img_format="RGB",
        categories_names=thing_classes,
    ),
    num_workers=4,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.name}",
)

