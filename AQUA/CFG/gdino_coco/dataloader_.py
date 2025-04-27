from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator

from datasets import DetrDatasetMapper
from datasets.builtin_meta import COCO_CATEGORIES, _get_builtin_metadata

# Register Instance
register_coco_instances(
    "train",
    {},
    "inputs/data/annotations/labels.json",
    "inputs/data/train2017",
)
register_coco_instances(
    "test",
    {},
    "inputs/data/annotations/labels.json",
    "inputs/data/train2017"
)
# Register Metadata
metadata = _get_builtin_metadata("coco")
MetadataCatalog.get("train").set(
    evaluator_type="coco",
    **metadata,
)
MetadataCatalog.get("test").set(
    evaluator_type="coco",
    **metadata,
)


dataloader = OmegaConf.create()
thing_classes = [k["name"] for k in COCO_CATEGORIES if k["isthing"] == 1]

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="train"),
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
    dataset=L(get_detection_dataset_dicts)(names="test", filter_empty=False),
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
    dataset_name="${..test.dataset.names}",
)


def register_coco_subset(name: str, n: int):
    """
    Registers subset(={name}_sub) of dataset in the DatasetCatalog and MetadataCatalog.
    Creates a new dataset entry with the specified subset of the original dataset.

    :param name: The name of the original dataset to create a subset from. Must already be registered in DatasetCatalog.
    :param n: The number of samples to include in the subset. Must be greater than zero.
    :return: None
    :raises ValueError: If the specified dataset name is not found in DatasetCatalog.
    """
    if n<1: return
    if name not in DatasetCatalog.list():
        raise ValueError(f"[ERROR] Dataset '{name}' is not registered in DatasetCatalog.")

    full_dataset = DatasetCatalog.get(name)
    subset = full_dataset[:n]
    DatasetCatalog.register(name + "_sub", lambda: subset)
    MetadataCatalog.get(name + "_sub").set(
        evaluator_type="coco",
        **metadata,
    )