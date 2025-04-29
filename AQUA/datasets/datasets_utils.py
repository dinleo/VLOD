from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog, get_detection_dataset_dicts
from datasets.builtin_meta import _get_builtin_metadata

def build_dataset(name, meta_name, json_file, image_root, filter_empty=True):
    if name not in DatasetCatalog.list():
        register_coco_instances(
            name,
            _get_builtin_metadata(meta_name),
            json_file,
            image_root,
        )
    return get_detection_dataset_dicts(name, filter_empty=filter_empty)


def build_sub_dataset(original_name: str, n: int):
    """
    Registers subset(={name}_sub) of dataset in the DatasetCatalog and MetadataCatalog.
    Creates a new dataset entry with the specified subset of the original dataset.

    :param name: The name of the original dataset to create a subset from. Must already be registered in DatasetCatalog.
    :param n: The number of samples to include in the subset. Must be greater than zero.
    :return: None
    :raises ValueError: If the specified dataset name is not found in DatasetCatalog.
    """
    if n<1: return

    full_dataset = DatasetCatalog.get(original_name)
    metadata = MetadataCatalog.get(original_name).as_dict()

    new_name = original_name + "_sub"
    subset = full_dataset[:n]
    metadata['name'] = new_name
    DatasetCatalog.register(new_name, lambda: subset)
    MetadataCatalog.get(new_name).set(
        **metadata
    )
    return get_detection_dataset_dicts(new_name, filter_empty=False)