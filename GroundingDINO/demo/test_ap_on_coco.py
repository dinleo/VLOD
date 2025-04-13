import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
torch.set_printoptions(sci_mode=False, precision=4)
from torch.utils.data import DataLoader

from groundingdino.models import build_model
import groundingdino.datasets.transforms as T
from groundingdino.util import box_ops, get_tokenlizer
from groundingdino.util.misc import clean_state_dict, collate_fn
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.captions import create_caption_from_labels, PostProcessCoco

# from torchvision.datasets import CocoDetection
import torchvision
from groundingdino.datasets.cocogrounding_eval import CocoGroundingEvaluator


def load_model(model_config_path: str, model_checkpoint_path: str, device: str = "cuda"):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)  # target: list

        # import ipdb; ipdb.set_trace()

        w, h = img.size
        boxes = [obj["bbox"] for obj in target]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]  # xywh -> xyxy
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        # filt invalid boxes/masks/keypoints
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = [obj["category_id"] for obj in target]

        target_new = {}
        image_id = self.ids[idx]
        target_new["image_id"] = image_id
        target_new["boxes"] = boxes
        target_new["labels"] = labels
        target_new["orig_size"] = torch.as_tensor([int(h), int(w)])

        if self._transforms is not None:
            img, target = self._transforms(img, target_new)

        return img, target


def main(args):
    # config
    cfg = SLConfig.fromfile(args.config_file)
    n = args.num_sample

    # build model
    model = load_model(args.config_file, args.checkpoint_path)
    model = model.to(args.device)
    model = model.eval()

    # build dataloader
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dataset = CocoDetection(
        args.image_dir, args.anno_path, transforms=transform)
    data_loader = DataLoader(
        dataset, batch_size=cfg.test_batch, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    if n == -1:
        n = len(dataset)
    save_name = args.checkpoint_path.split("/")[-1].split(".")[-2] + "/" + args.anno_path.split("/")[-1].split(".")[
        -2] + "_" + str(n)
    if cfg.dev_test:
        save_name = "dev_test/" + save_name

    # build post captions
    id2name = {cat["id"]: cat["name"] for cat in dataset.coco.dataset["categories"]}
    category_dict = dataset.coco.dataset['categories']
    cat_list_all = [item['name'] for item in category_dict]
    cap_all = " . ".join(cat_list_all) + ' .'
    print("All Prompt:", cap_all)

    tokenlizer = get_tokenlizer.get_tokenlizer(cfg.text_encoder_type)

    # build evaluator
    evaluator = CocoGroundingEvaluator(
        dataset.coco, iou_types=("bbox",), useCats=True)

    with tqdm(total=n, desc="Process") as pbar:
        # run inference
        for i, (images, targets) in enumerate(data_loader):
            if i >= n:
                break
            # get images and captions
            images = images.tensors.to(args.device)


            captions, cat_lists = [], []
            for target in targets:
                cap, cat_list = create_caption_from_labels(id2name, target["labels"])
                if cfg.tg_cat_only:
                    captions.append(cap)
                    cat_lists.append(cat_list)
                else:
                    captions.append(cap_all)
                    cat_lists.append(cat_list_all)
            postprocessor = PostProcessCoco(
                cat_lists=cat_lists, cats2id_dict=dataset.coco.cats, tokenlizer=tokenlizer, train_mode=False)
            # feed to the model
            outputs = model(images, captions=captions)

            orig_target_sizes = torch.stack(
                [t["orig_size"] for t in targets], dim=0).to(images.device)
            results = postprocessor(outputs, orig_target_sizes)
            cocogrounding_res = {
                target["image_id"]: output for target, output in zip(targets, results)}
            evaluator.update(cocogrounding_res)

            pbar.update(1)

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    evaluator.save_coco_eval_json(save_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Grounding DINO eval on COCO", add_help=True)
    # load model
    parser.add_argument("--config_file", "-c", type=str,
                        required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--device", type=str, default="cuda",
                        help="running device (default: cuda)")

    # post processing
    parser.add_argument("--num_select", type=int, default=300,
                        help="number of topk to select")

    # coco info
    parser.add_argument("--anno_path", type=str,
                        required=True, help="coco root")
    parser.add_argument("--image_dir", type=str,
                        required=True, help="coco image dir")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="number of workers for dataloader")
    parser.add_argument("--num_sample", type=int, default=-1,
                        help="number of test samples")
    args = parser.parse_args()

    main(args)
