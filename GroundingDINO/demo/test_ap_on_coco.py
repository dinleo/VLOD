import argparse
import os
import sys
import time
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler, Subset

from groundingdino.models import build_model
import groundingdino.datasets.transforms as T
from groundingdino.util import box_ops, get_tokenlizer
from groundingdino.util.misc import clean_state_dict, collate_fn
from groundingdino.util.slconfig import SLConfig

# from torchvision.datasets import CocoDetection
import torchvision

from groundingdino.util.vl_utils import build_captions_and_token_span, create_positive_map_from_span
from groundingdino.datasets.cocogrounding_eval import CocoGroundingEvaluator


def load_model(model_config_path: str, model_checkpoint_path: str, device: str = "cuda"):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model

def find_coco_id(coco_cat: dict, name: str):
    for _, cat_info in coco_cat.items():
        if cat_info["name"].lower() == name.lower():
            return cat_info["id"]
    return 0

def create_caption_from_labels(id2name, labels):
    cat_names = [id2name[l] for l in labels]
    cat_list = sorted(list(set(cat_names)))  # 중복 제거 및 정렬
    return " . ".join(cat_list) + " .", cat_list

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


class PostProcessCocoGrounding(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self, num_select=300, cat_lists=None, cats_dict=None, tokenlizer=None) -> None:
        super().__init__()
        self.num_select = num_select

        assert cat_lists is not None
        new_pos_map_list = []
        for cat_list in cat_lists:
            captions, cat2tokenspan = build_captions_and_token_span(cat_list, True)
            tokenspanlist = [cat2tokenspan[cat] for cat in cat_list]
            positive_map = create_positive_map_from_span(
                tokenlizer(captions), tokenspanlist)  # 80, 256. normed

            # id_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46,
            #           41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90
            #           , 80:91}
            id_map = {}
            for i, c in enumerate(cat_list):
                id_map[i] = find_coco_id(cats_dict, c)

            # build a mapping from label_id to pos_map
            new_pos_map = torch.zeros((92, 256))
            for k, v in id_map.items():
                new_pos_map[v] = positive_map[k]
            new_pos_map_list.append(new_pos_map)

        self.positive_maps = torch.stack(new_pos_map_list, dim=0)

    @torch.no_grad()
    def forward(self, outputs, target_sizes, not_to_xyxy=False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        # pos map to logit
        prob_to_token = out_logits.sigmoid()  # bs, 900, 256
        pos_maps = self.positive_maps.to(prob_to_token.device)
        # (bs, 900, 256) @ (cls, 256).T -> (bs, 900, cls)
        prob_to_label = prob_to_token @ pos_maps.transpose(1, 2)

        # if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
        #     import ipdb; ipdb.set_trace()

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = prob_to_label
        topk_values, topk_indexes = torch.topk(
            prob.view(out_logits.shape[0], -1), num_select, dim=1) # top 300 in (900 * cls = ex: 81900)
        scores = topk_values
        topk_boxes = topk_indexes // prob.shape[2] # 0~899
        labels = topk_indexes % prob.shape[2] # 0~cls

        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        boxes = torch.gather(
            boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b}
                   for s, l, b in zip(scores, labels, boxes)]

        return results


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
    print(save_name)

    # build post captions
    id2name = {cat["id"]: cat["name"] for cat in dataset.coco.dataset["categories"]}
    category_dict = dataset.coco.dataset['categories']
    cat_list_tgt = [item['name'] for item in category_dict]
    caption_tgt = " . ".join(cat_list_tgt) + ' .'
    print("Targets prompt:", caption_tgt)

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
                if cfg.all_cap:
                    captions.append(caption_tgt)
                    cat_lists.append(cat_list_tgt)
                else:
                    captions.append(cap)
                    cat_lists.append(cat_list)
            postprocessor = PostProcessCocoGrounding(
                cat_lists=cat_lists, cats_dict=dataset.coco.cats, tokenlizer=tokenlizer)
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
