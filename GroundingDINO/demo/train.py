import argparse
import os
import torch
torch.set_printoptions(sci_mode=False, precision=4)
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
import wandb
from utils_.hf_up import upload

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.misc import clean_state_dict, collate_fn
import groundingdino.datasets.transforms as T
from groundingdino.util import get_tokenlizer, box_ops

from test_ap_on_coco import CocoDetection, load_model, find_coco_id, build_captions_and_token_span, create_positive_map_from_span

class PostProcessCocoTraining(nn.Module):
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
        # 300 개 뽑는데, 같은 박스 내에서 logit 큰게 여러개 있으면 중복선정도 가능
        # boxes 는 복사해서 생성되므로 별개의 prediction 으로 여김
        # 따라서 topk_prob 의 argmax != labels
        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        boxes = torch.gather(
            boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        topk_prob = torch.gather(
            prob, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 92))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = {'scores': scores, 'labels': labels, 'boxes': boxes, 'prob': topk_prob}
        return results

@torch.no_grad()
def hungarian(outputs, targets,
              set_cost_class=1.0,
              set_cost_bbox=5.0,
              set_cost_giou=2.0):
    """
    Hungarian matching 수행: cost = cls + bbox + giou

    Args:
        outputs: dict with keys: 'boxes' (B, N, 4), 'prob' (B, N, C), 'labels' (B, N)
        targets: list of GT dicts, each with 'boxes' (cxcywh) and 'labels'
    Returns:
        list of Tensors: 각 GT에 대응하는 DT 인덱스
    """
    bs = outputs["boxes"].shape[0]
    dt_boxes_batch = outputs["boxes"].clone()
    dt_probs_batch = outputs["prob"].clone()
    matched_indices = []

    for b in range(bs):
        dt_boxes = dt_boxes_batch[b]       # (N, 4)
        dt_probs = dt_probs_batch[b]       # (N, C)
        tgt = targets[b]
        img_h, img_w = map(float, tgt["orig_size"])
        dt_boxes[:, [0, 2]] /= img_w
        dt_boxes[:, [1, 3]] /= img_h

        tgt_boxes = box_ops.box_cxcywh_to_xyxy(tgt["boxes"].to(dt_boxes.device))  # (M, 4)
        tgt_labels = torch.as_tensor(tgt["labels"], device=dt_boxes.device)       # (M,)

        if len(tgt_boxes) == 0 or len(dt_boxes) == 0:
            matched_indices.append(torch.full((len(tgt_boxes),), -1, dtype=torch.long))
            continue

        # -----------------------------
        # 1. Classification cost (focal/CE)
        # -----------------------------
        cls_cost = -dt_probs[:, tgt_labels]  # (N, M): i-th DT, j-th GT class의 확률 음수

        # -----------------------------
        # 2. BBox L1 cost
        # -----------------------------
        cost_bbox = torch.cdist(dt_boxes, tgt_boxes, p=1)  # (N, M)

        # -----------------------------
        # 3. GIoU cost
        # -----------------------------
        giou = box_ops.generalized_box_iou(dt_boxes, tgt_boxes)  # (N, M)
        cost_giou = -giou  # GIoU는 클수록 좋으므로 음수로 변환

        # -----------------------------
        # 4. Total cost
        # -----------------------------
        C = (set_cost_class * cls_cost +
             set_cost_bbox * cost_bbox +
             set_cost_giou * cost_giou)  # (N, M)

        C = C.cpu().detach().numpy()
        row_ind, col_ind = linear_sum_assignment(C)

        match = torch.full((len(tgt_boxes),), -1, dtype=torch.long)
        match[col_ind] = torch.as_tensor(row_ind, dtype=torch.long)
        matched_indices.append(match)

    return matched_indices

def criterion(results, targets, cls_weight=1.0, l1_weight=2.0, giou_weight=0.5):

    """
       Hungarian 매칭 결과를 바탕으로 classification + box (L1 + GIoU) loss 계산.

       Args:
           results (dict): model outputs {'boxes': [B, N, 4], 'labels': [B, N], ...}
           targets (list): list of GT dicts, each with 'boxes' (cxcywh) and 'labels'
           matching (list of Tensor): GT → DT index (size [num_gt]), -1 means unmatched
           cls_weight, l1_weight, giou_weight: 가중치

       Returns:
           dict: {
               'loss_cls': ...,
               'loss_bbox': ...,
               'loss_giou': ...,
               'loss_total': ...
           }
       """
    device = results["boxes"].device
    total_cls_loss = 0.0
    total_l1_loss = 0.0
    total_giou_loss = 0.0
    num_boxes = 0
    matching = hungarian(results, targets, giou_weight)
    for b, match in enumerate(matching):
        dt_boxes = results["boxes"][b]  # [N, 4]
        dt_logits = results["prob"][b]  # [N, C]
        tgt = targets[b]
        img_h, img_w = map(float, tgt["orig_size"])

        matched_inds = match
        if matched_inds.numel() == 0:
            continue
        valid_mask = matched_inds >= 0
        if valid_mask.sum() == 0:
            continue

        gt_inds = torch.arange(len(matched_inds), device=device)[valid_mask]
        dt_inds = matched_inds[valid_mask]

        # 각 GT에 대한 예측 box, label, logit 추출
        tgt_boxes = box_ops.box_cxcywh_to_xyxy(tgt["boxes"].to(device)[gt_inds])  # GT: [M, 4] → xyxy
        pred_boxes = dt_boxes[dt_inds]  # [M, 4]
        pred_boxes[:, [0, 2]] /= img_w
        pred_boxes[:, [1, 3]] /= img_h
        tgt_labels = torch.as_tensor(tgt["labels"], device=device)[gt_inds]  # [M]
        pred_logits = dt_logits[dt_inds]  # [M, C]

        # Classification Loss (Cross Entropy)
        tgt_onehot = F.one_hot(tgt_labels, num_classes=92).float()
        cls_cost = pred_logits * tgt_onehot
        loss_cls = - cls_cost.sum()

        # Box L1 Loss
        loss_bbox = F.l1_loss(pred_boxes, tgt_boxes, reduction='sum')

        # GIoU Loss
        giou = box_ops.generalized_box_iou(pred_boxes, tgt_boxes)
        loss_giou = 1.0 - torch.diag(giou)  # GIoU → 1 - GIoU
        loss_giou = loss_giou.sum()

        total_cls_loss += loss_cls * cls_weight
        total_l1_loss += loss_bbox * l1_weight
        total_giou_loss += loss_giou * giou_weight
        num_boxes += len(gt_inds)

    # normalize by number of boxes
    num_boxes = max(num_boxes, 1)
    total_cls_loss = total_cls_loss / num_boxes
    total_l1_loss = total_l1_loss / num_boxes
    total_giou_loss = total_giou_loss / num_boxes

    total = total_cls_loss + total_l1_loss + total_giou_loss

    return {
        "loss_cls": total_cls_loss,
        "loss_bbox": total_l1_loss,
        "loss_giou": total_giou_loss,
        "loss_total": total
    }



class Trainer:
    def __init__(self, model, optimizer, tokenlizer, device, dataset, cfg):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.tokenlizer = tokenlizer
        self.device = device
        self.dataset = dataset
        self.cfg = cfg
        self.id2name = {cat["id"]: cat["name"] for cat in dataset.coco.dataset["categories"]}
        self.category_dict = dataset.coco.cats
        self.step = 0
        load_dotenv()
        WDB = os.getenv('WANDB_API_KEY')
        wandb.login(key=WDB)
        self.wandb = wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=cfg.wandb_name,
        )

    def create_caption_from_labels(self, labels):
        cat_names = [self.id2name[l] for l in labels]
        cat_list = sorted(list(set(cat_names)))  # 중복 제거 및 정렬
        return " . ".join(cat_list) + " .", cat_list

    def train(self, dataloader, max_step):
        self.model.train()
        total_loss = 0
        print("Total Len:" , len(dataloader))

        with tqdm(total=max_step, desc="Process") as pbar:
            for images, targets in dataloader:
                images = images.tensors.to(self.device)

                # 1. 이미지 별 캡션 생성 및 positive map 구축
                captions, cat_lists = [], []
                for target in targets:
                    cap, cat_list = self.create_caption_from_labels(target["labels"])
                    captions.append(cap)
                    cat_lists.append(cat_list)

                postprocessor = PostProcessCocoTraining(
                    cat_lists=cat_lists, cats_dict=self.category_dict, tokenlizer=self.tokenlizer)

                # 2. 모델 forward
                outputs = self.model(images, captions=captions)
                orig_target_sizes = torch.stack(
                    [t["orig_size"] for t in targets], dim=0).to(images.device)
                results = postprocessor(outputs, orig_target_sizes)


                # 3. 손실 계산
                loss_dict = criterion(results, targets)
                loss = loss_dict["loss_total"]

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                if self.step % self.cfg.log_frq == 0:
                    log_res = {key: val.item() for key, val in loss_dict.items()}
                    self.wandb.log(log_res, step=self.step)
                if self.step != 0 and self.step % self.cfg.save_frq == 0:
                    self.save_checkpoint(self.step)
                self.step += 1
                pbar.update(1)

        avg_loss = total_loss / max_step

        return avg_loss

    def save_checkpoint(self, step, save_dir="output/weights"):
        os.makedirs(save_dir, exist_ok=True)
        ckpt_path = os.path.join(save_dir, f"ckpt_{step}.pth")
        torch.save({"model": self.model.state_dict()}, ckpt_path)
        print(f"Model checkpoint saved to {ckpt_path}")
        upload(self.cfg.wandb_name)


def main(args):
    device = args.device
    cfg = SLConfig.fromfile(args.config_file)
    cfg.device = device

    model = load_model(args.config_file, args.checkpoint_path)
    model = model.to(device)

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = CocoDetection(args.image_dir, args.anno_path, transforms=transform)
    data_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                             num_workers=4, collate_fn=collate_fn)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    tokenlizer = get_tokenlizer.get_tokenlizer(cfg.text_encoder_type)

    trainer = Trainer(model, optimizer, tokenlizer, device, dataset, cfg)
    trainer.train(data_loader, cfg.max_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounding DINO training on COCO")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--config_file", "-c", type=str, required=True)
    parser.add_argument("--checkpoint_path", "-p", type=str, default=None)
    parser.add_argument("--anno_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)

    args = parser.parse_args()
    main(args)
