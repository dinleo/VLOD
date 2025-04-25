import argparse
import os
import torch
torch.set_printoptions(sci_mode=False, precision=4)
import torch.optim as optim
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
import wandb
from utils_.hf_up import upload

from models.groundingdino.util.slconfig import SLConfig
from models.groundingdino.util.misc import collate_fn
from models import groundingdino as T
from models.groundingdino.util import get_tokenlizer, box_ops
from models.groundingdino.util import create_caption_from_labels, PostProcessCoco

from test_ap_on_coco import CocoDetection, load_model

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

def criterion(results, targets, cls_weight=0.1, bbox_weight=2.0, giou_weight=1.0):

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
        dt_prob = results["prob"][b]  # [N, C]
        tgt = targets[b]
        img_h, img_w = map(float, tgt["orig_size"])

        matched_inds = match
        if matched_inds.numel() == 0:
            continue
        valid_mask = matched_inds >= 0
        if valid_mask.sum() == 0:
            continue

        gt_inds = torch.arange(len(matched_inds), device=device)[valid_mask]
        dt_match_inds = matched_inds[valid_mask]

        # 각 GT에 대한 예측 box, label, prob 추출
        tgt_boxes = box_ops.box_cxcywh_to_xyxy(tgt["boxes"].to(device)[gt_inds])  # GT: [M, 4] → xyxy
        pred_boxes = dt_boxes[dt_match_inds]  # [M, 4]
        pred_boxes[:, [0, 2]] /= img_w
        pred_boxes[:, [1, 3]] /= img_h
        tgt_labels = torch.as_tensor(tgt["labels"], device=device)[gt_inds]  # [M]

        # Classification Loss (B-Cross Entropy)

        # Only consider class in prompt
        class_sets = torch.unique(tgt_labels) # [M]
        lb_to_cls = {c.item(): i for i, c in enumerate(class_sets)}

        # DT
        selected_probs = dt_prob[:, class_sets] # [N, prompt C]
        N, C = selected_probs.shape
        # GT
        target_mask = torch.zeros((N, C), device=device) # [N, prompt C]
        gt_prompt_indices = torch.tensor([lb_to_cls[l.item()] for l in tgt_labels], device=device)
        target_mask[dt_match_inds, gt_prompt_indices] = 1.0

        loss_cls = F.binary_cross_entropy(selected_probs, target_mask, reduction='sum')

        # Box L1 Loss
        loss_bbox = F.l1_loss(pred_boxes, tgt_boxes, reduction='sum')

        # GIoU Loss
        giou = box_ops.generalized_box_iou(pred_boxes, tgt_boxes)
        loss_giou = 1.0 - torch.diag(giou)  # GIoU → 1 - GIoU
        loss_giou = loss_giou.sum()

        total_cls_loss += loss_cls * cls_weight
        total_l1_loss += loss_bbox * bbox_weight
        total_giou_loss += loss_giou * giou_weight
        num_boxes += len(gt_inds)

    # normalize by number of boxes
    num_boxes = max(num_boxes, 1)
    total_cls_loss = total_cls_loss / num_boxes
    total_l1_loss = total_l1_loss / num_boxes
    total_giou_loss = total_giou_loss / num_boxes

    total = total_cls_loss + total_l1_loss + total_giou_loss
    res = {
        "loss_cls": total_cls_loss,
        "loss_bbox": total_l1_loss,
        "loss_giou": total_giou_loss,
        "loss_total": total
    }

    return res



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
        if not cfg.dev_test:
            load_dotenv()
            WDB = os.getenv('WANDB_API_KEY')
            wandb.login(key=WDB)
            self.wandb = wandb.init(
                entity=cfg.wandb_entity,
                project=cfg.wandb_project,
                name=cfg.title,
            )

    def train(self, dataloader, max_step):
        self.model.train()
        total_loss = 0
        print("Data Len:" , len(dataloader))
        print("All Cat Len", len(self.category_dict))

        with tqdm(total=max_step, desc="Process") as pbar:
            for images, targets in dataloader:
                images = images.tensors.to(self.device)

                # 1. 이미지 별 캡션 생성 및 positive map 구축
                captions, cat_lists = [], []
                for target in targets:
                    cap, cat_list = create_caption_from_labels(self.id2name, target["labels"])
                    captions.append(cap)
                    cat_lists.append(cat_list)

                postprocessor = PostProcessCoco(
                    cat_lists=cat_lists, cats2id_dict=self.category_dict, tokenlizer=self.tokenlizer)

                # 2. 모델 forward
                outputs = self.model(images, captions=captions)
                orig_target_sizes = torch.stack(
                    [t["orig_size"] for t in targets], dim=0).to(images.device)
                results = postprocessor(outputs, orig_target_sizes)


                # 3. 손실 계산
                loss_dict = criterion(results, targets)
                loss = loss_dict["loss_total"]
                # from torchviz import make_dot
                # make_dot(loss, params=dict(self.model.named_parameters()), show_attrs=True, show_saved=True).render("output/loss", format="pdf")
                # exit()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                if self.step % self.cfg.log_frq == 0 and not self.cfg.dev_test:
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
        upload(self.cfg.up_dir)


def main(args):
    device = args.device
    cfg = SLConfig.fromfile(args.config_file)
    cfg.device = device
    cfg.up_dir = args.up_dir
    cfg.title = args.title
    if cfg.dev_test:
        print("<<<< TEST MODE >>>>")
        cfg.train_batch = 1

    model = load_model(args.config_file, args.checkpoint_path)
    model = model.to(device)

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = CocoDetection(args.image_dir, args.anno_path, transforms=transform)
    data_loader = DataLoader(dataset, batch_size=cfg.train_batch, shuffle=True,
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
    parser.add_argument("--up_dir", type=str, default="")
    parser.add_argument("--title", type=str, default="")

    args = parser.parse_args()
    main(args)
