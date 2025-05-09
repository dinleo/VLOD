# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import move_device_like
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer

from models.aqua.model.region.find_topk import find_top_rpn_proposals
from models.aqua.util.box_ops import box_cxcywh_to_xyxy
from models.model_utils import safe_init
from torchvision.ops import box_iou

class RegionQueryGenerator(nn.Module):
    def __init__(
        self,
        rpn_yaml_path=None,
        nms_iou_thresh: float=0.1,
        gt_iou_thresh: float=0.5,
        pre_nms_topk: int=256,
        post_nms_topk: int=256,
        min_box_size: float=10,
    ):
        """
        Args:
            rpn_yaml_path (str): Path to RPN model config file (.yaml). If None, precomputed boxes must be provided.
            nms_iou_thresh (float): IoU threshold for Non-Maximum Suppression (NMS).
            gt_iou_thresh (float): Minimum IoU threshold for a proposal to be matched to a ground truth box.
            pre_nms_topk (int): Number of top-scoring proposals to keep per feature level before NMS.
            post_nms_topk (int): Total number of proposals to keep per image after NMS (across all levels).
            min_box_size (float): Minimum box size in pixels. Boxes smaller than this are discarded.
        """
        super().__init__()
        self.rpn = None
        self.nms_iou_thresh = nms_iou_thresh
        self.gt_iou_thresh = gt_iou_thresh
        self.pre_nms_topk = pre_nms_topk
        self.post_nms_topk = post_nms_topk
        self.min_box_size = min_box_size
        # record
        self.gt_prop_iou_mean = 0
        self.gt_count = 0
        self.discard_gt_count = 0

        if rpn_yaml_path:
            cfg = get_cfg()
            cfg.merge_from_file(rpn_yaml_path)
            model = GeneralizedRCNN(cfg)
            if cfg.MODEL.WEIGHTS:
                DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
            self.rpn = model

    def forward(self, dict_input):
        """
        Args:
            dict_input:
                - gt_instances: List of Instances with fields:
                    - gt_boxes: (cx, cy, w, h), un-normalized
                    - gt_classes: int labels (0-based)
                    - image_size: (H, W)
                - image_features: required if self.rpn is used
        Returns:
            dict:
                - nms_prob: (B, K) scores after NMS; ≥100 means GT, class = score - 100
                - nms_boxes: (B, K, 4) boxes after NMS; (x1, y1, x2, y2), normalized scale
                - nms_index: (B, K) index of original proposal in [0, 899]
                - gt_labels: (B, K) int labels, -1 for non-GT
        """
        gt_instances = dict_input['gt_instances']
        image_sizes = [i.image_size for i in gt_instances]

        if self.rpn:
            # image_features -> [Scale, Batch, (Boxes, Logit)]
            rpn_output = self.rpn(dict_input['image_features'])
        else:
            rpn_output = dict_input

        # logits (Scale, Batch, Proposal) must be raw logit. not sigmoid prob
        # boxes (Scale, Batch, Proposal, 4) must be normalized cxcywh coordinate
        pred_logits = rpn_output['multiscale_pred_logits']
        pred_boxes = rpn_output['multiscale_pred_boxes']
        device = pred_logits[0].device
        dtype = pred_logits[0].dtype

        # Make prob
        multiscale_prob = []
        for scale, pred_logits_s in enumerate(pred_logits):
            pred = pred_logits_s.sigmoid()
            if pred.dim() == 3:
                # DETR class agnostic output
                pred = pred.mean(dim=2)
            multiscale_prob.append(pred)

        # Make boxes
        multiscale_proposal = []
        scales = torch.tensor(image_sizes, device=device, dtype=dtype)
        scales = torch.stack([scales[:, 1], scales[:, 0], scales[:, 1], scales[:, 0]], dim=1)  # (B, 4)
        for scale, pred_boxes_s in enumerate(pred_boxes):
            proposal = box_cxcywh_to_xyxy(pred_boxes_s)  # (B, N, 4)
            proposal *= scales[:, None, :]

            multiscale_proposal.append(proposal)

        # Replace GT score injection (proposal matching instead of GT insertion)
        for b, gt_ins in enumerate(gt_instances):
            gt_b = gt_ins.gt_boxes.tensor.to(device)  # (M, 4)
            cls_b = gt_ins.gt_classes.to(device)  # (M,)
            prop_b = multiscale_proposal[-1][b]  # (N, 4)
            prob_b = multiscale_prob[-1][b]  # (N,)

            if gt_b.numel() == 0:
                continue

            ious = box_iou(gt_b, prop_b)
            max_ious, max_idx = ious.max(dim=1)  # max over proposals (per GT)
            iou_sum = max_ious.sum()

            # print(f"{max_ious.mean()} {max_ious}")
            self.gt_prop_iou_mean = (self.gt_prop_iou_mean * self.gt_count) + iou_sum
            self.gt_count += len(max_ious)
            self.gt_prop_iou_mean /= self.gt_count

            for i in range(gt_b.size(0)):
                if self.gt_iou_thresh < max_ious[i]:
                    prob_b[max_idx[i]] = cls_b[i].float() + 100
                else:
                    # print(f"discard {max_ious[i]}")
                    self.discard_gt_count += 1

        # NMS
        nms_output = find_top_rpn_proposals(multiscale_proposal, multiscale_prob, image_sizes, self.nms_iou_thresh, self.pre_nms_topk, self.post_nms_topk, self.min_box_size, training=False)
        nms_prob = nms_output['nms_prob']
        nms_boxes = nms_output['nms_boxes']
        nms_index = nms_output['nms_index']
        for batch in range(len(nms_boxes)):
            nms_boxes[batch] /= scales[batch][None, :]

        # nms_prob: shape (B, K)
        gt_labels = []
        for p in nms_prob:
            gt_label = torch.where(
                p >= 100,
                (p - 100).long(),
                torch.full_like(p, -1, dtype=torch.long)
            )
            gt_labels.append(gt_label)

        output = {
            'nms_prob': nms_prob,
            'nms_boxes': nms_boxes,
            'nms_index': nms_index,
            'gt_labels': gt_labels,
        }
        return output

    def print_gt_matching(self):
        print(f"GT and Top-Proposal IOU Mean: {self.gt_prop_iou_mean}")
        print(f"Total GT Count: {self.gt_count}")
        print(f"Discarded GT Count: {self.discard_gt_count}")
        print(f"Discarded Ratio: {self.discard_gt_count / self.gt_count}")

class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float]=[123.675, 116.280, 103.530],
        pixel_std: Tuple[float]=[123.675, 116.280, 103.530],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def visualize_training(self, batched_input, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_input (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_input and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_input, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, input):
        """
        Args:
            batched_input: a list, batched output of :class:`DatasetMapper` .
                Each item in the list contains the input for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        # if not self.training:
        #     return self.inference(batched_input)
        batched_input = input['batched_input']
        dino_images = input['images']
        dino_gt_instances = input['gt_instances']
        multiscale_features = input['multiscale_features']

        images = self.preprocess_image(batched_input)
        if "instances" in batched_input[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_input]
        else:
            gt_instances = None
        features = self.backbone(images.tensor)

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_input[0]
            proposals = [x["proposals"].to(self.device) for x in batched_input]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_input, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(
        self,
        batched_input: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given input.

        Args:
            batched_input (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI output.
            do_postprocess (bool): whether to apply post-processing on the output.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network output.
        """
        assert not self.training

        images = self.preprocess_image(batched_input)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_input[0]
                proposals = [x["proposals"].to(self.device) for x in batched_input]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_input, images.image_sizes)
        return results

    def preprocess_image(self, batched_input: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [self._move_to_current_device(x["image"]) for x in batched_input]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images

    @staticmethod
    def _postprocess(instances, batched_input: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_input, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results


class ProposalNetwork(nn.Module):
    """
    A meta architecture that only predicts object proposals.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def forward(self, batched_input):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [self._move_to_current_device(x["image"]) for x in batched_input]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        features = self.backbone(images.tensor)

        if "instances" in batched_input[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_input]
        elif "targets" in batched_input[0]:
            log_first_n(
                logging.WARN, "'targets' in the model input is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_input]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_input, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results

def build_region_query_generator(args):
    model = safe_init(RegionQueryGenerator, args)

    return model
