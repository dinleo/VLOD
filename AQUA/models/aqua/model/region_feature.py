# region_feature.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.rpn import AnchorGenerator, RegionProposalNetwork
from torchvision.ops import RoIAlign


class RegionFeature(nn.Module):
    def __init__(self, 
                 anchor_sizes=((32,), (64,), (128,), (256,)),
                 aspect_ratios=((0.5, 1.0, 2.0),) * 5,
                 pre_nms_top_n_train=2000,
                 post_nms_top_n_train=1000,
                 pre_nms_top_n_test=1000,
                 post_nms_top_n_test=300,
                 nms_thresh=0.7,
                 roi_output_size=7,
                 sampling_ratio=2,
                 output_dim=1408):
        super().__init__()

        self.anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios,
        )

        self.rpn = RegionProposalNetwork(
            self.anchor_generator,
            head=None,  # default head is applied
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n={"training": pre_nms_top_n_train, "testing": pre_nms_top_n_test},
            post_nms_top_n={"training": post_nms_top_n_train, "testing": post_nms_top_n_test},
            nms_thresh=nms_thresh,
        )

        self.roi_align = RoIAlign(
            output_size=(roi_output_size, roi_output_size),
            spatial_scale=1.0,  # to be set dynamically based on feature map
            sampling_ratio=sampling_ratio,
            aligned=True,
        )

        self.fc = nn.Linear(roi_output_size * roi_output_size * 256, output_dim)  # assuming input C=256 backbone

    def forward(self, features, image_sizes, targets=None):
        """
        Args:
            features: List of backbone feature maps (multi-scale)
            image_sizes: Original image sizes
            targets: (Optional) ground truth boxes during training
        Returns:
            region_features: (B, N, output_dim)
        """
        # features is a list of feature maps
        feature_maps = {f"{i}": feat for i, feat in enumerate(features)}
        proposals, _ = self.rpn(feature_maps, image_sizes, targets)

        # RoIAlign expects list of feature maps -> picking the last scale feature for now
        last_feature = features[-1]
        B, C, H, W = last_feature.shape
        spatial_scale = float(H) / image_sizes[0][0]  # height ratio
        self.roi_align.spatial_scale = spatial_scale

        # generate batch indices for proposals
        proposal_boxes = []
        batch_indices = []
        for batch_idx, boxes in enumerate(proposals):
            proposal_boxes.append(boxes)
            batch_indices.append(torch.full((boxes.shape[0], 1), batch_idx, device=boxes.device))
        proposal_boxes = torch.cat(proposal_boxes, dim=0)
        batch_indices = torch.cat(batch_indices, dim=0)
        roi_boxes = torch.cat([batch_indices, proposal_boxes], dim=1)  # (idx, x1, y1, x2, y2)

        roi_feats = self.roi_align(last_feature, roi_boxes)
        roi_feats = roi_feats.flatten(start_dim=1)
        roi_feats = self.fc(roi_feats)

        return roi_feats
