from omegaconf import OmegaConf

from detectron2.config import LazyCall as L
from detectron2.solver.build import get_default_optimizer_params

from solver.criterion.base_criterion import *
from solver.optimizer.scheduler import modified_coco_scheduler

solver = OmegaConf.create()

solver.optimizer = L(torch.optim.AdamW)(
    params=L(get_default_optimizer_params)(
        # params.model is meant to be set to the model object, before instantiating
        # the optimizer.
        base_lr="${..lr}",
        weight_decay_norm=0.0,
    ),
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.1,
)

solver.criterion = L(BaseCriterion)(
    matcher=L(HungarianMatcher)(
        cost_class=L(FocalLossCost)(
            alpha=0.25,
            gamma=2.0,
            weight=1.0,
        ),
        cost_bbox=L(L1Cost)(weight=5.0),
        cost_giou=L(GIoUCost)(weight=1.0),
    ),
    loss_class=L(FocalLoss)(
        alpha=0.25,
        gamma=2.0,
        loss_weight=1.0,
    ),
    loss_bbox=L(L1Loss)(loss_weight=5.0),
    loss_giou=L(GIoULoss)(
        eps=1e-6,
        loss_weight=1.0,
    ),
)