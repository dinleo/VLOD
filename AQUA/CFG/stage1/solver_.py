from omegaconf import OmegaConf

from detectron2.config import LazyCall as L
from detectron2.solver.build import get_default_optimizer_params

from solver.criterion.stage1_criterion import Stage1Criterion
from solver.optimizer.scheduler import modified_coco_scheduler
import torch

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

solver.criterion = L(Stage1Criterion)(
    align_weight=1,
    contrast_weight=1,
    magnitude_weight=1
)

solver.lr_scheduler = L(modified_coco_scheduler)(
    epochs=4,
    decay_epochs=1,
    warmup_epochs=0,
    base_steps=5000,
)