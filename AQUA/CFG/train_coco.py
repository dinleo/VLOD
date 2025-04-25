from .gdino_coco.cfg_utils import get_config
from .gdino_coco.coco_schedule import modified_coco_scheduler

branch_name = "test"
iter_per_epoch = 1000
test_batch = 1

# CFG Instance
train = get_config("gdino_coco/train.py").train
model = get_config("gdino_coco/models.py").model
dataloader = get_config("gdino_coco/dataloader.py").dataloader
optimizer = get_config("gdino_coco/optim.py").AdamW
lr_multiplier = modified_coco_scheduler(8, 4, base_steps=iter_per_epoch)

# modify train

train.name = branch_name
train.dev_test = True
train.output_dir = "./output"
train.max_iter = 8 * iter_per_epoch
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2
train.seed = 42
train.eval_period = iter_per_epoch * 8
train.checkpointer=dict(period=iter_per_epoch * 8, max_to_keep=100)
train.fast_dev_run.enabled = False

# modify model
model.ckpt = "ckpt/org_b.pth"
model.criterion = "base_criterion"

# modify dataloader config
dataloader.train.dataset.filter_empty = False
dataloader.train.num_workers = 8
dataloader.train.total_batch_size = 1
dataloader.evaluator.output_dir = train.output_dir

# modify optimizer
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1
optimizer.lr = 0.001

# ex
# tg_cat_only = True
