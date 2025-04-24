from CFG.coco.cfg_utils import get_config
from CFG.coco.coco_schedule import modified_coco_scheduler

# Train
iter_per_epoch = 1000
train = get_config("coco/train.py").train
optimizer = get_config("coco/optim.py").AdamW
lr_multiplier = modified_coco_scheduler(8, 4, base_steps=iter_per_epoch)
model = get_config("coco/models.py").model
dataloader = get_config("coco/dataloader.py").dataloader

# Model
model.ckpt = "ckpt/org_b.pth"

# modify training config
train.output_dir = "./output"
train.max_iter = 8 * iter_per_epoch
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2
train.seed = 42
train.eval_period = iter_per_epoch * 8
train.checkpointer=dict(period=iter_per_epoch * 8, max_to_keep=100)
train.fast_dev_run.enabled = False

# modify optimizer config
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1
optimizer.lr = 0.001

# modify dataloader config
dataloader.train.dataset.filter_empty = False
dataloader.train.num_workers = 8
dataloader.train.total_batch_size = 1

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir

train.name = "test"
train.dev_test = True
# ex
tg_cat_only = True

test_batch = 1
