import datetime
from .gdino_coco.cfg_utils import get_config
from .gdino_coco.coco_schedule import modified_coco_scheduler
from .gdino_coco.dataloader import register_coco_subset


date_str = datetime.datetime.now().strftime("%m%d_%H%M")

branch_name = "test"
iter_per_epoch = 1000
train_batch = 1
test_sample = 100

# CFG Instance
train = get_config("gdino_coco/train.py").train
model = get_config("gdino_coco/models.py").model
dataloader = get_config("gdino_coco/dataloader.py").dataloader
optimizer = get_config("gdino_coco/optim.py").AdamW
lr_multiplier = modified_coco_scheduler(8, 4, base_steps=iter_per_epoch)

# modify train
train.name = branch_name
train.dev_test = True
train.output_dir = f"./output/{branch_name}_{date_str}"
train.max_iter = 8 * iter_per_epoch
train.eval_period = train.max_iter
train.eval_sample = test_sample
train.checkpointer=dict(period=iter_per_epoch * 8, max_to_keep=100)

# modify model
model.ckpt = "ckpt/org_b.pth"
model.criterion = "base_criterion"

# modify dataloader config
dataloader.evaluator.output_dir = train.output_dir
dataloader.train.dataset.filter_empty = False
dataloader.train.num_workers = 8
dataloader.train.total_batch_size = train_batch
register_coco_subset("test", test_sample)

# modify optimizer
optimizer.lr = 0.00001
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# ex
# tg_cat_only = True
