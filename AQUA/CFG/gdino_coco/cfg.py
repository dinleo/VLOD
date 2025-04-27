import datetime
date_str = datetime.datetime.now().strftime("%m%d_%H%M")
from CFG.cfg_utils import get_config
from CFG.gdino_coco.dataloader_ import register_coco_subset
from CFG.gdino_coco.solver_ import modified_coco_scheduler

# Train cfg
branch_name = "test"
iter_per_epoch = 1000
epoch = 8
test_sample = 100 # Test all data if set -1
dev_test = True

# CFG Instance
train = get_config("gdino_coco/train_.py").train
model = get_config("gdino_coco/models_.py").model
dataloader = get_config("gdino_coco/dataloader_.py").dataloader
solver = get_config("gdino_coco/solver_.py").solver

# modify train
train.name = branch_name
train.dev_test = dev_test
train.output_dir = f"./outputs/{branch_name}_{date_str}"
train.max_iter = epoch * iter_per_epoch
train.eval_period = train.max_iter
train.eval_sample = test_sample
train.checkpointer=dict(period=iter_per_epoch * epoch, max_to_keep=100)

# modify model
model.ckpt = "inputs/ckpt/org_b.pth"

# modify dataloader config
dataloader.train.dataset.filter_empty = False
dataloader.train.num_workers = 8
dataloader.train.total_batch_size = 1
dataloader.evaluator.output_dir = train.output_dir
register_coco_subset("test", test_sample)

# modify optimizer
solver.optimizer.lr = 0.00001
solver.optimizer.weight_decay = 1e-4
solver.optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1
solver.lr_scheduler = modified_coco_scheduler(epoch, epoch//2, base_steps=iter_per_epoch)

# ex
# tg_cat_only = True
