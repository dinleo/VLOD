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
dataloader = get_config("gdino_coco/dataloader_.py").dataloader
model = get_config("gdino_coco/models_.py").model
runner = get_config("gdino_coco/runner_.py").runner
solver = get_config("gdino_coco/solver_.py").solver

# modify dataloader
dataloader.train.dataset.filter_empty = False
dataloader.train.num_workers = 8
dataloader.train.total_batch_size = 1
dataloader.evaluator.output_dir = runner.output_dir
register_coco_subset("test", test_sample)

# modify model
model.ckpt = "inputs/ckpt/org_b.pth"

# modify runner
runner.name = branch_name
runner.eval_only = False
runner.dev_test = dev_test
runner.output_dir = f"./outputs/{branch_name}_{date_str}"
runner.max_iter = epoch * iter_per_epoch
runner.eval_period = runner.max_iter
runner.eval_sample = test_sample
runner.checkpointer=dict(period=iter_per_epoch * epoch, max_to_keep=100)

# modify solver
solver.optimizer.lr = 0.00001
solver.optimizer.weight_decay = 1e-4
solver.optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1
solver.lr_scheduler = modified_coco_scheduler(epoch, epoch//2, base_steps=iter_per_epoch)

# ex
# tg_cat_only = True
