from CFG.cfg_utils import get_config
import os
data_dir = os.getenv("DATA", "inputs/data")
ckpt_dir = os.getenv("CKPT", "inputs/ckpt")
branch_name = os.getenv("BRANCH", "test")

# Base cfg
project_name = "stage1" # Cur Subdir
iter_per_epoch = 1000
epoch = 8
batch = 8
eval_sample = 1000 # If set -1, evaluate all testdata
dev_test = False # If set True, Enable fast debugging(batch=1, max_iter=200)


# CFG Instance
dataloader = get_config(f"{project_name}/dataloader_.py").dataloader
model = get_config(f"{project_name}/models_.py").model
runner = get_config(f"{project_name}/runner_.py").runner
solver = get_config(f"{project_name}/solver_.py").solver


# modify dataloader
dataloader.train.num_workers = 8
dataloader.train.total_batch_size = batch
dataloader.train.dataset.filter_empty = True
dataloader.train.dataset.image_root = f"{data_dir}/train2017"
dataloader.train.dataset.json_file = f"{data_dir}/annotations/instances_train2017.json"
dataloader.test.dataset.image_root = f"{data_dir}/val2017"
dataloader.test.dataset.json_file = f"{data_dir}/annotations/instances_val2017.json"
dataloader.test_sub.dataset.n = eval_sample


# modify model
model.build.args.ckpt_path = ""
model.build.args.aqua.args.ckpt_path = ""
model.build.args.aqua.args.blip_ckpt_path = f"{ckpt_dir}/blip2.pth"
model.build.args.detr_backbone.args.ckpt_path = f"{ckpt_dir}/gdino.pth"


# modify solver
solver.optimizer.lr = 0.001
solver.optimizer.weight_decay = 0.01
solver.lr_scheduler.epochs=epoch
solver.lr_scheduler.decay_epochs=epoch//2
solver.lr_scheduler.base_steps=iter_per_epoch


# modify runner
runner.name = f"{project_name}/{branch_name}" # Usage: output_dir_postfix, wandb_name
runner.device = "cuda"
runner.output_dir = f"./outputs/{runner.name}"
runner.dev_test = dev_test # only iterate 200 & no wandb logging
runner.max_iter = epoch * iter_per_epoch

# eval
runner.do_eval = False
runner.eval_only = False
runner.eval_period = iter_per_epoch
runner.eval_sample = eval_sample


# logging config
runner.checkpointer=dict(period=iter_per_epoch, max_to_keep=100)
runner.wandb.entity = "dinleo11"
runner.wandb.project = "Aqua"
