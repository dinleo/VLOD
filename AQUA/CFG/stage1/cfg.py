from CFG.cfg_utils import get_config

# Base cfg
project_name = "stage1" # Cur Subdir
branch_name = "test"
iter_per_epoch = 1000
epoch = 8
eval_sample = 100 # If set -1, evaluate all testdata
dev_test = True # If set True, Enable fast debugging(batch=1, max_iter=200)


# CFG Instance
dataloader = get_config(f"{project_name}/dataloader_.py").dataloader
model = get_config(f"{project_name}/models_.py").model
runner = get_config(f"{project_name}/runner_.py").runner
solver = get_config(f"{project_name}/solver_.py").solver


# modify dataloader
dataloader.train.dataset.filter_empty = False
dataloader.train.num_workers = 8
dataloader.train.total_batch_size = 1
dataloader.test_sub.dataset.n = eval_sample


# modify model
model.build.args.ckpt_path = ""
model.build.args.aqua.args.ckpt_path = ""
model.build.args.aqua.args.blip_ckpt_path = "inputs/ckpt/blip2.pth"
model.build.args.backbone.args.ckpt_path = "inputs/ckpt/org_b.pth"
# model.build.args.aqua.args.region_query_generator.args.rpn_yaml_path \
#     = f"CFG/{project_name}/faster_rcnn_R_50_FPN_3x.yaml"


# modify runner
runner.name = f"{project_name}/{branch_name}" # Usage: output_dir_postfix, wandb_name
runner.device = "cuda"
runner.eval_only = False
runner.dev_test = dev_test # only iterate 200 & no wandb logging
runner.output_dir = f"./outputs"
runner.max_iter = epoch * iter_per_epoch
runner.eval_period = runner.max_iter
runner.eval_sample = eval_sample
runner.checkpointer=dict(period=iter_per_epoch * epoch, max_to_keep=100)
runner.wandb.entity = "dinleo11"
runner.wandb.project = "vlod"


# modify solver
solver.optimizer.lr = 0.00001
solver.optimizer.weight_decay = 1e-4
solver.optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1
solver.lr_scheduler.epochs=epoch
solver.lr_scheduler.decay_epochs=epoch//2
solver.lr_scheduler.base_steps=iter_per_epoch
