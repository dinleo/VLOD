#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging
import os
import sys
import time
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
from detectron2.utils.events import (
    CommonMetricPrinter, 
    JSONWriter, 
    TensorboardXWriter
)
from detectron2.checkpoint import DetectionCheckpointer

from models.groundingdino import get_model

from models.groundingdino.util import ema
from models.groundingdino.util.events import WandbWriter
from models.groundingdino.util.utils import clean_state_dict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from criterion import get_criterion


class Trainer(SimpleTrainer):
    """
    We've combine Simple and AMP Trainer together.
    """

    def __init__(
        self,
        model,
        criterion,
        dataloader,
        optimizer,
        amp=False,
        clip_grad_params=None,
        grad_scaler=None,
        batch_size_scale=1,
    ):
        super().__init__(model=model, data_loader=dataloader, optimizer=optimizer)

        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported

        if amp:
            if grad_scaler is None:
                from torch.cuda.amp import GradScaler

                grad_scaler = GradScaler()
        self.grad_scaler = grad_scaler
        
        # set True to use amp training
        self.amp = amp

        # gradient clip hyper-params
        self.clip_grad_params = clip_grad_params
        
        # batch_size_scale
        self.batch_size_scale = batch_size_scale

        # criterion
        self.criterion = criterion

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[Trainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[Trainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        inputs = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        with autocast(enabled=self.amp):
            outputs = self.model(inputs)
            targets = outputs["targets"]
            loss_dict = self.criterion(outputs, targets)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """

        if self.amp:
            self.grad_scaler.scale(losses).backward()
            if self.clip_grad_params is not None:
                self.grad_scaler.unscale_(self.optimizer)
                self.clip_grads(self.model.parameters())
            if self.iter % self.batch_size_scale == 0:
                # print(self.iter)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                self.optimizer.zero_grad()
        else:
            losses.backward()
            if self.clip_grad_params is not None:
                self.clip_grads(self.model.parameters())
            if self.iter % self.batch_size_scale == 0:
                # print(self.iter)
                self.optimizer.step()
                self.optimizer.zero_grad()

        self._write_metrics(loss_dict, data_time)

    def clip_grads(self, params):
        params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return torch.nn.utils.clip_grad_norm_(
                parameters=params,
                **self.clip_grad_params,
            )

    def state_dict(self):
        ret = super().state_dict()
        if self.grad_scaler and self.amp:
            ret["grad_scaler"] = self.grad_scaler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if self.grad_scaler and self.amp:
            self.grad_scaler.load_state_dict(state_dict["grad_scaler"])


def load_model(model_config, model_checkpoint_path):
    model_config.device = "cuda"
    model = get_model(model_config)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    # _ = model.eval()
    return model

def load_criterion(model_config):
    model_config.device = "cuda"
    criterion = get_criterion(model_config)

    return criterion


def do_test(cfg, model, eval_only=False):
    logger = logging.getLogger("detectron2")
    test_loader = instantiate(cfg.dataloader.test)
    evaluator = instantiate(cfg.dataloader.evaluator)
    if eval_only:
        logger.info("Run evaluation under eval-only mode")
        if cfg.train.model_ema.enabled and cfg.train.model_ema.use_ema_weights_for_eval_only:
            logger.info("Run evaluation with EMA.")
        else:
            logger.info("Run evaluation without EMA.")
        if "evaluator" in cfg.dataloader:
            ret = inference_on_dataset(
                model, test_loader, evaluator
            )
            print_csv_format(ret)
        return ret
    
    logger.info("Run evaluation without EMA.")
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, test_loader, evaluator
        )
        print_csv_format(ret)

        if cfg.train.model_ema.enabled:
            logger.info("Run evaluation with EMA.")
            with ema.apply_model_ema_and_restore(model):
                if "evaluator" in cfg.dataloader:
                    ema_ret = inference_on_dataset(
                        model, test_loader, evaluator
                    )
                    print_csv_format(ema_ret)
                    ret.update(ema_ret)
        return ret


def do_train(cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    # model = instantiate(cfg.model)
    model_cfg = cfg.model  # change the path of the model config file
    checkpoint_path = model_cfg.ckpt  # change the path of the model
    model = load_model(model_cfg, checkpoint_path)
    # model.set_tsk_id(args.tsk_id)
    # logger = logging.getLogger("detectron2")
    # logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)
    # model.add_cls_prompt(cfg.dataloader.train.mapper.categories_names)

    criterion = load_criterion(model_cfg)
    criterion.to(cfg.train.device)

    # instantiate optimizer
    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    # build training loader
    train_loader = instantiate(cfg.dataloader.train)
    
    # create ddp model
    model = create_ddp_model(model, **cfg.train.ddp)

    # build model ema
    ema.may_build_model_ema(cfg, model)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        dataloader=train_loader,
        optimizer=optim,
        amp=cfg.train.amp.enabled,
        clip_grad_params=cfg.train.clip_grad.params if cfg.train.clip_grad.enabled else None,
    )
    
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
        # save model ema
        **ema.may_get_ema_checkpointer(cfg, model)
    )

    if comm.is_main_process():
        # writers = default_writers(cfg.train.output_dir, cfg.train.max_iter)
        output_dir = cfg.train.output_dir
        PathManager.mkdirs(output_dir)
        writers = [
            CommonMetricPrinter(cfg.train.max_iter),
            JSONWriter(os.path.join(output_dir, "metrics.json")),
            TensorboardXWriter(output_dir),
        ]
        if not cfg.train.dev_test:
            writers.append(WandbWriter(cfg))

    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            ema.EMAHook(cfg, model) if cfg.train.model_ema.enabled else None,
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter(
                writers,
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)
    
    # Enable fast debugging by running several iterations to check for any bugs.
    if cfg.train.dev_test:
        cfg.dataloader.train.total_batch_size = 1
        cfg.train.max_iter = 20
        cfg.train.eval_period = 10
        cfg.train.log_period = 1
    if 0 < cfg.train.eval_sample:
        cfg.dataloader.test.dataset.names = "test_n"
    # args.eval_only = True
    if args.eval_only:
        model_cfg = cfg.model  # change the path of the model config file
        checkpoint_path = model_cfg.ckpt  # change the path of the model
        model = load_model(model_cfg, checkpoint_path)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        
        # using ema for evaluation
        ema.may_build_model_ema(cfg, model)
        DetectionCheckpointer(model, **ema.may_get_ema_checkpointer(cfg, model)).load(cfg.train.init_checkpoint)
        # Apply ema state for evaluation
        if cfg.train.model_ema.enabled and cfg.train.model_ema.use_ema_weights_for_eval_only:
            ema.apply_model_ema(model)
        print(do_test(cfg, model, eval_only=True))
    else:
        do_train(cfg)


if __name__ == "__main__":
    parser = default_argument_parser()
    # parser.add_argument("--tsk-id", type=int, required=True, help="task id")
    args = parser.parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
