from omegaconf import OmegaConf

runner = OmegaConf.create()
runner.update(
    name="",
    seed=42,
    # Directory where output files are written to
    output_dir="./output",
    # The initialize checkpoint to be loaded
    init_checkpoint="",
    # The total training iterations
    max_iter=90000,
    # options for Automatic Mixed Precision
    amp=dict(enabled=False),
    # options for DistributedDataParallel
    ddp=dict(
        broadcast_buffers=False,
        find_unused_parameters=True,
        fp16_compression=False,
    ),
    # options for Gradient Clipping during training
    clip_grad=dict(
        enabled=True,
        params=dict(
            max_norm=0.1,
            norm_type=2,
        ),
    ),
    # options for Fast Debugging
    dev_test=False,
    # options for PeriodicCheckpointer, which saves a model checkpoint
    # after every `checkpointer.period` iterations,
    # and only `checkpointer.max_to_keep` number of checkpoint will be kept.
    checkpointer=dict(period=1000, max_to_keep=100),
    # run evaluation after every `eval_period` number of iterations
    eval_period=1000,
    # output log to console every `log_period` number of iterations.
    log_period=20,
    # logging training info to Wandb
    # note that you should add wandb writer in `train_net.py``
    wandb=dict(
        entity="",
        project="",
        name=""
    ),
    # model ema
    model_ema=dict(
        enabled=False,
        decay=0.999,
        device="",
        use_ema_weights_for_eval_only=False,
    ),
    # the training device, choose from {"cuda", "cpu"}
    device="cuda",
    # ...
)
