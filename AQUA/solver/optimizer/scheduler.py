from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import MultiStepParamScheduler

def default_X_scheduler(num_X):
    """
    Returns the config for a default multi-step LR scheduler such as "1x", "3x",
    commonly referred to in papers, where every 1x has the total length of 1440k
    training images (~12 COCO epochs). LR is decayed twice at the end of training
    following the strategy defined in "Rethinking ImageNet Pretraining", Sec 4.
    Args:
        num_X: a positive real number
    Returns:
        DictConfig: configs that define the multiplier for LR during training
    """
    # total number of iterations assuming 16 batch size, using 1440000/16=90000
    total_steps_16bs = num_X * 90000

    if num_X <= 2:
        scheduler = MultiStepParamScheduler(
            values=[1.0, 0.1, 0.01],
            # note that scheduler is scale-invariant. This is equivalent to
            # milestones=[6, 8, 9]
            milestones=[60000, 80000, 90000],
        )
    else:
        scheduler = MultiStepParamScheduler(
            values=[1.0, 0.1, 0.01],
            milestones=[total_steps_16bs - 60000, total_steps_16bs - 20000, total_steps_16bs],
        )
    return WarmupParamScheduler(
        scheduler=scheduler,
        warmup_length=1000 / total_steps_16bs,
        warmup_method="linear",
        warmup_factor=0.001,
    )


def default_coco_scheduler(epochs=50, decay_epochs=40, warmup_epochs=0):
    """
    Returns the config for a default multi-step LR scheduler such as "50epochs",
    commonly referred to in papers, where every 1x has the total length of 1440k
    training images (~12 COCO epochs). LR is decayed once at the end of training.

    Args:
        epochs (int): total training epochs.
        decay_epochs (int): lr decay steps.
        warmup_epochs (int): warmup epochs.

    Returns:
        DictConfig: configs that define the multiplier for LR during training
    """
    # total number of iterations assuming 16 batch size, using 1440000/16=90000
    total_steps_16bs = epochs * 7500
    decay_steps = decay_epochs * 7500
    warmup_steps = warmup_epochs * 7500
    scheduler = MultiStepParamScheduler(
        values=[1.0, 0.1],
        milestones=[decay_steps, total_steps_16bs],
    )
    return WarmupParamScheduler(
        scheduler=scheduler,
        warmup_length=warmup_steps / total_steps_16bs,
        warmup_method="linear",
        warmup_factor=0.001,
    )


# default coco scheduler
lr_multiplier_1x = default_X_scheduler(1)
lr_multiplier_2x = default_X_scheduler(2)
lr_multiplier_3x = default_X_scheduler(3)
lr_multiplier_6x = default_X_scheduler(6)
lr_multiplier_9x = default_X_scheduler(9)

# default scheduler for detr
lr_multiplier_50ep = default_coco_scheduler(50, 40, 0)
lr_multiplier_36ep = default_coco_scheduler(36, 30, 0)
lr_multiplier_24ep = default_coco_scheduler(24, 20, 0)
lr_multiplier_12ep = default_coco_scheduler(12, 11, 0)

# warmup scheduler for detr
lr_multiplier_50ep_warmup = default_coco_scheduler(50, 40, 1e-3)
lr_multiplier_12ep_warmup = default_coco_scheduler(12, 11, 1e-3)


#
def modified_coco_scheduler(epochs=4, decay_epochs=1, warmup_epochs=0, base_steps=5000):
    """
    Returns the config for a default multi-step LR scheduler such as "50epochs",
    commonly referred to in papers, where every 1x has the total length of 1440k
    training images (~12 COCO epochs). LR is decayed once at the end of training.

    Args:
        epochs (int): total training epochs.
        decay_epochs (int): lr decay steps.
        warmup_epochs (int): warmup epochs.

    Returns:
        DictConfig: configs that define the multiplier for LR during training
    """
    # total number of iterations assuming 16 batch size, using 1440000/16=90000
    total_steps_16bs = epochs * base_steps
    decay_steps = decay_epochs * base_steps
    warmup_steps = warmup_epochs * base_steps

    if decay_steps >= total_steps_16bs:
        scheduler = MultiStepParamScheduler(
            values=[1.0],
            milestones=[total_steps_16bs],
        )
    else:
        scheduler = MultiStepParamScheduler(
            values=[1.0, 0.1],
            milestones=[decay_steps, total_steps_16bs],
        )
    return WarmupParamScheduler(
        scheduler=scheduler,
        warmup_length=warmup_steps / total_steps_16bs,
        warmup_method="linear",
        warmup_factor=0.001,
    )


def modified_voc_scheduler(total_epochs=4, decay_epochs1=1, decay_epochs2=2, warmup_epochs=0, base_steps=5000):
    """
    Returns the config for a default multi-step LR scheduler such as "50epochs",
    commonly referred to in papers, where every 1x has the total length of 1440k
    training images (~12 COCO epochs). LR is decayed once at the end of training.

    Args:
        epochs (int): total training epochs.
        decay_epochs (int): lr decay steps.
        warmup_epochs (int): warmup epochs.

    Returns:
        DictConfig: configs that define the multiplier for LR during training
    """
    # total number of iterations assuming 16 batch size, using 1440000/16=90000
    total_steps_16bs = total_epochs * base_steps
    decay_steps = decay_epochs1 * base_steps

    warmup_steps = warmup_epochs * base_steps
    scheduler = MultiStepParamScheduler(
        values=[1.0, 0.1, 0.01],
        milestones=[decay_steps, decay_epochs2 * base_steps, total_steps_16bs],
    )
    return WarmupParamScheduler(
        scheduler=scheduler,
        warmup_length=warmup_steps / total_steps_16bs,
        warmup_method="linear",
        warmup_factor=0.001,
    )