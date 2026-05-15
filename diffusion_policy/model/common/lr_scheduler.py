from typing import Union, Optional
from enum import Enum
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import math


class SchedulerType(str, Enum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"


def _get_constant_schedule_lambda(last_epoch=-1):
    return lambda _: 1


def _get_constant_schedule_with_warmup_lambda(num_warmup_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0
    return lr_lambda


def _get_linear_schedule_with_warmup_lambda(num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    return lr_lambda


def _get_cosine_schedule_with_warmup_lambda(num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return lr_lambda


def _get_cosine_with_hard_restarts_schedule_with_warmup_lambda(num_warmup_steps, num_training_steps, num_cycles=1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))
    return lr_lambda


def _get_polynomial_decay_schedule_with_warmup_lambda(num_warmup_steps, num_training_steps, lr_end, power, lr_init):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if current_step > num_training_steps:
            return lr_end / lr_init
        pct_remaining = 1 - (current_step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
        decay = (lr_init - lr_end) * pct_remaining ** power + lr_end
        return decay / lr_init
    return lr_lambda


def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    **kwargs
):
    """
    Unified API to get any scheduler from its name.
    """
    name = SchedulerType(name)

    if name == SchedulerType.CONSTANT:
        return LambdaLR(optimizer, _get_constant_schedule_lambda(), **kwargs)

    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return LambdaLR(optimizer, _get_constant_schedule_with_warmup_lambda(num_warmup_steps), **kwargs)

    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    if name == SchedulerType.LINEAR:
        return LambdaLR(optimizer, _get_linear_schedule_with_warmup_lambda(num_warmup_steps, num_training_steps), **kwargs)

    if name == SchedulerType.COSINE:
        return LambdaLR(optimizer, _get_cosine_schedule_with_warmup_lambda(num_warmup_steps, num_training_steps), **kwargs)

    if name == SchedulerType.COSINE_WITH_RESTARTS:
        return LambdaLR(optimizer, _get_cosine_with_hard_restarts_schedule_with_warmup_lambda(num_warmup_steps, num_training_steps), **kwargs)

    if name == SchedulerType.POLYNOMIAL:
        lr_init = optimizer.defaults["lr"]
        lr_end = kwargs.pop("lr_end", 0)
        power = kwargs.pop("power", 1.0)
        return LambdaLR(optimizer, _get_polynomial_decay_schedule_with_warmup_lambda(num_warmup_steps, num_training_steps, lr_end, power, lr_init), **kwargs)

    raise ValueError(f"Unknown scheduler type: {name}")
