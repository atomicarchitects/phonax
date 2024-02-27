from typing import Callable, Dict, Optional

import optax

from .amsgrad import scale_by_amsgrad
from .utils import flatten_dict, unflatten_dict


def exponential_decay(
    lr: float,
    steps_per_interval: int,
    *,
    transition_steps: float = 0.0,
    decay_rate: float = 0.5,
    transition_begin: float = 0.0,
    staircase: bool = True,
    end_value: Optional[float] = None,
):
    return optax.exponential_decay(
        init_value=lr,
        transition_steps=transition_steps * steps_per_interval,
        decay_rate=decay_rate,
        transition_begin=transition_begin * steps_per_interval,
        staircase=staircase,
        end_value=end_value,
    )


def piecewise_constant_schedule(
    lr: float, steps_per_interval: int, *, boundaries_and_scales: Dict[float, float]
):
    boundaries_and_scales = {
        boundary * steps_per_interval: scale
        for boundary, scale in boundaries_and_scales.items()
    }
    return optax.piecewise_constant_schedule(
        init_value=lr, boundaries_and_scales=boundaries_and_scales
    )


def constant_schedule(lr, steps_per_interval):
    return optax.constant_schedule(lr)


def optimizer(
    steps_per_interval: int,
    max_num_intervals: int,
    weight_decay=0.0,
    lr=0.01,
    algorithm: Callable = scale_by_amsgrad,
    scheduler: Callable = constant_schedule,
):
    def weight_decay_mask(params):
        # print('gin function params',params)
        params = flatten_dict(params)
        mask = {
            k: any(("linear_down" in ki) or ("symmetric_contraction" in ki) for ki in k)
            for k in params
        }
        # assert any(any(("linear_down" in ki) for ki in k) for k in params)
        # assert any(any(("symmetric_contraction" in ki) for ki in k) for k in params)
        return unflatten_dict(mask)

    return (
        optax.chain(
            optax.add_decayed_weights(weight_decay, mask=weight_decay_mask),
            algorithm(),
            optax.scale_by_schedule(scheduler(lr, steps_per_interval)),
            optax.scale(-1.0),  # Gradient descent.
        ),
        steps_per_interval,
        max_num_intervals,
    )
