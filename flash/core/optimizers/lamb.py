# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# Implemented by @ananyahjha93
# also found at: https://github.com/gridai-labs/aavae/tree/main/src/optimizers
# References:
#     - https://arxiv.org/pdf/1904.00962.pdf
#     - https://github.com/pytorch/pytorch/blob/1.6/torch/optim/adam.py
import math
from typing import Tuple

import torch
from torch import nn
from torch.optim.optimizer import Optimizer

from flash.core.utilities.imports import _CORE_TESTING

# Skip doctests if requirements aren't available
if not _CORE_TESTING:
    __doctest_skip__ = ["LAMB"]


class LAMB(Optimizer):
    r"""Extends ADAM in pytorch to incorporate LAMB algorithm from the paper:
    `Large batch optimization for deep learning: Training BERT in 76 minutes <https://arxiv.org/pdf/1904.00962.pdf>`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        exclude_from_layer_adaptation (bool, optional): layers which do not need LAMB
            layer adaptation (default: False)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond <https://arxiv.org/pdf/1904.09237.pdf>`_
            (default: False)

    Example:
        >>> model = nn.Linear(10, 1)
        >>> optimizer = LAMB(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> # loss_fn(model(input), target).backward()
        >>> optimizer.step()

    .. warning::
        Since the default weight decay for LAMB is set to 0., we do not club together
        0. weight decay and exclusion from layer adaptation like LARS. This would cause
        the optimizer to exclude all layers from layer adaptation.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0,
        exclude_from_layer_adaptation: bool = False,
        amsgrad: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            exclude_from_layer_adaptation=exclude_from_layer_adaptation,
            amsgrad=amsgrad,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("LAMB does not support sparse gradients")
                amsgrad = group["amsgrad"]
                exclude_from_layer_adaptation = group["exclude_from_layer_adaptation"]

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])

                numerator = exp_avg / bias_correction1
                update = numerator / denom

                if group["weight_decay"] != 0:
                    update = update.add(p.data, alpha=group["weight_decay"])

                trust_ratio = 1.0
                if not exclude_from_layer_adaptation:
                    w_norm = torch.norm(p.data)
                    g_norm = torch.norm(update)

                    if w_norm > 0 and g_norm > 0:
                        trust_ratio = w_norm / g_norm

                p.add_(update, alpha=-group["lr"] * trust_ratio)

        return loss
