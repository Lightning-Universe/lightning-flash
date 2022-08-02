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
from typing import List, Union

from torch import Tensor

from flash.core.registry import FlashRegistry
from flash.core.utilities.imports import _VISSL_AVAILABLE

if _VISSL_AVAILABLE:
    import vissl.losses  # noqa: F401
    from classy_vision.generic.distributed_util import set_cpu_device
    from classy_vision.losses import ClassyLoss, LOSS_REGISTRY
    from vissl.config.attr_dict import AttrDict
else:
    AttrDict = object
    ClassyLoss = object


def _recursive_register(module):
    named_tensors = [(key, value) for key, value in module.__dict__.items() if isinstance(value, Tensor)]
    for name, tensor in named_tensors:
        delattr(module, name)
        module.register_buffer(name, tensor)

    for child_module in module.modules():
        if child_module is not module:
            _recursive_register(child_module)


def get_loss_fn(loss_name: str, cfg: AttrDict):
    set_cpu_device()
    loss_fn = LOSS_REGISTRY[loss_name](cfg)
    loss_fn.__dict__["loss_name"] = loss_name

    _recursive_register(loss_fn)
    return loss_fn


def swav_loss(
    embedding_dim: int = 128,
    temperature: float = 0.1,
    use_double_precision: bool = False,
    normalize_last_layer: bool = True,
    num_iters: int = 3,
    epsilon: float = 0.05,
    num_crops: int = 8,
    crops_for_assign: List[int] = [0, 1],
    num_prototypes: Union[int, List[int]] = 3000,
    temp_hard_assignment_iters: int = 0,
    output_dir: str = ".",
    queue_length: int = 0,
    start_iter: int = 0,
    local_queue_length: int = 0,
    **kwargs,
) -> ClassyLoss:
    loss_name = "swav_loss"
    cfg = AttrDict(
        {
            "embedding_dim": embedding_dim,
            "temperature": temperature,
            "use_double_precision": use_double_precision,
            "normalize_last_layer": normalize_last_layer,
            "num_iters": num_iters,
            "epsilon": epsilon,
            "num_crops": num_crops,
            "crops_for_assign": crops_for_assign,
            "num_prototypes": [num_prototypes] if isinstance(num_prototypes, int) else num_prototypes,
            "temp_hard_assignment_iters": temp_hard_assignment_iters,
            "output_dir": output_dir,
            "queue": AttrDict(
                {
                    "queue_length": queue_length,
                    "start_iter": start_iter,
                    "local_queue_length": local_queue_length,
                }
            ),
        }
    )

    return get_loss_fn(loss_name, cfg)


def barlow_twins_loss(
    lambda_: float = 0.0051,
    scale_loss: float = 0.024,
    latent_embedding_dim: int = 8192,
    **kwargs,
) -> ClassyLoss:
    loss_name = "barlow_twins_loss"
    cfg = AttrDict(
        {
            "lambda_": lambda_,
            "scale_loss": scale_loss,
            "embedding_dim": latent_embedding_dim,
        }
    )

    return get_loss_fn(loss_name, cfg)


def simclr_loss(
    temperature: float = 0.1,
    embedding_dim: int = 128,
    effective_batch_size: int = 1,  # set by setup training hook
    world_size: int = 1,  # set by setup training hook
    **kwargs,
) -> ClassyLoss:
    loss_name = "simclr_info_nce_loss"
    cfg = AttrDict(
        {
            "temperature": temperature,
            "buffer_params": AttrDict(
                {
                    "world_size": world_size,
                    "embedding_dim": embedding_dim,
                    "effective_batch_size": effective_batch_size,
                }
            ),
        }
    )

    return get_loss_fn(loss_name, cfg)


def register_vissl_losses(register: FlashRegistry):
    for loss_fn in (swav_loss, barlow_twins_loss, simclr_loss):
        register(loss_fn)
