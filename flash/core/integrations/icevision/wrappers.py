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
from functools import partial

import torch

from flash.core.data.utils import _STAGES_PREFIX
from flash.core.utilities.imports import _ICEVISION_AVAILABLE

if _ICEVISION_AVAILABLE:
    from icevision.models.ross.efficientdet.lightning.model_adapter import ModelAdapter as EffDetModelAdapter


def _log_with_name_and_prog_bar_override(log, adapter, name, value, **kwargs):
    if "prog_bar" not in kwargs:
        kwargs["prog_bar"] = True
    metric = name.split("/")[-1]
    metric = f"{_STAGES_PREFIX[adapter.trainer.state.stage]}_{metric}"
    return log(metric, value, **kwargs)


def _effdet_validation_step(validation_step, batch, batch_idx):
    images = batch[0][0]
    batch[0][1]["img_scale"] = torch.ones_like(images[:, 0, 0, 0]).unsqueeze(1)
    batch[0][1]["img_size"] = (torch.ones_like(images[:, 0, 0, 0]) * images[0].shape[-1]).unsqueeze(1).repeat(1, 2)
    return validation_step(batch, batch_idx)


def wrap_icevision_adapter(adapter):
    if not isinstance(adapter.log, partial):
        adapter.log = partial(_log_with_name_and_prog_bar_override, adapter.log, adapter)

    if isinstance(adapter, EffDetModelAdapter) and not isinstance(adapter.validation_step, partial):
        adapter.validation_step = partial(_effdet_validation_step, adapter.validation_step)
    return adapter
