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
import torch

import flash
from flash.core.utilities.imports import _BAAL_AVAILABLE, _BAAL_GREATER_EQUAL_1_5_2

if _BAAL_AVAILABLE:
    # _patch_dropout_layers function was replaced with replace_layers_in_module helper
    # function in v1.5.2 (https://github.com/ElementAI/baal/pull/194 for more details)
    if _BAAL_GREATER_EQUAL_1_5_2:
        from baal.bayesian.common import replace_layers_in_module
        from baal.bayesian.consistent_dropout import _consistent_dropout_mapping_fn

        def _patch_dropout_layers(module: torch.nn.Module):
            return replace_layers_in_module(module, _consistent_dropout_mapping_fn)

    else:
        from baal.bayesian.consistent_dropout import _patch_dropout_layers


class InferenceMCDropoutTask(flash.Task):
    def __init__(self, module: flash.Task, inference_iteration: int):
        super().__init__()
        self.parent_module = module
        self.trainer = module.trainer
        changed = _patch_dropout_layers(self.parent_module)
        if not changed:
            raise TypeError("The model should contain at least 1 dropout layer.")
        self.inference_iteration = inference_iteration

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        with torch.no_grad():
            out = []
            for _ in range(self.inference_iteration):
                out.append(self.parent_module.predict_step(batch, batch_idx, dataloader_idx=dataloader_idx)["preds"])

        # BaaL expects a shape [num_samples, num_classes, num_iterations]
        return torch.stack(out).permute((1, 2, 0))
