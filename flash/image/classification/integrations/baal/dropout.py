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
from pytorch_lightning.utilities.exceptions import MisconfigurationException

import flash
from flash.core.utilities.imports import _BAAL_AVAILABLE

if _BAAL_AVAILABLE:
    from baal.bayesian.dropout import _patch_dropout_layers


class InferenceMCDropoutTask(flash.Task):
    def __init__(self, module: flash.Task, inference_iteration: int):
        super().__init__()
        self.parent_module = module
        self.trainer = module.trainer
        changed = _patch_dropout_layers(self.parent_module)
        if not changed:
            raise MisconfigurationException("The model should contain at least 1 dropout layer.")
        self.inference_iteration = inference_iteration

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        with torch.no_grad():
            out = []
            for _ in range(self.inference_iteration):
                out.append(self.parent_module.predict_step(batch, batch_idx, dataloader_idx=dataloader_idx)["preds"])

        # BaaL expects a shape [num_samples, num_classes, num_iterations]
        return torch.stack(out).permute((1, 2, 0))
