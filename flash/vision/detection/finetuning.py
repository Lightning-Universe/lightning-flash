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
import pytorch_lightning as pl

from flash.core.finetuning import FlashBaseFinetuning


class ObjectDetectionFineTuning(FlashBaseFinetuning):
    """
    Freezes the backbone during Detector training.
    """

    def __init__(self, train_bn: bool = True):
        self.train_bn = train_bn

    def freeze_before_training(self, pl_module: pl.LightningModule) -> None:
        model = pl_module.model
        self.freeze(modules=model.backbone, train_bn=self.train_bn)
