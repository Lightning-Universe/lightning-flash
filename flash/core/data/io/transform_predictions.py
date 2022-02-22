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
import functools

import pytorch_lightning as pl
from pytorch_lightning import Callback

from flash.core.data.io.output import Output
from flash.core.data.io.output_transform import OutputTransform


class TransformPredictions(Callback):
    """``TransformPredictions`` is a :class:`~pytorch_lightning.callbacks.base.Callback` which can be used to apply an
    :class:`~flash.core.data.io.output_transform.OutputTransform` and an :class:`~flash.core.data.io.output.Output` to
    model predictions.

    Args:
        output_transform: The :class:`~flash.core.data.io.output_transform.OutputTransform` to apply.
        output: The :class:`~flash.core.data.io.output.Output` to apply.
    """

    def __init__(self, output_transform: OutputTransform, output: Output):
        super().__init__()

        self.output_transform = output_transform
        self.output = output

    def on_predict_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        predict_step = pl_module.predict_step

        @functools.wraps(predict_step)
        def wrapper(*args, **kwargs):
            predictions = predict_step(*args, **kwargs)
            if predictions is not None:
                predictions = self.output_transform(predictions)
                predictions = [self.output(prediction) for prediction in predictions]
            return predictions

        pl_module.predict_step = wrapper

    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pl_module.predict_step = pl_module.predict_step.__wrapped__
