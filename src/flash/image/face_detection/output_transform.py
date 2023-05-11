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
from typing import Any

from flash.core.data.io.input import DataKeys
from flash.core.data.io.output_transform import OutputTransform
from flash.core.utilities.imports import _FASTFACE_AVAILABLE

if _FASTFACE_AVAILABLE:
    import fastface as ff


class FaceDetectionOutputTransform(OutputTransform):
    """Generates preds from model output."""

    @staticmethod
    def per_batch_transform(batch: Any) -> Any:
        scales = batch["scales"]
        paddings = batch["paddings"]

        batch.pop("scales", None)
        batch.pop("paddings", None)

        preds = batch[DataKeys.PREDS]

        # preds: list of Tensor(N, 5) as x1, y1, x2, y2, score
        preds = [preds[preds[:, 5] == batch_idx, :5] for batch_idx in range(len(preds))]
        preds = ff.utils.preprocess.adjust_results(preds, scales, paddings)

        return preds
