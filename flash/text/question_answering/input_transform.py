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
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from torch.utils.data.dataloader import default_collate

from flash.core.data.io.input import DataKeys
from flash.core.integrations.transformers.input_transform import TransformersInputTransform


@dataclass
class QuestionAnsweringInputTransform(TransformersInputTransform):
    @staticmethod
    def default_collate(samples: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        # TODO: This should be handled by the InputTransformProcessor
        chained_samples = []
        chained_metadata = []
        for s in samples:
            for sample in s:
                chained_metadata.append(sample.pop(DataKeys.METADATA, None))
                chained_samples.append(sample)

        result = default_collate(chained_samples)
        if any(m is not None for m in chained_metadata):
            result[DataKeys.METADATA] = chained_metadata
        return result

    def collate(self) -> Callable:
        return self.default_collate
