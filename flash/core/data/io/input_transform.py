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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

import flash
from flash.core.data.callback import ControlFlow, FlashCallback
from flash.core.data.io.input import DataKeys
from flash.core.data.utils import convert_to_modules
from flash.core.utilities.stages import RunningStage


class _InputTransformProcessorV2(torch.nn.Module):
    """
    This class is used to encapsulate the following functions of a InputTransformInputTransform Object:
    Inside a worker:
        per_sample_transform: Function to transform an individual sample
        collate: Function to merge sample into a batch
        per_batch_transform: Function to transform an individual batch

    Inside main process:
        per_sample_transform_on_device: Function to transform an individual sample
        collate: Function to merge sample into a batch
        per_batch_transform_on_device: Function to transform an individual batch
    """

    def __init__(
        self,
        input_transform: "flash.core.data.input_transform.InputTransform",
        collate_fn: Callable,
        per_sample_transform: Callable,
        per_batch_transform: Callable,
        stage: RunningStage,
        apply_per_sample_transform: bool = True,
        on_device: bool = False,
        callbacks: Optional[List[FlashCallback]] = None,
    ):
        super().__init__()
        self.input_transform = input_transform
        self.callback = ControlFlow(callbacks or [])
        self.collate_fn = convert_to_modules(collate_fn)
        self.per_sample_transform = convert_to_modules(per_sample_transform)
        self.per_batch_transform = convert_to_modules(per_batch_transform)
        self.apply_per_sample_transform = apply_per_sample_transform
        self.stage = stage
        self.on_device = on_device

    @staticmethod
    def _extract_metadata(
        samples: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
        metadata = [s.pop(DataKeys.METADATA, None) if isinstance(s, Mapping) else None for s in samples]
        return samples, metadata if any(m is not None for m in metadata) else None

    def forward(self, samples: Sequence[Any]) -> Any:
        if not self.on_device:
            for sample in samples:
                self.callback.on_load_sample(sample, self.stage)

        # we create a new dict to prevent from potential memory leaks
        # assuming that the dictionary samples are stored in between and
        # potentially modified before the transforms are applied.
        if isinstance(samples, dict):
            samples = dict(samples.items())

        if self.apply_per_sample_transform:
            _samples = []

            if isinstance(samples, Mapping):
                samples = [samples]

            for sample in samples:
                sample = self.per_sample_transform(sample)
                if self.on_device:
                    self.callback.on_per_sample_transform_on_device(sample, self.stage)
                else:
                    self.callback.on_per_sample_transform(sample, self.stage)
                _samples.append(sample)

            samples = type(_samples)(_samples)

            samples, metadata = self._extract_metadata(samples)
            try:
                samples = self.collate_fn(samples, metadata)
            except TypeError:
                samples = self.collate_fn(samples)
            if metadata and isinstance(samples, dict):
                samples[DataKeys.METADATA] = metadata
            self.callback.on_collate(samples, self.stage)

        samples = self.per_batch_transform(samples)
        if self.on_device:
            self.callback.on_per_batch_transform_on_device(samples, self.stage)
        else:
            self.callback.on_per_batch_transform(samples, self.stage)
        return samples

    def __str__(self) -> str:
        # todo: define repr function which would take object and string attributes to be shown
        return (
            "_InputTransformProcessor:\n"
            f"\t(per_sample_transform): {str(self.per_sample_transform)}\n"
            f"\t(collate_fn): {str(self.collate_fn)}\n"
            f"\t(per_batch_transform): {str(self.per_batch_transform)}\n"
            f"\t(apply_per_sample_transform): {str(self.apply_per_sample_transform)}\n"
            f"\t(on_device): {str(self.on_device)}\n"
            f"\t(stage): {str(self.stage)}"
        )
