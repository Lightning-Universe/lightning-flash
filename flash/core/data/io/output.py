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
from typing import Any, Mapping

import torch

import flash
from flash.core.data.properties import Properties
from flash.core.data.utils import convert_to_modules


class Output(Properties):
    """An :class:`.Output` encapsulates a single :meth:`~flash.core.data.io.output.Output.transform` method which
    is used to convert the model output into the desired output format when predicting."""

    def __init__(self):
        super().__init__()
        self._is_enabled = True

    def enable(self):
        """Enable output transformation."""
        self._is_enabled = True

    def disable(self):
        """Disable output transformation."""
        self._is_enabled = False

    @staticmethod
    def transform(sample: Any) -> Any:
        """Convert the given sample into the desired output format.

        Args:
            sample: The output from the :class:`.OutputTransform`.

        Returns:
            The converted output.
        """
        return sample

    def __call__(self, sample: Any) -> Any:
        if self._is_enabled:
            return self.transform(sample)
        return sample


class OutputMapping(Output):
    """If the model output is a dictionary, then the :class:`.OutputMapping` enables each entry in the dictionary
    to be passed to it's own :class:`.Output`."""

    def __init__(self, outputs: Mapping[str, Output]):
        super().__init__()

        self._outputs = outputs

    def transform(self, sample: Any) -> Any:
        if isinstance(sample, Mapping):
            return {key: output.transform(sample[key]) for key, output in self._outputs.items()}
        raise ValueError("The model output must be a mapping when using an OutputMapping.")

    def attach_data_pipeline_state(self, data_pipeline_state: "flash.core.data.data_pipeline.DataPipelineState"):
        for output in self._outputs.values():
            output.attach_data_pipeline_state(data_pipeline_state)


class _OutputProcessor(torch.nn.Module):
    def __init__(
        self,
        output: "Output",
    ):
        super().__init__()
        self.output = convert_to_modules(output)

    def forward(self, sample):
        return self.output(sample)
