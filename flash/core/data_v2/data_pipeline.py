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
import inspect
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

from pytorch_lightning.utilities.exceptions import MisconfigurationException

from flash.core.data.data_pipeline import DataPipelineState
from flash.core.data_v2.io.input import InputsStateContainer
from flash.core.data_v2.io.output import Output
from flash.core.data_v2.transforms.input_transform import InputTransform
from flash.core.data_v2.transforms.output_transform import OutputTransform
from flash.core.registry import FlashRegistry
from flash.core.utilities.stages import RunningStage


@dataclass
class InputContainer:

    args: Any
    kwargs: Any
    train_state: Dict
    train_input_transform: Optional[InputTransform]


class DataPipeline:
    """DataPipeline holds the engineering logic to connect input, input_transform, model, output_transform, output
    together."""

    def __init__(
        self,
        flash_datasets_registry: Optional[FlashRegistry] = None,
        inputs_state: Optional[InputsStateContainer] = None,
        output_transform: Optional[OutputTransform] = None,
        output: Optional[Output] = None,
    ) -> None:

        self._flash_datasets_registry = flash_datasets_registry
        self._inputs_state = inputs_state
        self._output_transform = output_transform
        self._output = output

    def initialize(self, data_pipeline_state: Optional[DataPipelineState] = None) -> DataPipelineState:
        """Creates the :class:`.DataPipelineState` and gives the reference to the: :class:`.Preprocess`,
        :class:`.Postprocess`, and :class:`.Serializer`. Once this has been called, any attempt to add new state will
        give a warning."""
        data_pipeline_state = data_pipeline_state or DataPipelineState()
        data_pipeline_state._initialized = False
        if self._inputs_state:
            self._inputs_state.attach_data_pipeline_state(data_pipeline_state)
        data_pipeline_state._initialized = True  # TODO: Not sure we need this
        breakpoint()
        return data_pipeline_state

    @property
    def example_input(self) -> str:
        return self._flash_dataset.example_input

    @staticmethod
    def _is_overriden(method_name: str, process_obj, super_obj: Any, prefix: Optional[str] = None) -> bool:
        """Cropped Version of https://github.com/PyTorchLightning/pytorch-
        lightning/blob/master/pytorch_lightning/utilities/model_helpers.py."""

        current_method_name = method_name if prefix is None else f"{prefix}_{method_name}"

        if not hasattr(process_obj, current_method_name):
            return False

        return getattr(process_obj, current_method_name).__code__ != getattr(super_obj, method_name).__code__

    @classmethod
    def _is_overriden_recursive(
        cls, method_name: str, process_obj, super_obj: Any, prefix: Optional[str] = None
    ) -> bool:
        """Cropped Version of https://github.com/PyTorchLightning/pytorch-
        lightning/blob/master/pytorch_lightning/utilities/model_helpers.py."""
        assert isinstance(process_obj, super_obj)
        if prefix is None and not hasattr(super_obj, method_name):
            raise MisconfigurationException(f"This function doesn't belong to the parent class {super_obj}")

        current_method_name = method_name if prefix is None else f"{prefix}_{method_name}"

        if not hasattr(process_obj, current_method_name):
            return DataPipeline._is_overriden_recursive(method_name, process_obj, super_obj)

        current_code = inspect.unwrap(getattr(process_obj, current_method_name)).__code__
        has_different_code = current_code != getattr(super_obj, method_name).__code__

        if not prefix:
            return has_different_code
        return has_different_code or cls._is_overriden_recursive(method_name, process_obj, super_obj)

    @classmethod
    def _resolve_function_hierarchy(
        cls, function_name, process_obj, stage: RunningStage, object_type: Optional[Type]
    ) -> str:
        prefixes = []

        if stage in (RunningStage.TRAINING, RunningStage.TUNING):
            prefixes += ["train", "fit"]
        elif stage == RunningStage.VALIDATING:
            prefixes += ["val", "fit"]
        elif stage == RunningStage.TESTING:
            prefixes += ["test"]
        elif stage == RunningStage.PREDICTING:
            prefixes += ["predict"]
        elif stage == RunningStage.SERVING:
            prefixes += ["serve"]

        prefixes += [None]

        for prefix in prefixes:
            if cls._is_overriden(function_name, process_obj, object_type, prefix=prefix):
                return function_name if prefix is None else f"{prefix}_{function_name}"

        return function_name
