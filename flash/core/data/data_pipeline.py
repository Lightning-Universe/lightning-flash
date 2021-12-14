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
from typing import Any, Dict, List, Optional, Set, Type, Union

from pytorch_lightning.utilities.exceptions import MisconfigurationException

from flash.core.data.io.input import Input, InputBase
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.io.output import _OutputProcessor, Output
from flash.core.data.io.output_transform import _OutputTransformProcessor, OutputTransform
from flash.core.data.process import Deserializer
from flash.core.data.properties import ProcessState
from flash.core.data.utils import _INPUT_TRANSFORM_FUNCS, _OUTPUT_TRANSFORM_FUNCS
from flash.core.utilities.stages import RunningStage


class DataPipelineState:
    """A class to store and share all process states once a :class:`.DataPipeline` has been initialized."""

    def __init__(self):
        self._state: Dict[Type[ProcessState], ProcessState] = {}

    def set_state(self, state: ProcessState):
        """Add the given :class:`.ProcessState` to the :class:`.DataPipelineState`."""

        self._state[type(state)] = state

    def get_state(self, state_type: Type[ProcessState]) -> Optional[ProcessState]:
        """Get the :class:`.ProcessState` of the given type from the :class:`.DataPipelineState`."""

        return self._state.get(state_type, None)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(state={self._state})"


class DataPipeline:
    """
    DataPipeline holds the engineering logic to connect
    :class:`~flash.core.data.io.input_transform.InputTransform` and/or
    :class:`~flash.core.data.io.output_transform.OutputTransform`
    objects to the ``DataModule``, Flash ``Task`` and ``Trainer``.
    """

    INPUT_TRANSFORM_FUNCS: Set[str] = _INPUT_TRANSFORM_FUNCS
    OUTPUT_TRANSFORM_FUNCS: Set[str] = _OUTPUT_TRANSFORM_FUNCS

    def __init__(
        self,
        input: Optional[Union[Input, List[InputBase]]] = None,
        input_transform: Optional[InputTransform] = None,
        output_transform: Optional[OutputTransform] = None,
        deserializer: Optional[Deserializer] = None,
        output: Optional[Output] = None,
    ) -> None:
        self.input = input

        self._input_transform_pipeline = input_transform or InputTransform(RunningStage.TRAINING)
        self._output_transform = output_transform or OutputTransform()
        self._output = output or Output()
        self._deserializer = deserializer or Deserializer()
        self._running_stage = None

    def initialize(self, data_pipeline_state: Optional[DataPipelineState] = None) -> DataPipelineState:
        """Creates the :class:`.DataPipelineState` and gives the reference to the: :class:`.InputTransform`,
        :class:`.OutputTransform`, and :class:`.Output`. Once this has been called, any attempt to add new state will
        give a warning."""
        data_pipeline_state = data_pipeline_state or DataPipelineState()
        if self.input is not None:
            if isinstance(self.input, list):
                for input in self.input:
                    if hasattr(input, "attach_data_pipeline_state"):
                        input.attach_data_pipeline_state(data_pipeline_state)
            else:
                self.input.attach_data_pipeline_state(data_pipeline_state)
        self._deserializer.attach_data_pipeline_state(data_pipeline_state)
        self._input_transform_pipeline.attach_data_pipeline_state(data_pipeline_state)
        self._output_transform.attach_data_pipeline_state(data_pipeline_state)
        self._output.attach_data_pipeline_state(data_pipeline_state)
        return data_pipeline_state

    @staticmethod
    def _is_overridden(method_name: str, process_obj, super_obj: Any, prefix: Optional[str] = None) -> bool:
        """Cropped Version of https://github.com/PyTorchLightning/pytorch-
        lightning/blob/master/pytorch_lightning/utilities/model_helpers.py."""

        current_method_name = method_name if prefix is None else f"{prefix}_{method_name}"

        if not hasattr(process_obj, current_method_name):
            return False

        # TODO: With the new API, all hooks are implemented to improve discoverability.
        return (
            getattr(process_obj, current_method_name).__code__
            != getattr(super_obj, current_method_name if super_obj == InputTransform else method_name).__code__
        )

    @classmethod
    def _is_overridden_recursive(
        cls, method_name: str, process_obj, super_obj: Any, prefix: Optional[str] = None
    ) -> bool:
        """Cropped Version of https://github.com/PyTorchLightning/pytorch-
        lightning/blob/master/pytorch_lightning/utilities/model_helpers.py."""
        assert isinstance(process_obj, super_obj), (process_obj, super_obj)
        if prefix is None and not hasattr(super_obj, method_name):
            raise MisconfigurationException(f"This function doesn't belong to the parent class {super_obj}")

        current_method_name = method_name if prefix is None else f"{prefix}_{method_name}"

        if not hasattr(process_obj, current_method_name):
            return DataPipeline._is_overridden_recursive(method_name, process_obj, super_obj)

        current_code = inspect.unwrap(getattr(process_obj, current_method_name)).__code__
        has_different_code = current_code != getattr(super_obj, current_method_name).__code__

        if not prefix:
            return has_different_code
        return has_different_code or cls._is_overridden_recursive(method_name, process_obj, super_obj)

    def output_transform_processor(self, running_stage: RunningStage, is_serving=False) -> _OutputTransformProcessor:
        return self._create_output_transform_processor(running_stage, is_serving=is_serving)

    def output_processor(self) -> _OutputProcessor:
        return _OutputProcessor(self._output)

    @classmethod
    def _resolve_function_hierarchy(
        cls, function_name, process_obj, stage: RunningStage, object_type: Optional[Type] = None
    ) -> str:
        if object_type is None:
            object_type = InputTransform

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
            if cls._is_overridden(function_name, process_obj, object_type, prefix=prefix):
                return function_name if prefix is None else f"{prefix}_{function_name}"

        return function_name

    def _create_output_transform_processor(
        self,
        stage: RunningStage,
        is_serving: bool = False,
    ) -> _OutputTransformProcessor:
        output_transform: OutputTransform = self._output_transform

        func_names: Dict[str, str] = {
            k: self._resolve_function_hierarchy(k, output_transform, stage, object_type=OutputTransform)
            for k in self.OUTPUT_TRANSFORM_FUNCS
        }

        return _OutputTransformProcessor(
            getattr(output_transform, func_names["uncollate"]),
            getattr(output_transform, func_names["per_batch_transform"]),
            getattr(output_transform, func_names["per_sample_transform"]),
            output=None if is_serving else self._output,
            is_serving=is_serving,
        )

    def __str__(self) -> str:
        input: Input = self.input
        input_transform: InputTransform = self._input_transform_pipeline
        output_transform: OutputTransform = self._output_transform
        output: Output = self._output
        deserializer: Deserializer = self._deserializer
        return (
            f"{self.__class__.__name__}("
            f"input={str(input)}, "
            f"deserializer={deserializer}, "
            f"input_transform={input_transform}, "
            f"output_transform={output_transform}, "
            f"output={output})"
        )
