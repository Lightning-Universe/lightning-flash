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
from typing import Any, Optional, Type

from pytorch_lightning.utilities.exceptions import MisconfigurationException

from flash.core.data.io.input_transform import InputTransform
from flash.core.utilities.stages import RunningStage


class DataPipeline:
    """
    DataPipeline holds the engineering logic to connect
    :class:`~flash.core.data.io.input_transform.InputTransform` and/or
    :class:`~flash.core.data.io.output_transform.OutputTransform`
    objects to the ``DataModule``, Flash ``Task`` and ``Trainer``.
    """

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
