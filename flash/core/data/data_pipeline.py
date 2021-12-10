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
import inspect
import weakref
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Type, TYPE_CHECKING, Union

import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from torch.utils.data import DataLoader, IterableDataset

import flash
from flash.core.data.auto_dataset import IterableAutoDataset
from flash.core.data.batch import _DeserializeProcessor, _DeserializeProcessorV2
from flash.core.data.input_transform import _create_collate_input_transform_processors
from flash.core.data.input_transform import InputTransform as NewInputTransform
from flash.core.data.io.input import Input
from flash.core.data.io.input_base import InputBase
from flash.core.data.io.input_transform import _InputTransformProcessor, DefaultInputTransform, InputTransform
from flash.core.data.io.output import _OutputProcessor, Output
from flash.core.data.io.output_transform import _OutputTransformProcessor, OutputTransform
from flash.core.data.process import Deserializer
from flash.core.data.properties import ProcessState
from flash.core.data.utils import _INPUT_TRANSFORM_FUNCS, _OUTPUT_TRANSFORM_FUNCS, _STAGES_PREFIX
from flash.core.utilities.imports import _PL_GREATER_EQUAL_1_4_3, _PL_GREATER_EQUAL_1_5_0
from flash.core.utilities.stages import _RUNNING_STAGE_MAPPING, RunningStage

if not _PL_GREATER_EQUAL_1_5_0:
    from pytorch_lightning.trainer.connectors.data_connector import _PatchDataLoader

if TYPE_CHECKING:
    from flash.core.model import Task


class DataLoaderGetter:
    """A utility class to be used when patching the ``{stage}_dataloader`` attribute of a LightningModule."""

    def __init__(self, dataloader):
        self.dataloader = dataloader

        # Dummy `__code__` attribute to trick is_overridden
        self.__code__ = self.__call__.__code__

    def __call__(self):
        return self.dataloader


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

        self._input_transform_pipeline = input_transform or DefaultInputTransform()
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
                [input.attach_data_pipeline_state(data_pipeline_state) for input in self.input]
            else:
                self.input.attach_data_pipeline_state(data_pipeline_state)
        self._deserializer.attach_data_pipeline_state(data_pipeline_state)
        self._input_transform_pipeline.attach_data_pipeline_state(data_pipeline_state)
        self._output_transform.attach_data_pipeline_state(data_pipeline_state)
        self._output.attach_data_pipeline_state(data_pipeline_state)
        return data_pipeline_state

    @property
    def example_input(self) -> str:
        return self._deserializer.example_input

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
            != getattr(super_obj, current_method_name if super_obj == NewInputTransform else method_name).__code__
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
        has_different_code = current_code != getattr(super_obj, method_name).__code__

        if not prefix:
            return has_different_code
        return has_different_code or cls._is_overridden_recursive(method_name, process_obj, super_obj)

    @staticmethod
    def _identity(samples: Sequence[Any]) -> Sequence[Any]:
        return samples

    def deserialize_processor(self) -> _DeserializeProcessor:
        if isinstance(self._input_transform_pipeline, NewInputTransform):
            return _DeserializeProcessorV2(
                self._deserializer,
                self._input_transform_pipeline,
                self._input_transform_pipeline._per_sample_transform,
                [],
            )
        return self._create_collate_input_transform_processors(RunningStage.PREDICTING)[0]

    def worker_input_transform_processor(
        self, running_stage: RunningStage, collate_fn: Optional[Callable] = None, is_serving: bool = False
    ) -> _InputTransformProcessor:
        if isinstance(self._input_transform_pipeline, NewInputTransform):
            return _create_collate_input_transform_processors(self._input_transform_pipeline, [])[0]
        return self._create_collate_input_transform_processors(
            running_stage, collate_fn=collate_fn, is_serving=is_serving
        )[1]

    def device_input_transform_processor(self, running_stage: RunningStage) -> _InputTransformProcessor:
        if isinstance(self._input_transform_pipeline, NewInputTransform):
            return _create_collate_input_transform_processors(self._input_transform_pipeline, [])[1]
        return self._create_collate_input_transform_processors(running_stage)[2]

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

    def _make_collates(self, on_device: bool, collate: Callable) -> Tuple[Callable, Callable]:
        if on_device:
            return self._identity, collate
        return collate, self._identity

    def _create_collate_input_transform_processors(
        self,
        stage: RunningStage,
        collate_fn: Optional[Callable] = None,
        is_serving: bool = False,
    ) -> Tuple[_DeserializeProcessor, _InputTransformProcessor, _InputTransformProcessor]:

        original_collate_fn = collate_fn

        input_transform: InputTransform = self._input_transform_pipeline
        prefix: str = _STAGES_PREFIX[stage]

        if collate_fn is not None:
            input_transform._default_collate = collate_fn

        func_names: Dict[str, str] = {
            k: self._resolve_function_hierarchy(k, input_transform, stage, InputTransform)
            for k in self.INPUT_TRANSFORM_FUNCS
        }

        collate_fn: Callable = getattr(input_transform, func_names["collate"])

        per_batch_transform_overridden: bool = self._is_overridden_recursive(
            "per_batch_transform", input_transform, InputTransform, prefix=prefix
        )

        per_sample_transform_on_device_overridden: bool = self._is_overridden_recursive(
            "per_sample_transform_on_device", input_transform, InputTransform, prefix=prefix
        )

        collate_in_worker_from_transform: Optional[bool] = getattr(
            input_transform, f"_{prefix}_collate_in_worker_from_transform", None
        )

        is_per_overridden = per_batch_transform_overridden and per_sample_transform_on_device_overridden
        if collate_in_worker_from_transform is None and is_per_overridden:
            raise MisconfigurationException(
                f"{self.__class__.__name__}: `per_batch_transform` and `per_sample_transform_on_device` "
                f"are mutually exclusive for stage {stage}"
            )

        if isinstance(collate_in_worker_from_transform, bool):
            worker_collate_fn, device_collate_fn = self._make_collates(not collate_in_worker_from_transform, collate_fn)
        else:
            worker_collate_fn, device_collate_fn = self._make_collates(
                per_sample_transform_on_device_overridden, collate_fn
            )

        worker_collate_fn = (
            worker_collate_fn.collate_fn
            if isinstance(worker_collate_fn, _InputTransformProcessor)
            else worker_collate_fn
        )

        per_sample_transform = getattr(input_transform, func_names["per_sample_transform"])

        deserialize_processor = _DeserializeProcessor(
            self._deserializer,
            input_transform,
            per_sample_transform,
            callbacks=input_transform.callbacks,
        )
        worker_input_transform_processor = _InputTransformProcessor(
            input_transform,
            worker_collate_fn,
            self._identity if is_serving else per_sample_transform,
            getattr(input_transform, func_names["per_batch_transform"]),
            stage,
            callbacks=input_transform.callbacks,
        )
        worker_input_transform_processor._original_collate_fn = original_collate_fn
        device_input_transform_processor = _InputTransformProcessor(
            input_transform,
            device_collate_fn,
            getattr(input_transform, func_names["per_sample_transform_on_device"]),
            getattr(input_transform, func_names["per_batch_transform_on_device"]),
            stage,
            apply_per_sample_transform=device_collate_fn != self._identity,
            on_device=True,
            callbacks=input_transform.callbacks,
        )
        return deserialize_processor, worker_input_transform_processor, device_input_transform_processor

    @staticmethod
    def _model_transfer_to_device_wrapper(
        func: Callable, input_transform: _InputTransformProcessor, model: "Task", stage: RunningStage
    ) -> Callable:

        if not isinstance(func, _StageOrchestrator):
            func = _StageOrchestrator(func, model)
        func.register_additional_stage(stage, input_transform)

        return func

    @staticmethod
    def _model_predict_step_wrapper(
        func: Callable, output_transform_processor: _OutputTransformProcessor, model: "Task"
    ) -> Callable:

        if not isinstance(func, _StageOrchestrator):
            _original = func
            func = _StageOrchestrator(func, model)
            func._original = _original
        func.register_additional_stage(RunningStage.PREDICTING, output_transform_processor)

        return func

    @staticmethod
    def _get_dataloader(model: "Task", loader_name: str) -> Tuple[DataLoader, str]:
        dataloader, attr_name = None, None
        if is_overridden(loader_name, model):
            dataloader = getattr(model, loader_name)
            attr_name = loader_name

        elif (
            model.trainer
            and hasattr(model.trainer, "datamodule")
            and model.trainer.datamodule
            and is_overridden(loader_name, model.trainer.datamodule, flash.DataModule)
        ):
            dataloader = getattr(model.trainer.datamodule, loader_name, None)
            attr_name = f"trainer.datamodule.{loader_name}"

        elif _PL_GREATER_EQUAL_1_5_0 and model.trainer is not None:
            source = getattr(model.trainer._data_connector, f"_{loader_name}_source")
            if not source.is_module():
                dataloader = source.dataloader()
                attr_name = loader_name

                if dataloader is not None:
                    # Update source as wrapped loader will be attached to model
                    source.instance = model
                    source.name = loader_name

        return dataloader, attr_name

    @staticmethod
    def _patch_dataloader(model: "Task", dataloader: Union[Callable, DataLoader], stage: RunningStage):
        if isinstance(dataloader, DataLoader):
            if _PL_GREATER_EQUAL_1_5_0:
                dataloader = DataLoaderGetter(dataloader)
            elif _PL_GREATER_EQUAL_1_4_3:
                dataloader = _PatchDataLoader(dataloader, _STAGES_PREFIX[stage])
                dataloader.patch(model)
            else:
                dataloader = _PatchDataLoader(dataloader)
        return dataloader

    @staticmethod
    def _set_loader(model: "Task", loader_name: str, new_loader: DataLoader) -> None:
        """This function is used to set the loader to model and/or datamodule."""
        *intermediates, final_name = loader_name.split(".")
        curr_attr = model

        # This relies on python calling all non-integral types by reference.
        # It may fail for integral types since those will be called by value.
        for intermediate in intermediates:
            curr_attr = getattr(curr_attr, intermediate)

        setattr(curr_attr, final_name, new_loader)
        setattr(model, final_name, new_loader)

    def _attach_input_transform_to_model(
        self,
        model: "Task",
        stage: Optional[RunningStage] = None,
        device_transform_only: bool = False,
        is_serving: bool = False,
    ) -> None:
        device_collate_fn = torch.nn.Identity()

        if not stage:
            stages = [RunningStage.TRAINING, RunningStage.VALIDATING, RunningStage.TESTING, RunningStage.PREDICTING]

        elif isinstance(stage, RunningStage):
            stages = [stage]

        for stage in stages:

            loader_name = f"{_STAGES_PREFIX[stage]}_dataloader"

            dataloader, whole_attr_name = self._get_dataloader(model, loader_name)

            if not dataloader:
                continue

            if callable(dataloader):
                dataloader = dataloader()

            if dataloader is None:
                continue

            if isinstance(dataloader, Sequence):
                was_seq = True
            else:
                dataloader = [dataloader]
                was_seq = False

            for idx, loader in enumerate(dataloader):
                # TODO: See lightning for proper reinstantiation of loader
                if isinstance(loader, DataLoader):
                    dl_args = {k: v for k, v in vars(loader).items() if not k.startswith("_")}

                    _, dl_args["collate_fn"], device_collate_fn = self._create_collate_input_transform_processors(
                        stage=stage, collate_fn=dl_args["collate_fn"], is_serving=is_serving
                    )

                    if isinstance(dl_args["dataset"], IterableDataset):
                        del dl_args["sampler"]

                    # don't have to reinstantiate loader if just rewrapping devices (happens during detach)
                    if not device_transform_only:
                        del dl_args["batch_sampler"]
                        loader = type(loader)(**dl_args)

                dataloader[idx] = loader

            # don't have to set attribute if rewrapping device part (happens during detach)
            if not device_transform_only:
                if not was_seq:
                    dataloader = dataloader[0]

                dataloader = self._patch_dataloader(model, dataloader, stage)

                self._set_loader(model, whole_attr_name, dataloader)

            model.transfer_batch_to_device = self._model_transfer_to_device_wrapper(
                model.transfer_batch_to_device, device_collate_fn, model, stage
            )

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

    def _attach_output_transform_to_model(
        self,
        model: "Task",
        stage: RunningStage,
        is_serving: bool = False,
    ) -> "Task":
        model.predict_step = self._model_predict_step_wrapper(
            model.predict_step, self._create_output_transform_processor(stage, is_serving=is_serving), model
        )
        return model

    def _attach_to_model(
        self,
        model: "Task",
        stage: RunningStage = None,
        is_serving: bool = False,
    ):
        # not necessary to detach. preprocessing and postprocessing for stage will be overwritten.
        self._attach_input_transform_to_model(model, stage)

        if not stage or stage == RunningStage.PREDICTING:
            self._attach_output_transform_to_model(model, RunningStage.PREDICTING, is_serving=is_serving)

    def _detach_from_model(self, model: "Task", stage: Optional[RunningStage] = None):
        self._detach_input_transform_from_model(model, stage)

        if not stage or stage == RunningStage.PREDICTING:
            self._detach_output_transform_from_model(model)

    def _detach_input_transform_from_model(self, model: "Task", stage: Optional[RunningStage] = None):
        if not stage:
            stages = [RunningStage.TRAINING, RunningStage.VALIDATING, RunningStage.TESTING, RunningStage.PREDICTING]
        elif isinstance(stage, RunningStage):
            stages = [stage]

        for stage in stages:

            device_collate = None
            if isinstance(model.transfer_batch_to_device, _StageOrchestrator):
                device_collate = model.transfer_batch_to_device.unregister_stage(stage)

                # if no additional funmc available: remove wrapper
                if model.transfer_batch_to_device.is_empty():
                    model.transfer_batch_to_device = model.transfer_batch_to_device.func

            if not device_collate:
                device_collate = self._identity

            loader_name = f"{_STAGES_PREFIX[stage]}_dataloader"

            dataloader, whole_attr_name = self._get_dataloader(model, loader_name)

            if not dataloader:
                continue

            if callable(dataloader):
                dataloader = dataloader()

            if isinstance(dataloader, Sequence):
                was_seq = True
            else:
                dataloader = [dataloader]
                was_seq = False

            for idx, loader in enumerate(dataloader):
                if isinstance(loader, DataLoader):
                    dl_args = {k: v for k, v in vars(loader).items() if not k.startswith("_")}

                    # TODO: Remove the partial function once resolved on Lightning side.
                    if isinstance(dl_args["collate_fn"], partial):
                        default_collate = dl_args["collate_fn"].keywords.get("default_collate", None)
                        if default_collate:
                            dl_args["collate_fn"] = default_collate

                    if isinstance(dl_args["collate_fn"], _InputTransformProcessor):
                        dl_args["collate_fn"] = dl_args["collate_fn"]._original_collate_fn

                        if isinstance(dl_args["dataset"], (IterableAutoDataset, IterableDataset)):
                            del dl_args["sampler"]

                        del dl_args["batch_sampler"]

                        loader = type(loader)(**dl_args)

                dataloader[idx] = loader

            if not was_seq:
                dataloader = dataloader[0]

            dataloader = self._patch_dataloader(model, dataloader, stage)

            self._set_loader(model, whole_attr_name, dataloader)

    @staticmethod
    def _detach_output_transform_from_model(model: "Task"):

        if hasattr(model.predict_step, "_original"):
            # don't delete the predict_step here since we don't know
            # if any other pipeline is attached which may rely on this!
            model.predict_step = model.predict_step._original

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


class _StageOrchestrator:
    def __init__(self, func_to_wrap: Callable, model: "Task") -> None:
        self.func = func_to_wrap

        self._stage_mapping = {k: None for k in RunningStage}
        self.model = weakref.proxy(model)

        functools.update_wrapper(self, self.func)

    def __call__(self, *args, **kwargs):
        outputs = self.func(*args, **kwargs)

        try:
            stage = self.model.trainer._running_stage
        except AttributeError:
            stage = self.model.trainer.state.stage

        internal_running_state = _RUNNING_STAGE_MAPPING[stage]
        additional_func = self._stage_mapping.get(internal_running_state, None)

        if additional_func:
            outputs = additional_func(outputs)

        return outputs

    def register_additional_stage(self, stage: RunningStage, stage_func: Optional[Callable] = None):
        assert stage_func is None or callable(stage_func)

        self._stage_mapping[stage] = stage_func.to(self.model.device, self.model.dtype)

    def unregister_stage(self, stage: RunningStage):
        ret_val = self._stage_mapping.pop(stage)
        self._stage_mapping[stage] = None
        if ret_val:
            ret_val = ret_val.cpu()
        return ret_val

    def is_empty(self):
        return all(v is None for v in self._stage_mapping.values()) or not self._stage_mapping
