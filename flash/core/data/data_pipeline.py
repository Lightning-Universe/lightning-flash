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
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple, Type, TYPE_CHECKING, Union

import torch
from pytorch_lightning.trainer.connectors.data_connector import _PatchDataLoader
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from torch.utils.data import DataLoader, IterableDataset

from flash.core.data.auto_dataset import IterableAutoDataset
from flash.core.data.batch import _DeserializeProcessor, _Postprocessor, _Preprocessor, _Sequential, _SerializeProcessor
from flash.core.data.data_source import DataSource
from flash.core.data.process import DefaultPreprocess, Deserializer, Postprocess, Preprocess, Serializer
from flash.core.data.properties import ProcessState
from flash.core.data.utils import _POSTPROCESS_FUNCS, _PREPROCESS_FUNCS, _STAGES_PREFIX
from flash.core.utilities.imports import _PL_GREATER_EQUAL_1_4_3

if TYPE_CHECKING:
    from flash.core.model import Task


class DataPipelineState:
    """A class to store and share all process states once a :class:`.DataPipeline` has been initialized."""

    def __init__(self):
        self._state: Dict[Type[ProcessState], ProcessState] = {}
        self._initialized = False

    def set_state(self, state: ProcessState):
        """Add the given :class:`.ProcessState` to the :class:`.DataPipelineState`."""

        if not self._initialized:
            self._state[type(state)] = state
        else:
            rank_zero_warn(
                f"Attempted to add a state ({state}) after the data pipeline has already been initialized. This will"
                " only have an effect when a new data pipeline is created.",
                UserWarning,
            )

    def get_state(self, state_type: Type[ProcessState]) -> Optional[ProcessState]:
        """Get the :class:`.ProcessState` of the given type from the :class:`.DataPipelineState`."""

        if state_type in self._state:
            return self._state[state_type]
        return None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(initialized={self._initialized}, state={self._state})"


class DataPipeline:
    """
    DataPipeline holds the engineering logic to connect
    :class:`~flash.core.data.process.Preprocess` and/or :class:`~flash.core.data.process.Postprocess` objects to
    the ``DataModule``, Flash ``Task`` and ``Trainer``.

    Example::

        class CustomPreprocess(Preprocess):
            pass

        class CustomPostprocess(Postprocess):
            pass

        custom_data_pipeline = DataPipeline(CustomPreprocess(), CustomPostprocess())

        # And it can attached to both the datamodule and model.

        datamodule.data_pipeline = custom_data_pipeline
        model.data_pipeline = custom_data_pipeline
    """

    PREPROCESS_FUNCS: Set[str] = _PREPROCESS_FUNCS
    POSTPROCESS_FUNCS: Set[str] = _POSTPROCESS_FUNCS

    def __init__(
        self,
        data_source: Optional[DataSource] = None,
        preprocess: Optional[Preprocess] = None,
        postprocess: Optional[Postprocess] = None,
        deserializer: Optional[Deserializer] = None,
        serializer: Optional[Serializer] = None,
    ) -> None:
        self.data_source = data_source

        self._preprocess_pipeline = preprocess or DefaultPreprocess()
        self._postprocess_pipeline = postprocess or Postprocess()
        self._serializer = serializer or Serializer()
        self._deserializer = deserializer or Deserializer()
        self._running_stage = None

    def initialize(self, data_pipeline_state: Optional[DataPipelineState] = None) -> DataPipelineState:
        """Creates the :class:`.DataPipelineState` and gives the reference to the: :class:`.Preprocess`,
        :class:`.Postprocess`, and :class:`.Serializer`. Once this has been called, any attempt to add new state will
        give a warning."""
        data_pipeline_state = data_pipeline_state or DataPipelineState()
        data_pipeline_state._initialized = False
        if self.data_source is not None:
            self.data_source.attach_data_pipeline_state(data_pipeline_state)
        self._preprocess_pipeline.attach_data_pipeline_state(data_pipeline_state)
        self._postprocess_pipeline.attach_data_pipeline_state(data_pipeline_state)
        self._serializer.attach_data_pipeline_state(data_pipeline_state)
        data_pipeline_state._initialized = True  # TODO: Not sure we need this
        return data_pipeline_state

    @property
    def example_input(self) -> str:
        return self._deserializer.example_input

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

    @staticmethod
    def _identity(samples: Sequence[Any]) -> Sequence[Any]:
        return samples

    def deserialize_processor(self) -> _DeserializeProcessor:
        return self._create_collate_preprocessors(RunningStage.PREDICTING)[0]

    def worker_preprocessor(
        self, running_stage: RunningStage, collate_fn: Optional[Callable] = None, is_serving: bool = False
    ) -> _Preprocessor:
        return self._create_collate_preprocessors(running_stage, collate_fn=collate_fn, is_serving=is_serving)[1]

    def device_preprocessor(self, running_stage: RunningStage) -> _Preprocessor:
        return self._create_collate_preprocessors(running_stage)[2]

    def postprocessor(self, running_stage: RunningStage, is_serving=False) -> _Postprocessor:
        return self._create_uncollate_postprocessors(running_stage, is_serving=is_serving)

    def serialize_processor(self) -> _SerializeProcessor:
        return _SerializeProcessor(self._serializer)

    @classmethod
    def _resolve_function_hierarchy(
        cls, function_name, process_obj, stage: RunningStage, object_type: Optional[Type] = None
    ) -> str:
        if object_type is None:
            object_type = Preprocess

        prefixes = []

        if stage in (RunningStage.TRAINING, RunningStage.TUNING):
            prefixes += ["train", "fit"]
        elif stage == RunningStage.VALIDATING:
            prefixes += ["val", "fit"]
        elif stage == RunningStage.TESTING:
            prefixes += ["test"]
        elif stage == RunningStage.PREDICTING:
            prefixes += ["predict"]

        prefixes += [None]

        for prefix in prefixes:
            if cls._is_overriden(function_name, process_obj, object_type, prefix=prefix):
                return function_name if prefix is None else f"{prefix}_{function_name}"

        return function_name

    def _make_collates(self, on_device: bool, collate: Callable) -> Tuple[Callable, Callable]:
        if on_device:
            return self._identity, collate
        return collate, self._identity

    def _create_collate_preprocessors(
        self,
        stage: RunningStage,
        collate_fn: Optional[Callable] = None,
        is_serving: bool = False,
    ) -> Tuple[_DeserializeProcessor, _Preprocessor, _Preprocessor]:

        original_collate_fn = collate_fn

        preprocess: Preprocess = self._preprocess_pipeline
        prefix: str = _STAGES_PREFIX[stage]

        if collate_fn is not None:
            preprocess._default_collate = collate_fn

        func_names: Dict[str, str] = {
            k: self._resolve_function_hierarchy(k, preprocess, stage, Preprocess) for k in self.PREPROCESS_FUNCS
        }

        collate_fn: Callable = getattr(preprocess, func_names["collate"])

        per_batch_transform_overriden: bool = self._is_overriden_recursive(
            "per_batch_transform", preprocess, Preprocess, prefix=prefix
        )

        per_sample_transform_on_device_overriden: bool = self._is_overriden_recursive(
            "per_sample_transform_on_device", preprocess, Preprocess, prefix=prefix
        )

        collate_in_worker_from_transform: Optional[bool] = getattr(
            preprocess, f"_{prefix}_collate_in_worker_from_transform", None
        )

        is_per_overriden = per_batch_transform_overriden and per_sample_transform_on_device_overriden
        if collate_in_worker_from_transform is None and is_per_overriden:
            raise MisconfigurationException(
                f"{self.__class__.__name__}: `per_batch_transform` and `per_sample_transform_on_device` "
                f"are mutually exclusive for stage {stage}"
            )

        if isinstance(collate_in_worker_from_transform, bool):
            worker_collate_fn, device_collate_fn = self._make_collates(not collate_in_worker_from_transform, collate_fn)
        else:
            worker_collate_fn, device_collate_fn = self._make_collates(
                per_sample_transform_on_device_overriden, collate_fn
            )

        worker_collate_fn = (
            worker_collate_fn.collate_fn if isinstance(worker_collate_fn, _Preprocessor) else worker_collate_fn
        )

        assert_contains_tensor = self._is_overriden_recursive(
            "to_tensor_transform", preprocess, Preprocess, prefix=_STAGES_PREFIX[stage]
        )

        deserialize_processor = _DeserializeProcessor(
            self._deserializer,
            preprocess,
            getattr(preprocess, func_names["pre_tensor_transform"]),
            getattr(preprocess, func_names["to_tensor_transform"]),
        )
        worker_preprocessor = _Preprocessor(
            preprocess,
            worker_collate_fn,
            _Sequential(
                preprocess,
                None if is_serving else getattr(preprocess, func_names["pre_tensor_transform"]),
                None if is_serving else getattr(preprocess, func_names["to_tensor_transform"]),
                getattr(preprocess, func_names["post_tensor_transform"]),
                stage,
                assert_contains_tensor=assert_contains_tensor,
            ),
            getattr(preprocess, func_names["per_batch_transform"]),
            stage,
        )
        worker_preprocessor._original_collate_fn = original_collate_fn
        device_preprocessor = _Preprocessor(
            preprocess,
            device_collate_fn,
            getattr(preprocess, func_names["per_sample_transform_on_device"]),
            getattr(preprocess, func_names["per_batch_transform_on_device"]),
            stage,
            apply_per_sample_transform=device_collate_fn != self._identity,
            on_device=True,
        )
        return deserialize_processor, worker_preprocessor, device_preprocessor

    @staticmethod
    def _model_transfer_to_device_wrapper(
        func: Callable, preprocessor: _Preprocessor, model: "Task", stage: RunningStage
    ) -> Callable:

        if not isinstance(func, _StageOrchestrator):
            func = _StageOrchestrator(func, model)
        func.register_additional_stage(stage, preprocessor)

        return func

    @staticmethod
    def _model_predict_step_wrapper(func: Callable, postprocessor: _Postprocessor, model: "Task") -> Callable:

        if not isinstance(func, _StageOrchestrator):
            _original = func
            func = _StageOrchestrator(func, model)
            func._original = _original
        func.register_additional_stage(RunningStage.PREDICTING, postprocessor)

        return func

    @staticmethod
    def _get_dataloader(model: "Task", loader_name: str) -> Tuple[DataLoader, str]:
        dataloader, attr_name = None, None
        if is_overridden(loader_name, model):
            dataloader = getattr(model, loader_name)
            attr_name = loader_name

        elif model.trainer and hasattr(model.trainer, "datamodule") and model.trainer.datamodule:
            dataloader = getattr(model, f"trainer.datamodule.{loader_name}", None)
            attr_name = f"trainer.datamodule.{loader_name}"

        return dataloader, attr_name

    @staticmethod
    def _patch_dataloader(model: "Task", dataloader: Union[Callable, DataLoader], stage: RunningStage):
        if isinstance(dataloader, DataLoader):
            if _PL_GREATER_EQUAL_1_4_3:
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

    def _attach_preprocess_to_model(
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

            if isinstance(dataloader, (_PatchDataLoader, Callable)):
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

                    _, dl_args["collate_fn"], device_collate_fn = self._create_collate_preprocessors(
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

    def _create_uncollate_postprocessors(
        self,
        stage: RunningStage,
        is_serving: bool = False,
    ) -> _Postprocessor:
        save_per_sample = None
        save_fn = None

        postprocess: Postprocess = self._postprocess_pipeline

        func_names: Dict[str, str] = {
            k: self._resolve_function_hierarchy(k, postprocess, stage, object_type=Postprocess)
            for k in self.POSTPROCESS_FUNCS
        }

        # since postprocessing is exclusive for prediction, we don't have to check the resolution hierarchy here.
        if postprocess._save_path:
            save_per_sample: bool = self._is_overriden_recursive(
                "save_sample", postprocess, Postprocess, prefix=_STAGES_PREFIX[stage]
            )

            if save_per_sample:
                save_per_sample: Callable = getattr(postprocess, func_names["save_sample"])
            else:
                save_fn: Callable = getattr(postprocess, func_names["save_data"])

        return _Postprocessor(
            getattr(postprocess, func_names["uncollate"]),
            getattr(postprocess, func_names["per_batch_transform"]),
            getattr(postprocess, func_names["per_sample_transform"]),
            serializer=None if is_serving else self._serializer,
            save_fn=save_fn,
            save_per_sample=save_per_sample,
            is_serving=is_serving,
        )

    def _attach_postprocess_to_model(
        self,
        model: "Task",
        stage: RunningStage,
        is_serving: bool = False,
    ) -> "Task":
        model.predict_step = self._model_predict_step_wrapper(
            model.predict_step, self._create_uncollate_postprocessors(stage, is_serving=is_serving), model
        )
        return model

    def _attach_to_model(
        self,
        model: "Task",
        stage: RunningStage = None,
        is_serving: bool = False,
    ):
        # not necessary to detach. preprocessing and postprocessing for stage will be overwritten.
        self._attach_preprocess_to_model(model, stage)

        if not stage or stage == RunningStage.PREDICTING:
            self._attach_postprocess_to_model(model, RunningStage.PREDICTING, is_serving=is_serving)

    def _detach_from_model(self, model: "Task", stage: Optional[RunningStage] = None):
        self._detach_preprocessing_from_model(model, stage)

        if not stage or stage == RunningStage.PREDICTING:
            self._detach_postprocess_from_model(model)

    def _detach_preprocessing_from_model(self, model: "Task", stage: Optional[RunningStage] = None):
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

            if isinstance(dataloader, _PatchDataLoader):
                dataloader = dataloader()
            elif isinstance(dataloader, Callable):
                dataloader = dataloader()

            if isinstance(dataloader, Sequence):
                was_seq = True
            else:
                dataloader = [dataloader]
                was_seq = False

            for idx, loader in enumerate(dataloader):
                if isinstance(loader, DataLoader):
                    dl_args = {k: v for k, v in vars(loader).items() if not k.startswith("_")}

                    if isinstance(dl_args["collate_fn"], _Preprocessor):
                        dl_args["collate_fn"] = dl_args["collate_fn"]._original_collate_fn

                        if isinstance(dl_args["dataset"], IterableAutoDataset):
                            del dl_args["sampler"]

                        del dl_args["batch_sampler"]

                        loader = type(loader)(**dl_args)

                dataloader[idx] = loader

            if not was_seq:
                dataloader = dataloader[0]

            dataloader = self._patch_dataloader(model, dataloader, stage)

            self._set_loader(model, whole_attr_name, dataloader)

    @staticmethod
    def _detach_postprocess_from_model(model: "Task"):

        if hasattr(model.predict_step, "_original"):
            # don't delete the predict_step here since we don't know
            # if any other pipeline is attached which may rely on this!
            model.predict_step = model.predict_step._original

    def __str__(self) -> str:
        data_source: DataSource = self.data_source
        preprocess: Preprocess = self._preprocess_pipeline
        postprocess: Postprocess = self._postprocess_pipeline
        serializer: Serializer = self._serializer
        deserializer: Deserializer = self._deserializer
        return (
            f"{self.__class__.__name__}("
            f"data_source={str(data_source)}, "
            f"deserializer={deserializer}, "
            f"preprocess={preprocess}, "
            f"postprocess={postprocess}, "
            f"serializer={serializer})"
        )


class _StageOrchestrator:

    # This is used to map ``SANITY_CHECKING`` to ``VALIDATING``
    internal_mapping = {
        RunningStage.TRAINING: RunningStage.TRAINING,
        RunningStage.SANITY_CHECKING: RunningStage.VALIDATING,
        RunningStage.VALIDATING: RunningStage.VALIDATING,
        RunningStage.TESTING: RunningStage.TESTING,
        RunningStage.PREDICTING: RunningStage.PREDICTING,
        RunningStage.TUNING: RunningStage.TUNING,
    }

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

        internal_running_state = self.internal_mapping[stage]
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
