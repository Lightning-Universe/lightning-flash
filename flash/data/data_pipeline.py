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
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Type, TYPE_CHECKING, Union

from pytorch_lightning.trainer.connectors.data_connector import _PatchDataLoader
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities import imports
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data._utils.collate import default_collate, default_convert
from torch.utils.data.dataloader import DataLoader

from flash.data.auto_dataset import AutoDataset
from flash.data.batch import _PostProcessor, _PreProcessor, _Sequential
from flash.data.process import Postprocess, Preprocess
from flash.data.utils import _PREPROCESS_FUNCS, _STAGES_PREFIX

if TYPE_CHECKING:
    from flash.core.model import Task


class DataPipeline:
    """
    DataPipeline hold the engineering logic to connect ``Preprocess`` and/or ``PostProcess`` objects to
    the ``DataModule``, Flash ``Task`` and ``Trainer``.

    Example::

        class CustomPreprocess(Preprocess):
            pass

        class CustomPostprocess(Postprocess):
            pass

        custom_data_pipeline = DataPipeline(CustomPreprocess(), CustomPostprocess())

        # And it can set to either the datamodule or model.

        datamodule.data_pipeline = custom_data_pipeline

        model.data_pipeline = custom_data_pipeline
    """

    PREPROCESS_FUNCS = _PREPROCESS_FUNCS

    def __init__(self, preprocess: Optional[Preprocess] = None, postprocess: Optional[Postprocess] = None) -> None:
        self._preprocess_pipeline = preprocess or Preprocess()
        self._postprocess_pipeline = postprocess or Postprocess()
        self._postprocessor = None
        self._running_stage = None

    @staticmethod
    def _is_overriden(method_name: str, process_obj, super_obj: Any, prefix: Optional[str] = None) -> bool:
        """
        Cropped Version of
        https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/utilities/model_helpers.py
        """

        current_method_name = method_name if prefix is None else f'{prefix}_{method_name}'

        if not hasattr(process_obj, current_method_name):
            return False

        return getattr(process_obj, current_method_name).__code__ != getattr(super_obj, method_name).__code__

    @property
    def preprocess_state(self):
        if self._preprocess_pipeline:
            return self._preprocess_pipeline.state

    @classmethod
    def _is_overriden_recursive(
        cls, method_name: str, process_obj, super_obj: Any, prefix: Optional[str] = None
    ) -> bool:
        """
        Cropped Version of
        https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/utilities/model_helpers.py
        """
        assert isinstance(process_obj, super_obj)
        if prefix is None and not hasattr(super_obj, method_name):
            raise MisconfigurationException(f"This function doesn't belong to the parent class {super_obj}")

        current_method_name = method_name if prefix is None else f'{prefix}_{method_name}'

        if not hasattr(process_obj, current_method_name):
            return DataPipeline._is_overriden_recursive(method_name, process_obj, super_obj)

        current_code = inspect.unwrap(getattr(process_obj, current_method_name)).__code__
        has_different_code = current_code != getattr(super_obj, method_name).__code__

        if not prefix:
            return has_different_code
        else:
            return has_different_code or cls._is_overriden_recursive(method_name, process_obj, super_obj)

    @staticmethod
    def _identity(samples: Sequence[Any]) -> Sequence[Any]:
        return samples

    def worker_preprocessor(self, running_stage: RunningStage) -> _PreProcessor:
        return self._create_collate_preprocessors(running_stage)[0]

    def device_preprocessor(self, running_stage: RunningStage) -> _PreProcessor:
        return self._create_collate_preprocessors(running_stage)[1]

    @property
    def postprocessor(self) -> _PostProcessor:
        return self._postprocessor or self._create_uncollate_postprocessors()

    @postprocessor.setter
    def postprocessor(self, new_processor: _PostProcessor):
        self._postprocessor = new_processor

    @classmethod
    def _resolve_function_hierarchy(
        cls, function_name, process_obj, stage: RunningStage, object_type: Optional[Type] = None
    ) -> str:
        if object_type is None:
            object_type = Preprocess

        prefixes = ['']
        if stage in (RunningStage.TRAINING, RunningStage.TUNING):
            prefixes += ['train', 'fit']
        elif stage == RunningStage.VALIDATING:
            prefixes += ['val', 'fit']
        elif stage == RunningStage.TESTING:
            prefixes += ['test']
        elif stage == RunningStage.PREDICTING:
            prefixes += ['predict']

        for prefix in prefixes:
            if cls._is_overriden(function_name, process_obj, object_type, prefix=prefix):
                return f'{prefix}_{function_name}'

        return function_name

    def _create_collate_preprocessors(
        self,
        stage: RunningStage,
        collate_fn: Optional[Callable] = None,
    ) -> Tuple[_PreProcessor, _PreProcessor]:
        original_collate_fn = collate_fn
        if collate_fn is None:
            collate_fn = default_collate

        preprocess: Preprocess = self._preprocess_pipeline

        func_names: Dict[str, str] = {
            k: self._resolve_function_hierarchy(k, preprocess, stage, Preprocess)
            for k in self.PREPROCESS_FUNCS
        }

        if self._is_overriden_recursive("collate", preprocess, Preprocess, prefix=_STAGES_PREFIX[stage]):
            collate_fn: Callable = getattr(preprocess, func_names["collate"])

        per_batch_transform_overriden: bool = self._is_overriden_recursive(
            "per_batch_transform", preprocess, Preprocess, prefix=_STAGES_PREFIX[stage]
        )

        per_sample_transform_on_device_overriden: bool = self._is_overriden_recursive(
            "per_sample_transform_on_device", preprocess, Preprocess, prefix=_STAGES_PREFIX[stage]
        )

        skip_mutual_check: bool = getattr(preprocess, "skip_mutual_check", False)

        if (not skip_mutual_check and per_batch_transform_overriden and per_sample_transform_on_device_overriden):
            raise MisconfigurationException(
                f'{self.__class__.__name__}: `per_batch_transform` and `per_sample_transform_on_device` '
                f'are mutual exclusive for stage {stage}'
            )

        elif per_batch_transform_overriden:
            worker_collate_fn = collate_fn
            device_collate_fn = self._identity

        elif per_sample_transform_on_device_overriden:
            worker_collate_fn = self._identity
            device_collate_fn = collate_fn

        else:
            worker_collate_fn = collate_fn
            device_collate_fn = self._identity

        worker_collate_fn = worker_collate_fn.collate_fn if isinstance(
            worker_collate_fn, _PreProcessor
        ) else worker_collate_fn

        assert_contains_tensor = self._is_overriden_recursive(
            "to_tensor_transform", preprocess, Preprocess, prefix=_STAGES_PREFIX[stage]
        )

        worker_preprocessor = _PreProcessor(
            preprocess, worker_collate_fn,
            _Sequential(
                preprocess,
                getattr(preprocess, func_names['pre_tensor_transform']),
                getattr(preprocess, func_names['to_tensor_transform']),
                getattr(preprocess, func_names['post_tensor_transform']),
                stage,
                assert_contains_tensor=assert_contains_tensor,
            ), getattr(preprocess, func_names['per_batch_transform']), stage
        )
        worker_preprocessor._original_collate_fn = original_collate_fn
        device_preprocessor = _PreProcessor(
            preprocess,
            device_collate_fn,
            getattr(preprocess, func_names['per_sample_transform_on_device']),
            getattr(preprocess, func_names['per_batch_transform_on_device']),
            stage,
            apply_per_sample_transform=device_collate_fn != self._identity,
            on_device=True,
        )
        return worker_preprocessor, device_preprocessor

    @staticmethod
    def _model_transfer_to_device_wrapper(
        func: Callable, preprocessor: _PreProcessor, model: 'Task', stage: RunningStage
    ) -> Callable:

        if not isinstance(func, _StageOrchestrator):
            func = _StageOrchestrator(func, model)
        func.register_additional_stage(stage, preprocessor)

        return func

    @staticmethod
    def _model_predict_step_wrapper(func: Callable, postprocessor: _PostProcessor, model: 'Task') -> Callable:

        if not isinstance(func, _StageOrchestrator):
            _original = func
            func = _StageOrchestrator(func, model)
            func._original = _original
        func.register_additional_stage(RunningStage.PREDICTING, postprocessor)

        return func

    @staticmethod
    def _get_dataloader(model: 'Task', loader_name: str) -> Tuple[DataLoader, str]:
        dataloader, attr_name = None, None
        if hasattr(model, loader_name):
            dataloader = getattr(model, loader_name)
            attr_name = loader_name

        elif model.trainer and hasattr(model.trainer, 'datamodule') and model.trainer.datamodule:
            dataloader = getattr(model, f'trainer.datamodule.{loader_name}', None)
            attr_name = f'trainer.datamodule.{loader_name}'

        return dataloader, attr_name

    @staticmethod
    def _set_loader(model: 'Task', loader_name: str, new_loader: DataLoader) -> None:
        """
        This function is used to set the loader to model and/or datamodule
        """
        *intermediates, final_name = loader_name.split('.')
        curr_attr = model

        # This relies on python calling all non-integral types by reference.
        # It may fail for integral types since those will be called by value.
        for intermediate in intermediates:
            curr_attr = getattr(curr_attr, intermediate)

        setattr(curr_attr, final_name, new_loader)
        setattr(model, final_name, new_loader)

    def _attach_preprocess_to_model(
        self, model: 'Task', stage: Optional[RunningStage] = None, device_transform_only: bool = False
    ) -> None:
        if not stage:
            stages = [RunningStage.TRAINING, RunningStage.VALIDATING, RunningStage.TESTING, RunningStage.PREDICTING]

        elif isinstance(stage, RunningStage):
            stages = [stage]

        for stage in stages:

            if stage == RunningStage.PREDICTING:
                pass

            loader_name = f'{_STAGES_PREFIX[stage]}_dataloader'

            dataloader, whole_attr_name = self._get_dataloader(model, loader_name)

            if not dataloader:
                continue

            if isinstance(dataloader, (_PatchDataLoader, Callable)):
                dataloader = dataloader()

            if not dataloader:
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

                    dl_args['collate_fn'], device_collate_fn = self._create_collate_preprocessors(
                        stage=stage, collate_fn=dl_args['collate_fn']
                    )

                    # don't have to reinstantiate loader if just rewrapping devices (happens during detach)
                    if not device_transform_only:
                        del dl_args["batch_sampler"]
                        loader = type(loader)(**dl_args)

                dataloader[idx] = loader

            # don't have to set attribute if rewrapping device part (happens during detach)
            if not device_transform_only:
                if not was_seq:
                    dataloader = dataloader[0]

                if isinstance(dataloader, DataLoader):
                    dataloader = _PatchDataLoader(dataloader)

                self._set_loader(model, whole_attr_name, dataloader)

            model.transfer_batch_to_device = (
                self._model_transfer_to_device_wrapper(model.transfer_batch_to_device, device_collate_fn, model, stage)
            )

    def _create_uncollate_postprocessors(self) -> _PostProcessor:
        save_per_sample = None
        save_fn = None

        # since postprocessing is exclusive for prediction, we don't have to check the resolution hierarchy here.
        if self._postprocess_pipeline._save_path:
            save_per_sample = self._is_overriden('save_sample', self._postprocess_pipeline, Postprocess)

            if save_per_sample:
                save_per_sample = self._postprocess_pipeline._save_sample
            else:
                save_fn = self._postprocess_pipeline._save_data

        return _PostProcessor(
            self._postprocess_pipeline.uncollate,
            self._postprocess_pipeline.per_batch_transform,
            self._postprocess_pipeline.per_sample_transform,
            save_fn=save_fn,
            save_per_sample=save_per_sample
        )

    def _attach_postprocess_to_model(self, model: 'Task') -> 'Task':
        model.predict_step = self._model_predict_step_wrapper(
            model.predict_step, self._create_uncollate_postprocessors(), model
        )
        return model

    def _attach_to_model(self, model: 'Task', stage: RunningStage = None):
        # not necessary to detach. preprocessing and postprocessing for stage will be overwritten.
        self._attach_preprocess_to_model(model, stage)

        if not stage or stage == RunningStage.PREDICTING:
            self._attach_postprocess_to_model(model)

    def _detach_from_model(self, model: 'Task', stage: Optional[RunningStage] = None):
        self._detach_preprocessing_from_model(model, stage)

        if not stage or stage == RunningStage.PREDICTING:
            self._detach_postprocess_from_model(model)

    def _detach_preprocessing_from_model(self, model: 'Task', stage: Optional[RunningStage] = None):
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

            loader_name = f'{_STAGES_PREFIX[stage]}_dataloader'

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

                    if isinstance(dl_args['collate_fn'], _PreProcessor):
                        dl_args['collate_fn'] = dl_args['collate_fn']._original_collate_fn
                        del dl_args["batch_sampler"]
                        loader = type(loader)(**dl_args)

                dataloader[idx] = loader

            if not was_seq:
                dataloader = dataloader[0]

            if isinstance(dataloader, DataLoader):
                dataloader = _PatchDataLoader(dataloader)

            self._set_loader(model, whole_attr_name, dataloader)

    @staticmethod
    def _detach_postprocess_from_model(model: 'Task'):

        if hasattr(model.predict_step, '_original'):
            # don't delete the predict_step here since we don't know
            # if any other pipeline is attached which may rely on this!
            model.predict_step = model.predict_step._original

    def _generate_callable_auto_dataset(
        self, data: Union[Iterable, Any], running_stage: RunningStage = None
    ) -> Callable:

        def fn():
            return self._generate_auto_dataset(data, running_stage=running_stage)

        return fn

    def _generate_auto_dataset(self, data: Union[Iterable, Any], running_stage: RunningStage = None) -> AutoDataset:
        return AutoDataset(data=data, data_pipeline=self, running_stage=running_stage)

    def to_dataloader(
        self, data: Union[Iterable, Any], auto_collate: Optional[bool] = None, **loader_kwargs
    ) -> DataLoader:
        if 'collate_fn' in loader_kwargs:
            if auto_collate:
                raise MisconfigurationException('auto_collate and collate_fn are mutually exclusive')

        else:
            if auto_collate is None:
                auto_collate = True

            collate_fn = self.worker_collate_fn

            if collate_fn:
                loader_kwargs['collate_fn'] = collate_fn

            else:
                loader_kwargs['collate_fn'] = default_collate if auto_collate else default_convert

        return DataLoader(self._generate_auto_dataset(data), **loader_kwargs)

    def __str__(self) -> str:
        preprocess: Preprocess = self._preprocess_pipeline
        postprocess = self._postprocess_pipeline
        return f"{self.__class__.__name__}(preprocess={preprocess}, postprocess={postprocess})"


class _StageOrchestrator:

    # This is used to map ``SANITY_CHECKING`` to ``VALIDATING``
    internal_mapping = {
        RunningStage.TRAINING: RunningStage.TRAINING,
        RunningStage.SANITY_CHECKING: RunningStage.VALIDATING,
        RunningStage.VALIDATING: RunningStage.VALIDATING,
        RunningStage.TESTING: RunningStage.TESTING,
        RunningStage.PREDICTING: RunningStage.PREDICTING,
        RunningStage.TUNING: RunningStage.TUNING
    }

    def __init__(self, func_to_wrap: Callable, model: 'Task') -> None:
        self.func = func_to_wrap

        self._stage_mapping = {k: None for k in RunningStage}
        self.model = weakref.proxy(model)

        functools.update_wrapper(self, self.func)

    def __call__(self, *args, **kwargs):
        outputs = self.func(*args, **kwargs)

        internal_running_state = self.internal_mapping[self.model.trainer._running_stage]
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
        return all([v is None for v in self._stage_mapping.values()]) or not self._stage_mapping
