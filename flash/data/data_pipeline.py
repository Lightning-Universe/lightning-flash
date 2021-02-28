import os
from functools import partial, wraps
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Type, TYPE_CHECKING, Union

from pytorch_lightning.trainer.connectors.data_connector import _PatchDataLoader
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data._utils.collate import default_collate, default_convert
from torch.utils.data.dataloader import DataLoader

from flash.data.auto_dataset import AutoDataset
from flash.data.batch import _PostProcessor, _PreProcessor
from flash.data.process import Postprocess, Preprocess

if TYPE_CHECKING:
    from flash.core.model import Task


class DataPipeline:

    PREPROCESS_FUNCS = (
        "load_data", "load_sample", "pre_collate", "post_collate", "device_pre_collate", "device_post_collate"
    )
    POSTPROCESS_FUNCS = ("pre_uncollate", "post_uncollate", "save_data", "save_sample")
    LOADERS_PREFIX = {
        RunningStage.TRAINING: 'train',
        RunningStage.TESTING: 'test',
        RunningStage.EVALUATING: 'val',
        RunningStage.PREDICTING: 'predict'
    }

    def __init__(self, preprocess: Optional[Preprocess] = None, postprocess: Optional[Postprocess] = None):
        if preprocess is None:
            preprocess = Preprocess()

        if postprocess is None:
            postprocess = Postprocess()

        self._preprocess_pipeline = preprocess
        self._postprocess_pipeline = postprocess
        self._postprocessor = None
        self._running_stage = None

    def _is_overriden(self, method_name: str, super_obj: Any, prefix: Optional[str] = None) -> bool:
        """
        Cropped Version of
        https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/utilities/model_helpers.py
        """
        process_obj = self._preprocess_pipeline if isinstance(
            self._preprocess_pipeline, super_obj
        ) else self._postprocess_pipeline

        current_method_name = method_name if prefix is None else f'{prefix}_{method_name}'

        if not hasattr(process_obj, current_method_name):
            return False

        return getattr(process_obj, current_method_name).__code__ != getattr(super_obj, method_name).__code__

    @staticmethod
    def _do_nothing_collate(samples: Sequence[Any]) -> Sequence[Any]:
        return samples

    @staticmethod
    def _do_nothing_uncollate(batch: Any) -> Any:
        return batch

    def worker_preprocessor(self, running_stage: RunningStage) -> _PreProcessor:
        return self._create_collate_preprocessors(running_stage)[0]

    def device_preprocessor(self, running_stage: RunningStage) -> _PreProcessor:
        return self._create_collate_preprocessors(running_stage)[1]

    @property
    def postprocessor(self) -> _PostProcessor:
        if self._postprocessor is None:
            self._postprocessor = self._create_uncollate_postprocessors()
        return self._postprocessor

    @postprocessor.setter
    def postprocessor(self, new_processor: _PostProcessor):
        self._postprocessor = new_processor

    def _resolve_function_hierarchy(self, function_name, stage: RunningStage, object_type: Optional[Type] = None):
        if object_type is None:
            object_type = Preprocess

        prefixes = ['']

        # TODO: Check if tuning uses training or validation data
        if stage in (RunningStage.TRAINING, RunningStage.TUNING):
            prefixes = ['train', 'fit'] + prefixes
        elif stage == RunningStage.EVALUATING:
            prefixes = ['validation', 'fit'] + prefixes
        elif stage == RunningStage.TESTING:
            prefixes = ['test'] + prefixes
        elif stage == RunningStage.PREDICTING:
            prefixes = ['predict'] + prefixes

        for prefix in prefixes:
            if self._is_overriden(function_name, object_type, prefix=prefix):
                return f'{prefix}_{function_name}'

        return function_name

    def _create_collate_preprocessors(self,
                                      stage: RunningStage,
                                      collate_fn: Optional[Callable] = None) -> Tuple[_PreProcessor, _PreProcessor]:
        if collate_fn is None:
            collate_fn = default_collate

        func_names = {k: self._resolve_function_hierarchy(k, stage, Preprocess) for k in self.PREPROCESS_FUNCS}

        post_collate_overriden = self._is_overriden(func_names['post_collate'], Preprocess)

        device_pre_collate_overriden = self._is_overriden(func_names['device_pre_collate'], Preprocess)

        if post_collate_overriden and device_pre_collate_overriden:
            raise MisconfigurationException(
                f'{self.__class__.__name__}: post_collate and gpu_pre_collate are mutual exclusive.'
            )

        elif post_collate_overriden:
            worker_collate_fn = collate_fn
            device_collate_fn = self._do_nothing_collate

        elif device_pre_collate_overriden:
            worker_collate_fn = self._do_nothing_collate
            device_collate_fn = collate_fn

        else:
            worker_collate_fn = collate_fn
            device_collate_fn = self._do_nothing_collate

        worker_collate_fn = worker_collate_fn.collate_fn if isinstance(
            worker_collate_fn, _PreProcessor
        ) else worker_collate_fn

        worker_preprocessor = _PreProcessor(
            worker_collate_fn, getattr(self._preprocess_pipeline, func_names['pre_collate']),
            getattr(self._preprocess_pipeline, func_names['post_collate'])
        )
        device_preprocessor = _PreProcessor(
            device_collate_fn, getattr(self._preprocess_pipeline, func_names['device_pre_collate']),
            getattr(self._preprocess_pipeline, func_names['device_post_collate'])
        )
        return worker_preprocessor, device_preprocessor

    @staticmethod
    def _model_transfer_to_device_wrapper(
        func: Callable, preprocessor: _PreProcessor, model: 'Task', stage: RunningStage
    ) -> Callable:

        @wraps(func)
        def new_func(*args, **kwargs):
            moved_to_device = func(*args, **kwargs)
            # TODO: This may not be the best solution since it's abusing python scopes.
            # Search for a better working solution
            if model.running_stage == stage:
                moved_to_device = preprocessor(moved_to_device)
            return moved_to_device

        # Necessary to detach
        new_func._original = func
        new_func._processor = preprocessor
        new_func._stage = stage

        return new_func

    @staticmethod
    def _model_predict_step_wrapper(func: Callable, postprocessor: _PostProcessor) -> Callable:

        @wraps(func)
        def new_func(*args, **kwargs):
            predicted = func(*args, **kwargs)
            predicted = postprocessor(predicted)
            return predicted

        # necessary to detach
        new_func._original = func
        new_func._processor = postprocessor

        return new_func

    @staticmethod
    def _get_dataloader(model: 'Task', loader_name: str) -> Tuple[DataLoader, str]:
        dataloader, attr_name = None, None
        if hasattr(model, loader_name):
            dataloader = getattr(model, loader_name)
            attr_name = loader_name

        elif model.trainer is not None and hasattr(
            model.trainer, 'datamodule'
        ) and model.trainer.datamodule is not None:
            dataloader = getattr(model.trainer.datamodule, loader_name)
            attr_name = f'trainer.datamodule.{loader_name}'

        return dataloader, attr_name

    @staticmethod
    def _set_loader(model: 'Task', loader_name: str, new_loader: DataLoader):
        *intermediates, final_name = loader_name.split('.')
        curr_attr = model

        # This relies on python calling all non-integral types by reference.
        # It may fail for integral types since those will be called by value.
        for intermediate in intermediates:
            curr_attr = getattr(curr_attr, intermediate)

        setattr(curr_attr, final_name, new_loader)
        setattr(model, final_name, new_loader)

    def _attach_preprocess_to_model(
        self, model: 'Task', stages: Optional[RunningStage] = None, device_transform_only: bool = False
    ) -> None:
        if stages is None:
            stages = [RunningStage.TRAINING, RunningStage.EVALUATING, RunningStage.TESTING, RunningStage.PREDICTING]

        elif isinstance(stages, RunningStage):
            stages = [stages]

        for stage in stages:

            if stage == RunningStage.PREDICTING:
                print("here")
                pass

            loader_name = f'{self.LOADERS_PREFIX[stage]}_dataloader'

            dataloader, whole_attr_name = self._get_dataloader(model, loader_name)

            if dataloader is None:
                continue

            if isinstance(dataloader, _PatchDataLoader):
                dataloader = dataloader()
            elif isinstance(dataloader, Callable):
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
        if self._postprocess_pipeline._save_path is not None:
            save_per_sample = self._is_overriden('save_sample', Postprocess)

            if save_per_sample:
                save_per_sample = self._postprocess_pipeline._save_sample
            else:
                save_fn = self._postprocess_pipeline._save_data

        return _PostProcessor(
            self._postprocess_pipeline.uncollate,
            self._postprocess_pipeline.pre_uncollate,
            self._postprocess_pipeline.post_uncollate,
            save_fn=save_fn,
            save_per_sample=save_per_sample
        )

    def _attach_postprocess_to_model(self, model: 'Task') -> 'Task':
        model.predict_step = self._model_predict_step_wrapper(
            model.predict_step, self._create_uncollate_postprocessors()
        )
        return model

    def _attach_to_model(self, model: 'Task', stage: RunningStage = None):
        self._detach_from_model(model)
        model._preprocess = self._preprocess_pipeline
        self._attach_preprocess_to_model(model, stage)
        model._postprocess = self._postprocess_pipeline
        self._attach_postprocess_to_model(model)

    def _detach_from_model(self, model: 'Task', stages: Optional[RunningStage] = None):
        self._detach_preprocessing_from_model(model, stages)

        if stages is None or stages == RunningStage.PREDICTING:
            self._detach_postprocess_from_model(model)

    @staticmethod
    def _composed_collates(samples: Any, worker_collate: Callable, device_collate: Callable) -> Any:
        return device_collate(worker_collate(samples))

    def _detach_preprocessing_from_model(self, model: 'Task', stages: Optional[RunningStage] = None):
        if stages is None:
            stages = [RunningStage.TRAINING, RunningStage.EVALUATING, RunningStage.TESTING, RunningStage.PREDICTING]

        elif isinstance(stages, RunningStage):
            stages = [stages]

        for stage in stages:

            current_func = model.transfer_batch_to_device

            stages_to_rewrap = []

            # Traverse the decorators (multiple are possible) until decorator for specific stage was found.
            # Rewrap all previously traversed stages afterwards
            was_attached = False
            while True:
                # indicates that it was wrapped
                if hasattr(current_func, '_stage') and hasattr(current_func, '_original'):
                    if current_func._stage == stage:
                        was_attached = True
                        model.transfer_batch_to_device = current_func._original
                        break
                    else:
                        stages_to_rewrap.append(current_func._stage)
                        current_func = current_func._original

                else:
                    break

            if not was_attached:
                return

            for _stage in stages_to_rewrap:
                self._attach_preprocess_to_model(model, _stage, device_transform_only=True)

            device_collate = current_func._processor.collate_fn

            loader_name = f'{self.LOADERS_PREFIX[stage]}_dataloader'

            dataloader, whole_attr_name = self._get_dataloader(model, loader_name)

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
                    # TODO: See lightning for proper reinstantiation of loader
                    worker_collate = dataloader.collate_fn
                    dl_args = {k: v for k, v in vars(loader).items() if not k.startswith("_")}

                    dl_args['collate_fn'] = partial(
                        self._composed_collates, worker_collate=worker_collate, device_collate=device_collate
                    )
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
        else:
            pass

    def _generate_callable_auto_dataset(self, data: Union[Iterable, Any]) -> Callable:

        def fn():
            return self._generate_auto_dataset(data)

        return fn

    def _generate_auto_dataset(self, data: Union[Iterable, Any], running_stage: RunningStage = None) -> AutoDataset:
        return AutoDataset(data=data, data_pipeline=self, running_stage=running_stage)

    def to_dataloader(
        self, data: Union[Iterable, Any], auto_collate: Optional[bool] = None, **loader_kwargs
    ) -> DataLoader:
        if 'collate_fn' in loader_kwargs:
            if auto_collate is not None:
                raise MisconfigurationException('auto_collate and collate_fn are mutually exclusive')

        else:
            if auto_collate is None:
                auto_collate = True

            collate_fn = self.worker_collate_fn

            if collate_fn is not None:
                loader_kwargs['collate_fn'] = collate_fn

            else:
                if auto_collate:
                    loader_kwargs['collate_fn'] = default_collate
                else:
                    loader_kwargs['collate_fn'] = default_convert

        return DataLoader(self._generate_auto_dataset(data), **loader_kwargs)

    def __repr__(self) -> str:
        preprocess = self._preprocess_pipeline
        postprocess = self._postprocess_pipeline
        return f"{self.__class__.__name__}(preprocess={preprocess}, postprocess={postprocess})"
