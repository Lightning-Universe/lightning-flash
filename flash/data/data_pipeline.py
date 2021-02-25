import os
from functools import wraps
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

from pytorch_lightning.trainer.connectors.data_connector import _PatchDataLoader
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data._utils.collate import default_collate, default_convert
from torch.utils.data.dataloader import DataLoader

from flash.data.auto_dataset import AutoDataset
from flash.data.batch import _PostProcessor, _PreProcessor
from flash.data.process import Postprocess, Preprocess


class DataPipeline:

    PREPROCESS_FUNCS = ("load_data", "load_sample", "pre_collate", "post_collate", "device_post_collate")
    POSTPROCESS_FUNCS = ("pre_uncollate", "post_uncollate", "save_data", "save_sample")
    LOADERS_PREFIX = ('train', 'test', 'val', 'predict')

    def __init__(self, preprocess: Preprocess, postprocess: Postprocess):
        self._preprocess_pipeline = preprocess
        self._postprocess_pipeline = postprocess
        self._worker_preprocessor = None
        self._device_preprocessor = None
        self._postprocessor = None

    def load_data(self, data: Any, dataset: AutoDataset = None) -> Any:
        """Loads entire data from Dataset"""
        return self._preprocess_pipeline.load_data(data, dataset=dataset)

    def load_sample(self, sample: Any) -> Any:
        """Loads single sample from dataset"""
        return self._preprocess_pipeline.load_sample(sample)

    def pre_collate(self, sample: Any) -> Any:
        """Transforms to apply to the data before the collation (per-sample basis)"""
        return self._preprocess_pipeline.pre_collate(sample)

    def post_collate(self, batch: Any) -> Any:
        """Transforms to apply to a whole batch (if possible use this for efficiency)

        .. note::
            This option is mutually exclusive with :meth:`device_pre_collate`, since if both are specified, uncollation has to be applied.
        """
        return self._preprocess_pipeline.post_collate(batch)

    def device_pre_collate(self, sample: Any) -> Any:
        """Transforms to apply to the data before the collation (per-sample basis).

        .. note::
            This option is mutually exclusive with :meth:`post_collate`, since if both are specified, uncollation has to be applied.

        .. note::
            This function won't be called within the dataloader workers, since to make that happen each of the workers would have to create it's own CUDA-context which would pollute GPU memory (if on GPU).
        """
        return self._preprocess_pipeline.device_pre_collate(sample)

    def device_post_collate(self, batch: Any) -> Any:
        """
        Transforms to apply to a whole batch (if possible use this for efficiency).

        .. note::
            This function won't be called within the dataloader workers, since to make that happen each of the workers would have to create it's own CUDA-context which would pollute GPU memory (if on GPU).
        """
        return self._preprocess_pipeline.device_pre_collate(batch)

    def pre_uncollate(self, batch: Any) -> Any:
        """Transforms to apply to a whole batch before uncollation to single samples.
        Can involve both CPU and Device transforms as this is not applied in separate workers.
        """
        return self._postprocess_pipeline.pre_uncollate(batch)

    def post_uncollate(self, sample: Any) -> Any:
        """Transforms to apply to a single sample after splitting up the batch.
        Can involve both CPU and Device transforms as this is not applied in separate workers.
        """
        return self._postprocess_pipeline.post_uncollate(sample)

    def uncollate(self, batch: Any) -> Any:
        """Uncollates a batch into single samples.
        Tries to preserve the type whereever possible.
        """
        return self._postprocess_pipeline.uncollate(batch)

    def save_data(self, data: Any, path: str) -> None:
        """Saves all data together to a single path.
        """
        self._postprocess_pipeline.save_data(data, path)

    def save_sample(self, sample: Any, path: str) -> None:
        """Saves each sample individually to a given path.
        """
        self._postprocess_pipeline.save_sample(sample, path)

    def _is_overriden(self, method_name: str, super_obj: Any) -> bool:
        """Cropped Version of https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/utilities/model_helpers.py
        """
        process_obj = self._preprocess_pipeline if isinstance(
            self._preprocess_pipeline, super_obj
        ) else self._postprocess_pipeline

        if not hasattr(process_obj, method_name) or not hasattr(super_obj, method_name):
            return False

        return getattr(process_obj, method_name).__code__ != getattr(super_obj, method_name).__code__

    @staticmethod
    def _do_nothing_collate(samples: Sequence[Any]) -> Sequence[Any]:
        return samples

    @staticmethod
    def _do_nothing_uncollate(batch: Any) -> Any:
        return batch

    @property
    def worker_preprocessor(self) -> _PreProcessor:
        if self._worker_preprocessor is None:
            self._worker_preprocessor = self._create_collate_preprocessors()[0]
        return self._worker_preprocessor

    @worker_preprocessor.setter
    def worker_preprocessor(self, new_processor: _PreProcessor):
        self._worker_preprocessor = new_processor

    @property
    def device_preprocessor(self) -> _PreProcessor:
        if self._device_preprocessor is None:
            self._device_preprocessor = self._create_collate_preprocessors()[1]
        return self._device_preprocessor

    @device_preprocessor.setter
    def device_preprocessor(self, new_processor: _PreProcessor):

        self._device_preprocessor = new_processor

    @property
    def postprocessor(self) -> _PostProcessor:
        if self._postprocessor is None:
            self._postprocessor = self._create_uncollate_postprocessors()

        return self._postprocessor

    @postprocessor.setter
    def postprocessor(self, new_processor: _PostProcessor):
        self._postprocessor = new_processor

    def _create_collate_preprocessors(self,
                                      collate_fn: Optional[Callable] = None) -> Tuple[_PreProcessor, _PreProcessor]:
        if collate_fn is None:
            collate_fn = default_collate

        post_collate_overriden = self._is_overriden('post_collate', Preprocess)

        device_pre_collate_overriden = self._is_overriden('device_pre_collate', Preprocess)

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

        worker_preprocessor = _PreProcessor(worker_collate_fn, self.pre_collate, self.post_collate)
        device_preprocessor = _PreProcessor(device_collate_fn, self.device_pre_collate, self.device_post_collate)
        return worker_preprocessor, device_preprocessor

    @staticmethod
    def _model_transfer_to_device_wrapper(func: Callable, preprocessor: _PreProcessor) -> Callable:

        @wraps(func)
        def new_func(*args, **kwargs):
            moved_to_device = func(*args, **kwargs)
            return preprocessor(moved_to_device)

        return new_func

    @staticmethod
    def _model_predict_step_wrapper(func: Callable, uncollater: _PostProcessor) -> Callable:

        @wraps(func)
        def new_func(*args, **kwargs):
            predicted = func(*args, **kwargs)
            predicted = uncollater(predicted)
            return predicted

        return new_func

    def _get_dataloader(self, model: 'Task', loader_name: str):
        dataloader = None
        if hasattr(model, loader_name):
            dataloader = getattr(model, loader_name)()

        if model.trainer is not None and hasattr(model.trainer, 'datamodule') and model.trainer.datamodule is not None:
            dataloader = getattr(model.trainer.datamodule, loader_name)()

        return dataloader

    def _attach_preprocess_to_model(self, model: 'Task', loader_stage: str = 'all') -> None:
        if loader_stage == 'all':
            loader_stage = self.LOADERS_PREFIX

        elif isinstance(loader_stage, str):
            loader_stage = [loader_stage]

        for stage in loader_stage:
            loader_name = f'{stage}_dataloader'

            dataloader = self._get_dataloader(model, loader_name)

            if dataloader is None:
                continue

            if isinstance(dataloader, _PatchDataLoader):
                dataloader = dataloader()

            if isinstance(dataloader, Sequence):
                was_seq = True
            else:
                dataloader = [dataloader]
                was_seq = False

            for idx, loader in enumerate(dataloader):
                if isinstance(loader, DataLoader):
                    dl_args = {k: v for k, v in vars(loader).items() if not k.startswith("_")}

                    dl_args['collate_fn'], device_collate_fn = self._create_collate_preprocessors(
                        collate_fn=dl_args['collate_fn']
                    )

                    del dl_args["batch_sampler"]

                    loader = type(loader)(**dl_args)

                dataloader[idx] = loader

            if not was_seq:
                dataloader = dataloader[0]

            if isinstance(dataloader, DataLoader):
                dataloader = _PatchDataLoader(dataloader)

            setattr(model, loader_name, dataloader)

        model.transfer_batch_to_device = (
            self._model_transfer_to_device_wrapper(model.transfer_batch_to_device, device_collate_fn)
        )

    def _create_uncollate_postprocessors(self) -> _PostProcessor:
        save_per_sample = None
        save_fn = None

        if self._postprocess_pipeline._save_path is not None:
            save_per_sample = self._is_overriden('save_sample', Postprocess)

            if save_per_sample:
                save_fn = self._postprocess_pipeline._save_sample
            else:
                save_fn = self._postprocess_pipeline._save_data

        return _PostProcessor(
            self.uncollate, self.pre_uncollate, self.post_uncollate, save_fn=save_fn, save_per_sample=save_per_sample
        )

    def _attach_postprocess_to_model(self, model: 'Task') -> 'Task':
        # TODO: move this to on_predict_end?
        if not hasattr(model, "_predict_step"):
            model._predict_step = model.predict_step
        model.predict_step = self._model_predict_step_wrapper(
            model._predict_step, self._create_uncollate_postprocessors()
        )
        return model

    def _attach_to_model(self, model: 'Task', loader_stage: str = 'all'):
        model._preprocess = self._preprocess_pipeline
        self._attach_preprocess_to_model(model, loader_stage)
        if self._postprocess_pipeline is not None:
            model._postprocess = self._postprocess_pipeline
            self._attach_postprocess_to_model(model)

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
        return f"{self.__class__.__name__}(preprocess={self._preprocess_pipeline}, postprocess={self._postprocess_pipeline})"
