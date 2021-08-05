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
import os
from abc import ABC, abstractclassmethod, abstractmethod
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import torch
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import Tensor
from torch.utils.data._utils.collate import default_collate

import flash
from flash.core.data.batch import default_uncollate
from flash.core.data.callback import FlashCallback
from flash.core.data.data_source import DatasetDataSource, DataSource, DefaultDataKeys, DefaultDataSources
from flash.core.data.properties import Properties
from flash.core.data.states import CollateFn
from flash.core.data.utils import _PREPROCESS_FUNCS, _STAGES_PREFIX, convert_to_modules, CurrentRunningStageFuncContext


class BasePreprocess(ABC):
    @abstractmethod
    def get_state_dict(self) -> Dict[str, Any]:
        """Override this method to return state_dict."""

    @abstractclassmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool = False):
        """Override this method to load from state_dict."""


class Preprocess(BasePreprocess, Properties):
    """The :class:`~flash.core.data.process.Preprocess` encapsulates all the data processing logic that should run
    before the data is passed to the model. It is particularly useful when you want to provide an end to end
    implementation which works with 4 different stages: ``train``, ``validation``, ``test``,  and inference
    (``predict``).

    The :class:`~flash.core.data.process.Preprocess` supports the following hooks:

        - ``pre_tensor_transform``: Performs transforms on a single data sample.
            Example::

                * Input: Receive a PIL Image and its label.

                * Action: Rotate the PIL Image.

                * Output: Return the rotated PIL image and its label.

        - ``to_tensor_transform``: Converts a single data sample to a tensor / data structure containing tensors.
            Example::

                * Input: Receive the rotated PIL Image and its label.

                * Action: Convert the rotated PIL Image to a tensor.

                * Output: Return the tensored image and its label.

        - ``post_tensor_transform``: Performs transform on a single tensor sample.
            Example::

                * Input: Receive the tensored image and its label.

                * Action: Flip the tensored image randomly.

                * Output: Return the tensored image and its label.

        - ``per_batch_transform``: Performs transforms on a batch.
            In this example, we decided not to override the hook.

        - ``per_sample_transform_on_device``: Performs transform on a sample already on a ``GPU`` or ``TPU``.
            Example::

                * Input: Receive a tensored image on device and its label.

                * Action: Apply random transforms.

                * Output: Return an augmented tensored image on device and its label.

        - ``collate``: Converts a sequence of data samples into a batch.
            Defaults to ``torch.utils.data._utils.collate.default_collate``.
            Example::

                * Input: Receive a list of augmented tensored images and their respective labels.

                * Action: Collate the list of images into batch.

                * Output: Return a batch of images and their labels.

        - ``per_batch_transform_on_device``: Performs transform on a batch already on ``GPU`` or ``TPU``.
            Example::

                * Input: Receive a batch of images and their labels.

                * Action: Apply normalization on the batch by subtracting the mean
                    and dividing by the standard deviation from ImageNet.

                * Output: Return a normalized augmented batch of images and their labels.

    .. note::

        The ``per_sample_transform_on_device`` and ``per_batch_transform`` are mutually exclusive
        as it will impact performances.

    Data processing can be configured by overriding hooks or through transforms. The preprocess transforms are given as
    a mapping from hook names to callables. Default transforms can be configured by overriding the
    ``default_transforms`` or ``{train,val,test,predict}_default_transforms`` methods. These can then be overridden by
    the user with the ``{train,val,test,predict}_transform`` arguments to the ``Preprocess``. All of the hooks can be
    used in the transform mappings.

    Example::

        class CustomPreprocess(Preprocess):

            def default_transforms() -> Mapping[str, Callable]:
                return {
                    "to_tensor_transform": transforms.ToTensor(),
                    "collate": torch.utils.data._utils.collate.default_collate,
                }

            def train_default_transforms() -> Mapping[str, Callable]:
                return {
                    "pre_tensor_transform": transforms.RandomHorizontalFlip(),
                    "to_tensor_transform": transforms.ToTensor(),
                    "collate": torch.utils.data._utils.collate.default_collate,
                }

    When overriding hooks for particular stages, you can prefix with ``train``, ``val``, ``test`` or ``predict``. For
    example, you can achieve the same as the above example by implementing ``train_pre_tensor_transform`` and
    ``train_to_tensor_transform``.

    Example::

        class CustomPreprocess(Preprocess):

            def train_pre_tensor_transform(self, sample: PIL.Image) -> PIL.Image:
                return transforms.RandomHorizontalFlip()(sample)

            def to_tensor_transform(self, sample: PIL.Image) -> torch.Tensor:
                return transforms.ToTensor()(sample)

            def collate(self, samples: List[torch.Tensor]) -> torch.Tensor:
                return torch.utils.data._utils.collate.default_collate(samples)

    Each hook is aware of the Trainer running stage through booleans. These are useful for adapting functionality for a
    stage without duplicating code.

    Example::

        class CustomPreprocess(Preprocess):

            def pre_tensor_transform(self, sample: PIL.Image) -> PIL.Image:

                if self.training:
                    # logic for training

                elif self.validating:
                    # logic for validation

                elif self.testing:
                    # logic for testing

                elif self.predicting:
                    # logic for predicting
    """

    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        data_sources: Optional[Dict[str, "DataSource"]] = None,
        deserializer: Optional["Deserializer"] = None,
        default_data_source: Optional[str] = None,
    ):
        super().__init__()

        # resolve the default transforms
        train_transform = train_transform or self._resolve_transforms(RunningStage.TRAINING)
        val_transform = val_transform or self._resolve_transforms(RunningStage.VALIDATING)
        test_transform = test_transform or self._resolve_transforms(RunningStage.TESTING)
        predict_transform = predict_transform or self._resolve_transforms(RunningStage.PREDICTING)

        # used to keep track of provided transforms
        self._train_collate_in_worker_from_transform: Optional[bool] = None
        self._val_collate_in_worker_from_transform: Optional[bool] = None
        self._predict_collate_in_worker_from_transform: Optional[bool] = None
        self._test_collate_in_worker_from_transform: Optional[bool] = None

        # store the transform before conversion to modules.
        self.train_transform = self._check_transforms(train_transform, RunningStage.TRAINING)
        self.val_transform = self._check_transforms(val_transform, RunningStage.VALIDATING)
        self.test_transform = self._check_transforms(test_transform, RunningStage.TESTING)
        self.predict_transform = self._check_transforms(predict_transform, RunningStage.PREDICTING)

        self._train_transform = convert_to_modules(self.train_transform)
        self._val_transform = convert_to_modules(self.val_transform)
        self._test_transform = convert_to_modules(self.test_transform)
        self._predict_transform = convert_to_modules(self.predict_transform)

        if DefaultDataSources.DATASETS not in data_sources:
            data_sources[DefaultDataSources.DATASETS] = DatasetDataSource()

        self._data_sources = data_sources
        self._deserializer = deserializer
        self._default_data_source = default_data_source
        self._callbacks: List[FlashCallback] = []
        self._default_collate: Callable = default_collate

    @property
    def deserializer(self) -> Optional["Deserializer"]:
        return self._deserializer

    def _resolve_transforms(self, running_stage: RunningStage) -> Optional[Dict[str, Callable]]:
        from flash.core.data.data_pipeline import DataPipeline

        resolved_function = getattr(
            self, DataPipeline._resolve_function_hierarchy("default_transforms", self, running_stage, Preprocess)
        )

        with CurrentRunningStageFuncContext(running_stage, "default_transforms", self):
            transforms: Optional[Dict[str, Callable]] = resolved_function()
        return transforms

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        preprocess_state_dict = self.get_state_dict()
        if not isinstance(preprocess_state_dict, Dict):
            raise MisconfigurationException("get_state_dict should return a dictionary")
        preprocess_state_dict["_meta"] = {}
        preprocess_state_dict["_meta"]["module"] = self.__module__
        preprocess_state_dict["_meta"]["class_name"] = self.__class__.__name__
        preprocess_state_dict["_meta"]["_state"] = self._state
        destination["preprocess.state_dict"] = preprocess_state_dict
        self._ddp_params_and_buffers_to_ignore = ["preprocess.state_dict"]
        return super()._save_to_state_dict(destination, prefix, keep_vars)

    def _check_transforms(
        self, transform: Optional[Dict[str, Callable]], stage: RunningStage
    ) -> Optional[Dict[str, Callable]]:
        if transform is None:
            return transform

        if not isinstance(transform, Dict):
            raise MisconfigurationException(
                "Transform should be a dict. " f"Here are the available keys for your transforms: {_PREPROCESS_FUNCS}."
            )

        keys_diff = set(transform.keys()).difference(_PREPROCESS_FUNCS)

        if len(keys_diff) > 0:
            raise MisconfigurationException(
                f"{stage}_transform contains {keys_diff}. Only {_PREPROCESS_FUNCS} keys are supported."
            )

        is_per_batch_transform_in = "per_batch_transform" in transform
        is_per_sample_transform_on_device_in = "per_sample_transform_on_device" in transform

        if is_per_batch_transform_in and is_per_sample_transform_on_device_in:
            raise MisconfigurationException(
                f"{transform}: `per_batch_transform` and `per_sample_transform_on_device` " f"are mutually exclusive."
            )

        collate_in_worker: Optional[bool] = None

        if is_per_batch_transform_in or (not is_per_batch_transform_in and not is_per_sample_transform_on_device_in):
            collate_in_worker = True

        elif is_per_sample_transform_on_device_in:
            collate_in_worker = False

        setattr(self, f"_{_STAGES_PREFIX[stage]}_collate_in_worker_from_transform", collate_in_worker)
        return transform

    @staticmethod
    def _identity(x: Any) -> Any:
        return x

    def _get_transform(self, transform: Dict[str, Callable]) -> Callable:
        if self.current_fn in transform:
            return transform[self.current_fn]
        return self._identity

    @property
    def current_transform(self) -> Callable:
        if self.training and self._train_transform:
            return self._get_transform(self._train_transform)
        if self.validating and self._val_transform:
            return self._get_transform(self._val_transform)
        if self.testing and self._test_transform:
            return self._get_transform(self._test_transform)
        if self.predicting and self._predict_transform:
            return self._get_transform(self._predict_transform)
        return self._identity

    @property
    def transforms(self) -> Dict[str, Optional[Dict[str, Callable]]]:
        """The transforms currently being used by this :class:`~flash.core.data.process.Preprocess`."""
        return {
            "train_transform": self.train_transform,
            "val_transform": self.val_transform,
            "test_transform": self.test_transform,
            "predict_transform": self.predict_transform,
        }

    @property
    def callbacks(self) -> List["FlashCallback"]:
        if not hasattr(self, "_callbacks"):
            self._callbacks: List[FlashCallback] = []
        return self._callbacks

    @callbacks.setter
    def callbacks(self, callbacks: List["FlashCallback"]):
        self._callbacks = callbacks

    def add_callbacks(self, callbacks: List["FlashCallback"]):
        _callbacks = [c for c in callbacks if c not in self._callbacks]
        self._callbacks.extend(_callbacks)

    @staticmethod
    def default_transforms() -> Optional[Dict[str, Callable]]:
        """The default transforms to use.

        Will be overridden by transforms passed to the ``__init__``.
        """
        return None

    def pre_tensor_transform(self, sample: Any) -> Any:
        """Transforms to apply on a single object."""
        return self.current_transform(sample)

    def to_tensor_transform(self, sample: Any) -> Tensor:
        """Transforms to convert single object to a tensor."""
        return self.current_transform(sample)

    def post_tensor_transform(self, sample: Tensor) -> Tensor:
        """Transforms to apply on a tensor."""
        return self.current_transform(sample)

    def per_batch_transform(self, batch: Any) -> Any:
        """Transforms to apply to a whole batch (if possible use this for efficiency).

        .. note::

            This option is mutually exclusive with :meth:`per_sample_transform_on_device`,
            since if both are specified, uncollation has to be applied.
        """
        return self.current_transform(batch)

    def collate(self, samples: Sequence, metadata=None) -> Any:
        """Transform to convert a sequence of samples to a collated batch."""
        current_transform = self.current_transform
        if current_transform is self._identity:
            current_transform = self._default_collate

        # the model can provide a custom ``collate_fn``.
        collate_fn = self.get_state(CollateFn)
        if collate_fn is not None:
            collate_fn = collate_fn.collate_fn
        else:
            collate_fn = current_transform
            # return collate_fn.collate_fn(samples)

        parameters = inspect.signature(collate_fn).parameters
        if len(parameters) > 1 and DefaultDataKeys.METADATA in parameters:
            return collate_fn(samples, metadata)
        return collate_fn(samples)

    def per_sample_transform_on_device(self, sample: Any) -> Any:
        """Transforms to apply to the data before the collation (per-sample basis).

        .. note::

            This option is mutually exclusive with :meth:`per_batch_transform`,
            since if both are specified, uncollation has to be applied.

        .. note::

            This function won't be called within the dataloader workers, since to make that happen
            each of the workers would have to create it's own CUDA-context which would pollute GPU memory (if on GPU).
        """
        return self.current_transform(sample)

    def per_batch_transform_on_device(self, batch: Any) -> Any:
        """Transforms to apply to a whole batch (if possible use this for efficiency).

        .. note::

            This function won't be called within the dataloader workers, since to make that happen
            each of the workers would have to create it's own CUDA-context which would pollute GPU memory (if on GPU).
        """
        return self.current_transform(batch)

    def available_data_sources(self) -> Sequence[str]:
        """Get the list of available data source names for use with this
        :class:`~flash.core.data.process.Preprocess`.

        Returns:
            The list of data source names.
        """
        return list(self._data_sources.keys())

    def data_source_of_name(self, data_source_name: str) -> DataSource:
        """Get the :class:`~flash.core.data.data_source.DataSource` of the given name from the
        :class:`~flash.core.data.process.Preprocess`.

        Args:
            data_source_name: The name of the data source to look up.

        Returns:
            The :class:`~flash.core.data.data_source.DataSource` of the given name.

        Raises:
            MisconfigurationException: If the requested data source is not configured by this
                :class:`~flash.core.data.process.Preprocess`.
        """
        if data_source_name == "default":
            data_source_name = self._default_data_source
        data_sources = self._data_sources
        if data_source_name in data_sources:
            return data_sources[data_source_name]
        raise MisconfigurationException(
            f"No '{data_source_name}' data source is available for use with the {type(self)}. The available data "
            f"sources are: {', '.join(self.available_data_sources())}."
        )


class DefaultPreprocess(Preprocess):
    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        data_sources: Optional[Dict[str, "DataSource"]] = None,
        default_data_source: Optional[str] = None,
    ):
        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_sources=data_sources or {"default": DataSource()},
            default_data_source=default_data_source or "default",
        )

    def get_state_dict(self) -> Dict[str, Any]:
        return {**self.transforms}

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool):
        return cls(**state_dict)


class Postprocess(Properties):
    """The :class:`~flash.core.data.process.Postprocess` encapsulates all the data processing logic that should run
    after the model."""

    def __init__(self, save_path: Optional[str] = None):
        super().__init__()
        self._saved_samples = 0
        self._save_path = save_path

    @staticmethod
    def per_batch_transform(batch: Any) -> Any:
        """Transforms to apply on a whole batch before uncollation to individual samples.

        Can involve both CPU and Device transforms as this is not applied in separate workers.
        """
        return batch

    @staticmethod
    def per_sample_transform(sample: Any) -> Any:
        """Transforms to apply to a single sample after splitting up the batch.

        Can involve both CPU and Device transforms as this is not applied in separate workers.
        """
        return sample

    @staticmethod
    def uncollate(batch: Any) -> Any:
        """Uncollates a batch into single samples.

        Tries to preserve the type whereever possible.
        """
        return default_uncollate(batch)

    @staticmethod
    def save_data(data: Any, path: str) -> None:
        """Saves all data together to a single path."""
        torch.save(data, path)

    @staticmethod
    def save_sample(sample: Any, path: str) -> None:
        """Saves each sample individually to a given path."""
        torch.save(sample, path)

    # TODO: Are those needed ?
    def format_sample_save_path(self, path: str) -> str:
        path = os.path.join(path, f"sample_{self._saved_samples}.ptl")
        self._saved_samples += 1
        return path

    def _save_data(self, data: Any) -> None:
        self.save_data(data, self._save_path)

    def _save_sample(self, sample: Any) -> None:
        self.save_sample(sample, self.format_sample_save_path(self._save_path))


class Serializer(Properties):
    """A :class:`.Serializer` encapsulates a single ``serialize`` method which is used to convert the model output
    into the desired output format when predicting."""

    def __init__(self):
        super().__init__()
        self._is_enabled = True

    def enable(self):
        """Enable serialization."""
        self._is_enabled = True

    def disable(self):
        """Disable serialization."""
        self._is_enabled = False

    @staticmethod
    def serialize(sample: Any) -> Any:
        """Serialize the given sample into the desired output format.

        Args:
            sample: The output from the :class:`.Postprocess`.

        Returns:
            The serialized output.
        """
        return sample

    def __call__(self, sample: Any) -> Any:
        if self._is_enabled:
            return self.serialize(sample)
        return sample


class SerializerMapping(Serializer):
    """If the model output is a dictionary, then the :class:`.SerializerMapping` enables each entry in the
    dictionary to be passed to it's own :class:`.Serializer`."""

    def __init__(self, serializers: Mapping[str, Serializer]):
        super().__init__()

        self._serializers = serializers

    def serialize(self, sample: Any) -> Any:
        if isinstance(sample, Mapping):
            return {key: serializer.serialize(sample[key]) for key, serializer in self._serializers.items()}
        raise ValueError("The model output must be a mapping when using a SerializerMapping.")

    def attach_data_pipeline_state(self, data_pipeline_state: "flash.core.data.data_pipeline.DataPipelineState"):
        for serializer in self._serializers.values():
            serializer.attach_data_pipeline_state(data_pipeline_state)


class Deserializer(Properties):
    """Deserializer."""

    def deserialize(self, sample: Any) -> Any:  # TODO: Output must be a tensor???
        raise NotImplementedError

    @property
    @abstractmethod
    def example_input(self) -> str:
        raise NotImplementedError

    def __call__(self, sample: Any) -> Any:
        return self.deserialize(sample)


class DeserializerMapping(Deserializer):
    # TODO: This is essentially a duplicate of SerializerMapping, should be abstracted away somewhere
    """Deserializer Mapping."""

    def __init__(self, deserializers: Mapping[str, Deserializer]):
        super().__init__()

        self._deserializers = deserializers

    def deserialize(self, sample: Any) -> Any:
        if isinstance(sample, Mapping):
            return {key: deserializer.deserialize(sample[key]) for key, deserializer in self._deserializers.items()}
        raise ValueError("The model output must be a mapping when using a DeserializerMapping.")

    def attach_data_pipeline_state(self, data_pipeline_state: "flash.core.data.data_pipeline.DataPipelineState"):
        for deserializer in self._deserializers.values():
            deserializer.attach_data_pipeline_state(data_pipeline_state)
