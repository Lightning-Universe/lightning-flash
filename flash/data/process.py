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
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Type, TYPE_CHECKING, TypeVar, Union

import torch
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import Tensor
from torch.nn import Module
from torch.utils.data._utils.collate import default_collate

from flash.data.batch import default_uncollate
from flash.data.callback import FlashCallback
from flash.data.utils import _PREPROCESS_FUNCS, _STAGES_PREFIX, convert_to_modules

if TYPE_CHECKING:
    from flash.data.data_pipeline import DataPipelineState


@dataclass(unsafe_hash=True, frozen=True)
class ProcessState:
    """
    Base class for all process states
    """
    pass


STATE_TYPE = TypeVar('STATE_TYPE', bound=ProcessState)


class Properties:

    _running_stage: Optional[RunningStage] = None
    _current_fn: Optional[str] = None
    _data_pipeline_state: Optional['DataPipelineState'] = None
    _state: Dict[Type[ProcessState], ProcessState] = {}

    def get_state(self, state_type: Type[STATE_TYPE]) -> Optional[STATE_TYPE]:
        if self._data_pipeline_state is not None:
            return self._data_pipeline_state.get_state(state_type)
        else:
            return None

    def set_state(self, state: ProcessState):
        self.state[type(state)] = state
        if self._data_pipeline_state is not None:
            self._data_pipeline_state.set_state(state)

    def attach_data_pipeline_state(self, data_pipeline_state: 'DataPipelineState'):
        self._data_pipeline_state = data_pipeline_state
        for state in self.state.values():
            self._data_pipeline_state.set_state(state)

    @property
    def state(self) -> Dict[Type[ProcessState], ProcessState]:
        return self._state

    @state.setter
    def state(self, state: Dict[Type[ProcessState], ProcessState]):
        self._state = state

    @property
    def current_fn(self) -> Optional[str]:
        return self._current_fn

    @current_fn.setter
    def current_fn(self, current_fn: str):
        self._current_fn = current_fn

    @property
    def running_stage(self) -> Optional[RunningStage]:
        return self._running_stage

    @running_stage.setter
    def running_stage(self, running_stage: RunningStage):
        self._running_stage = running_stage

    @property
    def training(self) -> bool:
        return self._running_stage == RunningStage.TRAINING

    @training.setter
    def training(self, val: bool) -> None:
        if val:
            self._running_stage = RunningStage.TRAINING
        elif self.training:
            self._running_stage = None

    @property
    def testing(self) -> bool:
        return self._running_stage == RunningStage.TESTING

    @testing.setter
    def testing(self, val: bool) -> None:
        if val:
            self._running_stage = RunningStage.TESTING
        elif self.testing:
            self._running_stage = None

    @property
    def predicting(self) -> bool:
        return self._running_stage == RunningStage.PREDICTING

    @predicting.setter
    def predicting(self, val: bool) -> None:
        if val:
            self._running_stage = RunningStage.PREDICTING
        elif self.predicting:
            self._running_stage = None

    @property
    def validating(self) -> bool:
        return self._running_stage == RunningStage.VALIDATING

    @validating.setter
    def validating(self, val: bool) -> None:
        if val:
            self._running_stage = RunningStage.VALIDATING
        elif self.validating:
            self._running_stage = None


class Preprocess(Properties, Module):
    """
    The :class:`~flash.data.process.Preprocess` encapsulates
    all the data processing and loading logic that should run before the data is passed to the model.

    It is particularly relevant when you want to provide an end to end implementation which works
    with 4 different stages: ``train``, ``validation``, ``test``,  and inference (``predict``).

    You can override any of the preprocessing hooks to provide custom functionality.
    All hooks default to no-op (except the collate which is PyTorch default
    `collate <https://pytorch.org/docs/stable/data.html#dataloader-collate-fn>`_)

    The :class:`~flash.data.process.Preprocess` supports the following hooks:

        - ``load_data``: Function to receiving some metadata to generate a Mapping from.
            Example::

                * Input: Receive a folder path:

                * Action: Walk the folder path to find image paths and their associated labels.

                * Output: Return a list of image paths and their associated labels.

        - ``load_sample``: Function to load a sample from metadata sample.
            Example::

                * Input: Receive an image path and its label.

                * Action: Load a PIL Image from received image_path.

                * Output: Return the PIL Image and its label.

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
            Example::

                * Input: Receive a list of augmented tensored images and their respective labels.

                * Action: Collate the list of images into batch.

                * Output: Return a batch of images and their labels.

        - ``per_batch_transform_on_device``: Performs transform on a batch already on ``GPU`` or ``TPU``.
            Example::

                * Input: Receive a batch of images and their labels.

                * Action: Apply normalization on the batch by substracting the mean
                    and dividing by the standard deviation from ImageNet.

                * Output: Return a normalized augmented batch of images and their labels.

    .. note::

        By default, each hook will be no-op execpt the collate which is PyTorch default
        `collate <https://pytorch.org/docs/stable/data.html#dataloader-collate-fn>`_.
        To customize them, just override the hooks and ``Flash`` will take care of calling them at the right moment.

    .. note::

        The ``per_sample_transform_on_device`` and ``per_batch_transform`` are mutually exclusive
        as it will impact performances.

    To change the processing behavior only on specific stages,
    you can prefix all the above hooks adding ``train``, ``val``, ``test`` or ``predict``.

    For example, is useful to encapsulate ``predict`` logic as labels aren't availabled at inference time.

    Example::

        class CustomPreprocess(Preprocess):

            def predict_load_data(cls, data: Any, dataset: Optional[Any] = None) -> Mapping:
                # logic for predict data only.

    Each hook is aware of the Trainer ``running stage`` through booleans as follow.

    This is useful to adapt a hook internals for a stage without duplicating code.

    Example::

        class CustomPreprocess(Preprocess):

            def load_data(cls, data: Any, dataset: Optional[Any] = None) -> Mapping:

                if self.training:
                    # logic for train

                elif self.validating:
                    # logic from validation

                elif self.testing:
                    # logic for test

                elif self.predicting:
                    # logic for predict

    .. note::

        It is possible to wrap a ``Dataset`` within a :meth:`~flash.data.process.Preprocess.load_data` function.
        However, we don't recommend to do as such as it is better to rely entirely on the hooks.

    Example::

        from torchvision import datasets

        class CustomPreprocess(Preprocess):

            def load_data(cls, path_to_data: str) -> Iterable:

                return datasets.MNIST(path_to_data, download=True, transform=transforms.ToTensor())

    """

    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
    ):
        super().__init__()

        # used to keep track of provided transforms
        self._train_collate_in_worker_from_transform: Optional[bool] = None
        self._val_collate_in_worker_from_transform: Optional[bool] = None
        self._predict_collate_in_worker_from_transform: Optional[bool] = None
        self._test_collate_in_worker_from_transform: Optional[bool] = None

        self.train_transform = convert_to_modules(self._check_transforms(train_transform, RunningStage.TRAINING))
        self.val_transform = convert_to_modules(self._check_transforms(val_transform, RunningStage.VALIDATING))
        self.test_transform = convert_to_modules(self._check_transforms(test_transform, RunningStage.TESTING))
        self.predict_transform = convert_to_modules(self._check_transforms(predict_transform, RunningStage.PREDICTING))

        if not hasattr(self, "_skip_mutual_check"):
            self._skip_mutual_check = False

        self._callbacks: List[FlashCallback] = []

    # todo (tchaton) Add a warning if a transform is provided, but the hook hasn't been overriden !
    def _check_transforms(self, transform: Optional[Dict[str, Callable]],
                          stage: RunningStage) -> Optional[Dict[str, Callable]]:
        if transform is None:
            return transform

        if not isinstance(transform, Dict):
            raise MisconfigurationException(
                "Transform should be a dict. "
                f"Here are the available keys for your transforms: {_PREPROCESS_FUNCS}."
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
                f'{transform}: `per_batch_transform` and `per_sample_transform_on_device` '
                f'are mutually exclusive.'
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

    # todo (tchaton): Remove when merged. https://github.com/PyTorchLightning/pytorch-lightning/pull/7056
    def tmp_wrap(self, transform) -> Callable:
        if "on_device" in self.current_fn:

            def fn(batch: Any):
                if isinstance(batch, list) and len(batch) == 1 and isinstance(batch[0], dict):
                    return [transform(batch[0])]
                return transform(batch)

            return fn
        return transform

    def _get_transform(self, transform: Dict[str, Callable]) -> Callable:
        if self.current_fn in transform:
            return self.tmp_wrap(transform[self.current_fn])
        return self._identity

    @property
    def current_transform(self) -> Callable:
        if self.training and self.train_transform:
            return self._get_transform(self.train_transform)
        elif self.validating and self.val_transform:
            return self._get_transform(self.val_transform)
        elif self.testing and self.test_transform:
            return self._get_transform(self.test_transform)
        elif self.predicting and self.predict_transform:
            return self._get_transform(self.predict_transform)
        else:
            return self._identity

    @property
    def callbacks(self) -> List['FlashCallback']:
        if not hasattr(self, "_callbacks"):
            self._callbacks: List[FlashCallback] = []
        return self._callbacks

    @callbacks.setter
    def callbacks(self, callbacks: List['FlashCallback']):
        self._callbacks = callbacks

    def add_callbacks(self, callbacks: List['FlashCallback']):
        _callbacks = [c for c in callbacks if c not in self._callbacks]
        self._callbacks.extend(_callbacks)

    @classmethod
    def load_data(cls, data: Any, dataset: Optional[Any] = None) -> Mapping:
        """Loads entire data from Dataset. The input ``data`` can be anything, but you need to return a Mapping.

        Example::

            # data: "."
            # output: [("./cat/1.png", 1), ..., ("./dog/10.png", 0)]

            output: Mapping = load_data(data)

        """
        return data

    @classmethod
    def load_sample(cls, sample: Any, dataset: Optional[Any] = None) -> Any:
        """Loads single sample from dataset"""
        return sample

    def pre_tensor_transform(self, sample: Any) -> Any:
        """Transforms to apply on a single object."""
        return sample

    def to_tensor_transform(self, sample: Any) -> Tensor:
        """Transforms to convert single object to a tensor."""
        return sample

    def post_tensor_transform(self, sample: Tensor) -> Tensor:
        """Transforms to apply on a tensor."""
        return sample

    def per_batch_transform(self, batch: Any) -> Any:
        """Transforms to apply to a whole batch (if possible use this for efficiency).

        .. note::

            This option is mutually exclusive with :meth:`per_sample_transform_on_device`,
            since if both are specified, uncollation has to be applied.
        """
        return batch

    def collate(self, samples: Sequence) -> Any:
        return default_collate(samples)

    def per_sample_transform_on_device(self, sample: Any) -> Any:
        """Transforms to apply to the data before the collation (per-sample basis).

        .. note::

            This option is mutually exclusive with :meth:`per_batch_transform`,
            since if both are specified, uncollation has to be applied.

        .. note::

            This function won't be called within the dataloader workers, since to make that happen
            each of the workers would have to create it's own CUDA-context which would pollute GPU memory (if on GPU).
        """
        return sample

    def per_batch_transform_on_device(self, batch: Any) -> Any:
        """
        Transforms to apply to a whole batch (if possible use this for efficiency).

        .. note::

            This function won't be called within the dataloader workers, since to make that happen
            each of the workers would have to create it's own CUDA-context which would pollute GPU memory (if on GPU).
        """
        return batch


class Postprocess(Properties, Module):

    def __init__(self, save_path: Optional[str] = None):
        super().__init__()
        self._saved_samples = 0
        self._save_path = save_path

    def per_batch_transform(self, batch: Any) -> Any:
        """Transforms to apply on a whole batch before uncollation to individual samples.
        Can involve both CPU and Device transforms as this is not applied in separate workers.
        """
        return batch

    def per_sample_transform(self, sample: Any) -> Any:
        """Transforms to apply to a single sample after splitting up the batch.
        Can involve both CPU and Device transforms as this is not applied in separate workers.
        """
        return sample

    def uncollate(self, batch: Any) -> Any:
        """Uncollates a batch into single samples. Tries to preserve the type whereever possible."""
        return default_uncollate(batch)

    def save_data(self, data: Any, path: str) -> None:
        """Saves all data together to a single path.
        """
        torch.save(data, path)

    def save_sample(self, sample: Any, path: str) -> None:
        """Saves each sample individually to a given path."""
        torch.save(sample, path)

    # TODO: Are those needed ?
    def format_sample_save_path(self, path: str) -> str:
        path = os.path.join(path, f'sample_{self._saved_samples}.ptl')
        self._saved_samples += 1
        return path

    def _save_data(self, data: Any) -> None:
        self.save_data(data, self._save_path)

    def _save_sample(self, sample: Any) -> None:
        self.save_sample(sample, self.format_sample_save_path(self._save_path))


class Serializer(Properties):
    """A :class:`.Serializer` encapsulates a single ``serialize`` method which is used to convert the model ouptut into
    the desired output format when predicting."""

    def __init__(self):
        super().__init__()
        self._is_enabled = True

    def enable(self):
        """Enable serialization."""
        self._is_enabled = True

    def disable(self):
        """Disable serialization."""
        self._is_enabled = False

    def serialize(self, sample: Any) -> Any:
        return sample

    def __call__(self, sample: Any):
        if self._is_enabled:
            return self.serialize(sample)
        else:
            return sample


class SerializerMapping(Serializer):
    """If the model output is a dictionary, then the :class:`.SerializerMapping` enables each entry in the dictionary
    to be passed to it's own :class:`.Serializer`."""

    def __init__(self, serializers: Mapping[str, Serializer]):
        super().__init__()

        self._serializers = serializers

    def serialize(self, sample: Any) -> Any:
        if isinstance(sample, Mapping):
            return {key: serializer.serialize(sample[key]) for key, serializer in self._serializers}
        else:
            # TODO: some error here
            pass

    def attach_data_pipeline_state(self, data_pipeline_state: 'DataPipelineState'):
        for serializer in self._serializers.values():
            serializer.attach_data_pipeline_state(data_pipeline_state)

    @property
    def state(self) -> Dict[str, Dict[Type[ProcessState], ProcessState]]:
        return {key: serializer.state for key, serializer in self._serializers.items()}

    @state.setter
    def state(self, state: Dict[str, Dict[Type[ProcessState], ProcessState]]):
        for key, serializer in self._serializers.items():
            serializer.state = state[key]
