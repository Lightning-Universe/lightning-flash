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
from abc import ABC, abstractclassmethod, abstractmethod
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import Tensor
from torch.utils.data._utils.collate import default_collate

from flash.core.data.callback import ControlFlow, FlashCallback
from flash.core.data.io.input import Input, InputFormat, DatasetInput
from flash.core.data.process import Deserializer
from flash.core.data.properties import ProcessState, Properties
from flash.core.data.states import (
    CollateFn,
    PerBatchTransform,
    PerBatchTransformOnDevice,
    PerSampleTransformOnDevice,
    PostTensorTransform,
    PreTensorTransform,
    ToTensorTransform,
)
from flash.core.data.transforms import ApplyToKeys
from flash.core.data.utils import (
    _contains_any_tensor,
    _INPUT_TRANSFORM_FUNCS,
    _STAGES_PREFIX,
    convert_to_modules,
    CurrentFuncContext,
    CurrentRunningStageContext,
    CurrentRunningStageFuncContext,
)
from flash.core.utilities.stages import RunningStage


class BaseInputTransform(ABC):
    @abstractmethod
    def get_state_dict(self) -> Dict[str, Any]:
        """Override this method to return state_dict."""

    @abstractclassmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool = False):
        """Override this method to load from state_dict."""


class InputTransform(BaseInputTransform, Properties):
    """The :class:`~flash.core.data.io.input_transform.InputTransform` encapsulates all the data processing logic
    that should run before the data is passed to the model. It is particularly useful when you want to provide an
    end to end implementation which works with 4 different stages: ``train``, ``validation``, ``test``,  and
    inference (``predict``).

    The :class:`~flash.core.data.io.input_transform.InputTransform` supports the following hooks:

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

    Data processing can be configured by overriding hooks or through transforms. The input transforms are given as
    a mapping from hook names to callables. Default transforms can be configured by overriding the
    ``default_transforms`` or ``{train,val,test,predict}_default_transforms`` methods. These can then be overridden by
    the user with the ``{train,val,test,predict}_transform`` arguments to the ``InputTransform``.
    All of the hooks can be used in the transform mappings.

    Example::

        class CustomInputTransform(InputTransform):

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

        class CustomInputTransform(InputTransform):

            def train_pre_tensor_transform(self, sample: PIL.Image) -> PIL.Image:
                return transforms.RandomHorizontalFlip()(sample)

            def to_tensor_transform(self, sample: PIL.Image) -> torch.Tensor:
                return transforms.ToTensor()(sample)

            def collate(self, samples: List[torch.Tensor]) -> torch.Tensor:
                return torch.utils.data._utils.collate.default_collate(samples)

    Each hook is aware of the Trainer running stage through booleans. These are useful for adapting functionality for a
    stage without duplicating code.

    Example::

        class CustomInputTransform(InputTransform):

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
        train_transform: Optional[Union[Callable, List, Dict[str, Callable]]] = None,
        val_transform: Optional[Union[Callable, List, Dict[str, Callable]]] = None,
        test_transform: Optional[Union[Callable, List, Dict[str, Callable]]] = None,
        predict_transform: Optional[Union[Callable, List, Dict[str, Callable]]] = None,
        inputs: Optional[Dict[str, "Input"]] = None,
        deserializer: Optional["Deserializer"] = None,
        default_input: Optional[str] = None,
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

        if InputFormat.DATASETS not in inputs:
            inputs[InputFormat.DATASETS] = DatasetInput()

        self._inputs = inputs
        self._deserializer = deserializer
        self._default_input = default_input
        self._callbacks: List[FlashCallback] = []
        self._default_collate: Callable = default_collate

    @property
    def deserializer(self) -> Optional["Deserializer"]:
        return self._deserializer

    def _resolve_transforms(self, running_stage: RunningStage) -> Optional[Dict[str, Callable]]:
        from flash.core.data.data_pipeline import DataPipeline

        resolved_function = getattr(
            self, DataPipeline._resolve_function_hierarchy("default_transforms", self, running_stage, InputTransform)
        )

        with CurrentRunningStageFuncContext(running_stage, "default_transforms", self):
            transforms: Optional[Dict[str, Callable]] = resolved_function()
        return transforms

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        input_transform_state_dict = self.get_state_dict()
        if not isinstance(input_transform_state_dict, Dict):
            raise MisconfigurationException("get_state_dict should return a dictionary")
        input_transform_state_dict["_meta"] = {}
        input_transform_state_dict["_meta"]["module"] = self.__module__
        input_transform_state_dict["_meta"]["class_name"] = self.__class__.__name__
        input_transform_state_dict["_meta"]["_state"] = self._state
        destination["input_transform.state_dict"] = input_transform_state_dict
        self._ddp_params_and_buffers_to_ignore = ["input_transform.state_dict"]
        return super()._save_to_state_dict(destination, prefix, keep_vars)

    def _check_transforms(
        self, transform: Optional[Dict[str, Callable]], stage: RunningStage
    ) -> Optional[Dict[str, Callable]]:
        if transform is None:
            return transform

        if isinstance(transform, list):
            transform = {"pre_tensor_transform": ApplyToKeys(DefaultDataKeys.INPUT, torch.nn.Sequential(*transform))}
        elif callable(transform):
            transform = {"pre_tensor_transform": ApplyToKeys(DefaultDataKeys.INPUT, transform)}

        if not isinstance(transform, Dict):
            raise MisconfigurationException(
                "Transform should be a dict. "
                f"Here are the available keys for your transforms: {_INPUT_TRANSFORM_FUNCS}."
            )

        keys_diff = set(transform.keys()).difference(_INPUT_TRANSFORM_FUNCS)

        if len(keys_diff) > 0:
            raise MisconfigurationException(
                f"{stage}_transform contains {keys_diff}. Only {_INPUT_TRANSFORM_FUNCS} keys are supported."
            )

        is_per_batch_transform_in = "per_batch_transform" in transform
        is_per_sample_transform_on_device_in = "per_sample_transform_on_device" in transform

        if is_per_batch_transform_in and is_per_sample_transform_on_device_in:
            raise MisconfigurationException(
                f"{transform}: `per_batch_transform` and `per_sample_transform_on_device` are mutually exclusive."
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
        """The transforms currently being used by this
        :class:`~flash.core.data.io.input_transform.InputTransform`."""
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

    def _apply_sample_transform(self, sample: Any) -> Any:
        if isinstance(sample, list):
            return [self.current_transform(s) for s in sample]
        return self.current_transform(sample)

    def _apply_batch_transform(self, batch: Any):
        return self.current_transform(batch)

    def _apply_transform_on_sample(self, sample: Any, transform: Callable):
        if isinstance(sample, list):
            return [transform(s) for s in sample]

        return transform(sample)

    def _apply_transform_on_batch(self, batch: Any, transform: Callable):
        return transform(batch)

    def _apply_process_state_transform(
        self,
        process_state: ProcessState,
        sample: Optional[Any] = None,
        batch: Optional[Any] = None,
    ):
        # assert both sample and batch are not None
        if sample is None:
            assert batch is not None, "sample not provided, batch should not be None"
            mode = "batch"
        else:
            assert batch is None, "sample provided, batch should be None"
            mode = "sample"

        process_state_transform = self.get_state(process_state)

        if process_state_transform is not None:
            if process_state_transform.transform is not None:
                if mode == "sample":
                    return self._apply_transform_on_sample(sample, process_state_transform.transform)
                else:
                    return self._apply_transform_on_batch(batch, process_state_transform.transform)
            else:
                if mode == "sample":
                    return sample
                else:
                    return batch
        else:
            if mode == "sample":
                return self._apply_sample_transform(sample)
            else:
                return self._apply_batch_transform(batch)

    def pre_tensor_transform(self, sample: Any) -> Any:
        """Transforms to apply on a single object."""
        return self._apply_process_state_transform(PreTensorTransform, sample=sample)

    def to_tensor_transform(self, sample: Any) -> Tensor:
        """Transforms to convert single object to a tensor."""
        return self._apply_process_state_transform(ToTensorTransform, sample=sample)

    def post_tensor_transform(self, sample: Tensor) -> Tensor:
        """Transforms to apply on a tensor."""
        return self._apply_process_state_transform(PostTensorTransform, sample=sample)

    def per_batch_transform(self, batch: Any) -> Any:
        """Transforms to apply to a whole batch (if possible use this for efficiency).

        .. note::

            This option is mutually exclusive with :meth:`per_sample_transform_on_device`,
            since if both are specified, uncollation has to be applied.
        """
        return self._apply_process_state_transform(PerBatchTransform, batch=batch)

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
        return self._apply_process_state_transform(PerSampleTransformOnDevice, sample=sample)

    def per_batch_transform_on_device(self, batch: Any) -> Any:
        """Transforms to apply to a whole batch (if possible use this for efficiency).

        .. note::

            This function won't be called within the dataloader workers, since to make that happen
            each of the workers would have to create it's own CUDA-context which would pollute GPU memory (if on GPU).
        """
        return self._apply_process_state_transform(PerBatchTransformOnDevice, batch=batch)

    def available_inputs(self) -> Sequence[str]:
        """Get the list of available data source names for use with this
        :class:`~flash.core.data.io.input_transform.InputTransform`.

        Returns:
            The list of data source names.
        """
        return list(self._inputs.keys())

    def input_of_name(self, input_name: str) -> Input:
        """Get the :class:`~flash.core.data.io.input.Input` of the given name from the
        :class:`~flash.core.data.io.input_transform.InputTransform`.

        Args:
            input_name: The name of the data source to look up.

        Returns:
            The :class:`~flash.core.data.io.input.Input` of the given name.

        Raises:
            MisconfigurationException: If the requested data source is not configured by this
                :class:`~flash.core.data.io.input_transform.InputTransform`.
        """
        if input_name == "default":
            input_name = self._default_input
        inputs = self._inputs
        if input_name in inputs:
            return inputs[input_name]
        raise MisconfigurationException(
            f"No '{input_name}' data source is available for use with the {type(self)}. The available data "
            f"sources are: {', '.join(self.available_inputs())}."
        )


class DefaultInputTransform(InputTransform):
    def __init__(
        self,
        train_transform: Optional[Union[Callable, List, Dict[str, Callable]]] = None,
        val_transform: Optional[Union[Callable, List, Dict[str, Callable]]] = None,
        test_transform: Optional[Union[Callable, List, Dict[str, Callable]]] = None,
        predict_transform: Optional[Union[Callable, List, Dict[str, Callable]]] = None,
        inputs: Optional[Dict[str, "Input"]] = None,
        default_input: Optional[str] = None,
    ):
        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            inputs=inputs or {"default": Input()},
            default_input=default_input or "default",
        )

    def get_state_dict(self) -> Dict[str, Any]:
        return {**self.transforms}

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool):
        return cls(**state_dict)


class _InputTransformSequential(torch.nn.Module):
    """This class is used to chain 3 functions together for the _InputTransformProcessor ``per_sample_transform``
    function.

    1. ``pre_tensor_transform``
    2. ``to_tensor_transform``
    3. ``post_tensor_transform``
    """

    def __init__(
        self,
        input_transform: InputTransform,
        pre_tensor_transform: Optional[Callable],
        to_tensor_transform: Optional[Callable],
        post_tensor_transform: Callable,
        stage: RunningStage,
        assert_contains_tensor: bool = False,
    ):
        super().__init__()
        self.input_transform = input_transform
        self.callback = ControlFlow(self.input_transform.callbacks)
        self.pre_tensor_transform = convert_to_modules(pre_tensor_transform)
        self.to_tensor_transform = convert_to_modules(to_tensor_transform)
        self.post_tensor_transform = convert_to_modules(post_tensor_transform)
        self.stage = stage
        self.assert_contains_tensor = assert_contains_tensor

        self._current_stage_context = CurrentRunningStageContext(stage, input_transform, reset=False)
        self._pre_tensor_transform_context = CurrentFuncContext("pre_tensor_transform", input_transform)
        self._to_tensor_transform_context = CurrentFuncContext("to_tensor_transform", input_transform)
        self._post_tensor_transform_context = CurrentFuncContext("post_tensor_transform", input_transform)

    def forward(self, sample: Any) -> Any:
        self.callback.on_load_sample(sample, self.stage)

        with self._current_stage_context:
            if self.pre_tensor_transform is not None:
                with self._pre_tensor_transform_context:
                    sample = self.pre_tensor_transform(sample)
                    self.callback.on_pre_tensor_transform(sample, self.stage)

            if self.to_tensor_transform is not None:
                with self._to_tensor_transform_context:
                    sample = self.to_tensor_transform(sample)
                    self.callback.on_to_tensor_transform(sample, self.stage)

                if self.assert_contains_tensor:
                    if not _contains_any_tensor(sample):
                        raise MisconfigurationException(
                            "When ``to_tensor_transform`` is overriden, "
                            "``DataPipeline`` expects the outputs to be ``tensors``"
                        )

            with self._post_tensor_transform_context:
                sample = self.post_tensor_transform(sample)
                self.callback.on_post_tensor_transform(sample, self.stage)

            return sample

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"\t(pre_tensor_transform): {str(self.pre_tensor_transform)}\n"
            f"\t(to_tensor_transform): {str(self.to_tensor_transform)}\n"
            f"\t(post_tensor_transform): {str(self.post_tensor_transform)}\n"
            f"\t(assert_contains_tensor): {str(self.assert_contains_tensor)}\n"
            f"\t(stage): {str(self.stage)}"
        )


class _InputTransformProcessor(torch.nn.Module):
    """
    This class is used to encapsultate the following functions of a InputTransformInputTransform Object:
    Inside a worker:
        per_sample_transform: Function to transform an individual sample
            Inside a worker, it is actually make of 3 functions:
                * pre_tensor_transform
                * to_tensor_transform
                * post_tensor_transform
        collate: Function to merge sample into a batch
        per_batch_transform: Function to transform an individual batch
            * per_batch_transform

    Inside main process:
        per_sample_transform: Function to transform an individual sample
            * per_sample_transform_on_device
        collate: Function to merge sample into a batch
        per_batch_transform: Function to transform an individual batch
            * per_batch_transform_on_device
    """

    def __init__(
        self,
        input_transform: InputTransform,
        collate_fn: Callable,
        per_sample_transform: Union[Callable, _InputTransformSequential],
        per_batch_transform: Callable,
        stage: RunningStage,
        apply_per_sample_transform: bool = True,
        on_device: bool = False,
    ):
        super().__init__()
        self.input_transform = input_transform
        self.callback = ControlFlow(self.input_transform.callbacks)
        self.collate_fn = convert_to_modules(collate_fn)
        self.per_sample_transform = convert_to_modules(per_sample_transform)
        self.per_batch_transform = convert_to_modules(per_batch_transform)
        self.apply_per_sample_transform = apply_per_sample_transform
        self.stage = stage
        self.on_device = on_device

        extension = f"{'_on_device' if self.on_device else ''}"
        self._current_stage_context = CurrentRunningStageContext(stage, input_transform)
        self._per_sample_transform_context = CurrentFuncContext(f"per_sample_transform{extension}", input_transform)
        self._collate_context = CurrentFuncContext("collate", input_transform)
        self._per_batch_transform_context = CurrentFuncContext(f"per_batch_transform{extension}", input_transform)

    @staticmethod
    def _extract_metadata(
        samples: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
        metadata = [s.pop(DefaultDataKeys.METADATA, None) if isinstance(s, Mapping) else None for s in samples]
        return samples, metadata if any(m is not None for m in metadata) else None

    def forward(self, samples: Sequence[Any]) -> Any:
        # we create a new dict to prevent from potential memory leaks
        # assuming that the dictionary samples are stored in between and
        # potentially modified before the transforms are applied.
        if isinstance(samples, dict):
            samples = dict(samples.items())

        with self._current_stage_context:

            if self.apply_per_sample_transform:
                with self._per_sample_transform_context:
                    _samples = []

                    if isinstance(samples, Mapping):
                        samples = [samples]

                    for sample in samples:
                        sample = self.per_sample_transform(sample)
                        if self.on_device:
                            self.callback.on_per_sample_transform_on_device(sample, self.stage)
                        _samples.append(sample)

                samples = type(_samples)(_samples)

                with self._collate_context:
                    samples, metadata = self._extract_metadata(samples)
                    try:
                        samples = self.collate_fn(samples, metadata)
                    except TypeError:
                        samples = self.collate_fn(samples)
                    if metadata and isinstance(samples, dict):
                        samples[DefaultDataKeys.METADATA] = metadata
                    self.callback.on_collate(samples, self.stage)

            with self._per_batch_transform_context:
                samples = self.per_batch_transform(samples)
                if self.on_device:
                    self.callback.on_per_batch_transform_on_device(samples, self.stage)
                else:
                    self.callback.on_per_batch_transform(samples, self.stage)
            return samples

    def __str__(self) -> str:
        # todo: define repr function which would take object and string attributes to be shown
        return (
            "_InputTransformProcessor:\n"
            f"\t(per_sample_transform): {str(self.per_sample_transform)}\n"
            f"\t(collate_fn): {str(self.collate_fn)}\n"
            f"\t(per_batch_transform): {str(self.per_batch_transform)}\n"
            f"\t(apply_per_sample_transform): {str(self.apply_per_sample_transform)}\n"
            f"\t(on_device): {str(self.on_device)}\n"
            f"\t(stage): {str(self.stage)}"
        )
