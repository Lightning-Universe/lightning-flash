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
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from pytorch_lightning.utilities.enums import LightningEnum
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data._utils.collate import default_collate

from flash.core.data.data_pipeline import _Preprocessor, DataPipeline
from flash.core.data.data_source import DefaultDataKeys
from flash.core.data.properties import Properties
from flash.core.data.states import CollateFn
from flash.core.data.utils import _PREPROCESS_FUNCS, _STAGES_PREFIX
from flash.core.registry import FlashRegistry
from flash.core.utilities.stages import RunningStage

INPUT_TRANSFORM_TYPE = Optional[
    Union["InputTransform", Callable, Tuple[Union[LightningEnum, str], Dict[str, Any]], Union[LightningEnum, str]]
]


class InputTransformPlacement(LightningEnum):

    PER_SAMPLE_TRANSFORM = "per_sample_transform"
    PER_BATCH_TRANSFORM = "per_batch_transform"
    COLLATE = "collate"
    PER_SAMPLE_TRANSFORM_ON_DEVICE = "per_sample_transform_on_device"
    PER_BATCH_TRANSFORM_ON_DEVICE = "per_batch_transform_on_device"


def transform_context(func: Callable, current_fn: str) -> Callable:
    @wraps(func)
    def wrapper(self, *args, **kwargs) -> Any:
        self.current_fn = current_fn
        result = func(self, *args, **kwargs)
        self.current_fn = None
        return result

    return wrapper


class InputTransform(Properties):
    def configure_transforms(self, *args, **kwargs) -> Dict[InputTransformPlacement, Callable]:
        """The default transforms to use.

        Will be overridden by transforms passed to the ``__init__``.
        """

    def configure_per_sample_transform(self, *args, **kwargs) -> Callable:
        """The default transforms to use.

        Will be overridden by transforms passed to the ``__init__``.
        """
        return self._identity

    def configure_per_batch_transform(self, *args, **kwargs) -> Callable:
        """The default transforms to use.

        Will be overridden by transforms passed to the ``__init__``.
        """
        return self._identity

    def configure_collate(self, *args, **kwargs) -> Callable:
        """The default transforms to use.

        Will be overridden by transforms passed to the ``__init__``.
        """
        return default_collate

    def configure_per_sample_transform_on_device(self, *args, **kwargs) -> Callable:
        """The default transforms to use.

        Will be overridden by transforms passed to the ``__init__``.
        """
        return self._identity

    def configure_per_batch_transform_on_device(self, *args, **kwargs) -> Callable:
        """The default transforms to use.

        Will be overridden by transforms passed to the ``__init__``.
        """
        return self._identity

    def __init__(
        self,
        running_stage: RunningStage,
        transform: Union[Callable, List, Dict[str, Callable]] = None,
        **transform_kwargs,
    ):
        super().__init__()
        # used to keep track of provided transforms
        self._running_stage = running_stage
        self._collate_in_worker_from_transform: Optional[bool] = None

        self._transform_kwargs = transform_kwargs

        transform = transform or self._resolve_transforms(running_stage)
        self.transform = self._check_transforms(transform, running_stage)
        self.callbacks = []

    @property
    def current_transform(self) -> Callable:
        if self.transform:
            return self._get_transform(self.transform)
        return self._identity

    @property
    def transforms(self) -> Dict[str, Optional[Dict[str, Callable]]]:
        """The transforms currently being used by this :class:`~flash.core.data.process.Preprocess`."""
        return {
            "transform": self.transform,
        }

    @partial(transform_context, current_fn="per_sample_transform")
    def per_sample_transform(self, sample: Any) -> Any:
        if isinstance(sample, list):
            return [self.current_transform(s) for s in sample]
        return self.current_transform(sample)

    @partial(transform_context, current_fn="per_batch_transform")
    def per_batch_transform(self, batch: Any) -> Any:
        """Transforms to apply to a whole batch (if possible use this for efficiency).

        .. note::     This option is mutually exclusive with :meth:`per_sample_transform_on_device`,     since if both
        are specified, uncollation has to be applied.
        """
        return self.current_transform(batch)

    @partial(transform_context, current_fn="collate")
    def collate(self, samples: Sequence, metadata=None) -> Any:
        """Transform to convert a sequence of samples to a collated batch."""
        current_transform = self.current_transform
        if current_transform is self._identity:
            current_transform = default_collate

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

    @partial(transform_context, current_fn="per_sample_transform_on_device")
    def per_sample_transform_on_device(self, sample: Any) -> Any:
        """Transforms to apply to the data before the collation (per-sample basis).

        .. note::     This option is mutually exclusive with :meth:`per_batch_transform`,     since if both are
        specified, uncollation has to be applied. .. note::     This function won't be called within the dataloader
        workers, since to make that happen     each of the workers would have to create it's own CUDA-context which
        would pollute GPU memory (if on GPU).
        """
        if isinstance(sample, list):
            return [self.current_transform(s) for s in sample]
        return self.current_transform(sample)

    @partial(transform_context, current_fn="per_batch_transform_on_device")
    def per_batch_transform_on_device(self, batch: Any) -> Any:
        """Transforms to apply to a whole batch (if possible use this for efficiency).

        .. note::     This function won't be called within the dataloader workers, since to make that happen     each of
        the workers would have to create it's own CUDA-context which would pollute GPU memory (if on GPU).
        """
        return self.current_transform(batch)

    @classmethod
    def from_transform(
        cls,
        transform: INPUT_TRANSFORM_TYPE,
        running_stage: RunningStage,
        input_transforms_registry: Optional[FlashRegistry] = None,
    ) -> Optional["InputTransform"]:

        if isinstance(transform, InputTransform):
            transform.running_stage = running_stage
            return transform

        if isinstance(transform, Callable):
            return cls(running_stage, {InputTransformPlacement.PER_SAMPLE_TRANSFORM: transform})

        if isinstance(transform, tuple) or isinstance(transform, (LightningEnum, str)):
            enum, transform_kwargs = cls._sanitize_registry_transform(transform, input_transforms_registry)
            transform_cls = input_transforms_registry.get(enum)
            return transform_cls(running_stage, transform=None, **transform_kwargs)

        if not transform:
            return None

        raise MisconfigurationException(f"The format for the transform isn't correct. Found {transform}")

    @classmethod
    def from_train_transform(
        cls,
        transform: INPUT_TRANSFORM_TYPE,
        input_transforms_registry: Optional[FlashRegistry] = None,
    ) -> Optional["InputTransform"]:
        return cls.from_transform(
            transform=transform,
            running_stage=RunningStage.TRAINING,
            input_transforms_registry=input_transforms_registry,
        )

    @classmethod
    def from_val_transform(
        cls,
        transform: INPUT_TRANSFORM_TYPE,
        input_transforms_registry: Optional[FlashRegistry] = None,
    ) -> Optional["InputTransform"]:
        return cls.from_transform(
            transform=transform,
            running_stage=RunningStage.VALIDATING,
            input_transforms_registry=input_transforms_registry,
        )

    @classmethod
    def from_test_transform(
        cls,
        transform: INPUT_TRANSFORM_TYPE,
        input_transforms_registry: Optional[FlashRegistry] = None,
    ) -> Optional["InputTransform"]:
        return cls.from_transform(
            transform=transform, running_stage=RunningStage.TESTING, input_transforms_registry=input_transforms_registry
        )

    @classmethod
    def from_predict_transform(
        cls,
        transform: INPUT_TRANSFORM_TYPE,
        input_transforms_registry: Optional[FlashRegistry] = None,
    ) -> Optional["InputTransform"]:
        return cls.from_transform(
            transform=transform,
            running_stage=RunningStage.PREDICTING,
            input_transforms_registry=input_transforms_registry,
        )

    def _resolve_transforms(self, running_stage: RunningStage) -> Optional[Dict[str, Callable]]:
        from flash.core.data.data_pipeline import DataPipeline

        resolved_function = getattr(
            self,
            DataPipeline._resolve_function_hierarchy("configure_transforms", self, running_stage, InputTransform),
        )
        params = inspect.signature(resolved_function).parameters
        transforms_out: Optional[Dict[str, Callable]] = resolved_function(
            **{k: v for k, v in self._transform_kwargs.items() if k in params}
        )

        transforms_out = transforms_out or {}
        for placement in InputTransformPlacement:
            transform_name = f"configure_{placement.value}"
            resolved_function = getattr(
                self, DataPipeline._resolve_function_hierarchy(transform_name, self, running_stage, InputTransform)
            )
            params = inspect.signature(resolved_function).parameters
            transforms: Optional[Dict[str, Callable]] = resolved_function(
                **{k: v for k, v in self._transform_kwargs.items() if k in params}
            )
            if transforms != self._identity:
                transforms_out[placement] = transforms
        return transforms_out

    def _check_transforms(
        self, transform: Optional[Dict[str, Callable]], stage: RunningStage
    ) -> Optional[Dict[str, Callable]]:
        if transform is None:
            return transform

        keys_diff = set(transform.keys()).difference([v for v in InputTransformPlacement])

        if len(keys_diff) > 0:
            raise MisconfigurationException(
                f"{stage}_transform contains {keys_diff}. Only {_PREPROCESS_FUNCS} keys are supported."
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

        self._collate_in_worker_from_transform = collate_in_worker
        return transform

    @staticmethod
    def _identity(x: Any) -> Any:
        return x

    def _get_transform(self, transform: Dict[str, Callable]) -> Callable:
        if self.current_fn in transform:
            return transform[self.current_fn]
        return self._identity

    @classmethod
    def _sanitize_registry_transform(
        cls, transform: Tuple[Union[LightningEnum, str], Any], input_transforms_registry: Optional[FlashRegistry]
    ) -> Tuple[Union[LightningEnum, str], Dict]:
        msg = "The transform should be provided as a tuple with the following types (LightningEnum, Dict[str, Any]) "
        msg += "when requesting transform from the registry."
        if not input_transforms_registry:
            raise MisconfigurationException("You requested a transform from the registry, but it is empty.")
        if isinstance(transform, tuple) and len(transform) > 2:
            raise MisconfigurationException(msg)
        if isinstance(transform, (LightningEnum, str)):
            enum = transform
            transform_kwargs = {}
        else:
            enum, transform_kwargs = transform
        if not isinstance(enum, (LightningEnum, str)):
            raise MisconfigurationException(msg)
        if not isinstance(transform_kwargs, Dict):
            raise MisconfigurationException(msg)
        return enum, transform_kwargs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(running_stage={self.running_stage}, transform={self.transform})"

    def __getitem__(self, placement: InputTransformPlacement) -> Callable:
        return self.transform[placement]

    def _make_collates(self, on_device: bool, collate: Callable) -> Tuple[Callable, Callable]:
        if on_device:
            return self._identity, collate
        return collate, self._identity

    @property
    def dataloader_collate_fn(self):
        """Generate the function to be injected within the DataLoader as the collate_fn."""
        return self._create_collate_preprocessors()[0]

    @property
    def on_after_batch_transfer_fn(self):
        """Generate the function to be injected after the on_after_batch_transfer from the LightningModule."""
        return self._create_collate_preprocessors()[1]

    def _create_collate_preprocessors(self) -> Tuple[Any]:
        prefix: str = _STAGES_PREFIX[self.running_stage]

        func_names: Dict[str, str] = {
            k: DataPipeline._resolve_function_hierarchy(k, self, self.running_stage, InputTransform)
            for k in [v.value for v in InputTransformPlacement]
        }

        collate_fn: Callable = getattr(self, func_names["collate"])

        per_batch_transform_overriden: bool = DataPipeline._is_overriden_recursive(
            "per_batch_transform", self, InputTransform, prefix=prefix
        )

        per_sample_transform_on_device_overriden: bool = DataPipeline._is_overriden_recursive(
            "per_sample_transform_on_device", self, InputTransform, prefix=prefix
        )

        is_per_overriden = per_batch_transform_overriden and per_sample_transform_on_device_overriden
        if self._collate_in_worker_from_transform is None and is_per_overriden:
            raise MisconfigurationException(
                f"{self.__class__.__name__}: `per_batch_transform` and `per_sample_transform_on_device` "
                f"are mutually exclusive for stage {self.running_stage}"
            )

        if isinstance(self._collate_in_worker_from_transform, bool):
            worker_collate_fn, device_collate_fn = self._make_collates(
                not self._collate_in_worker_from_transform, collate_fn
            )
        else:
            worker_collate_fn, device_collate_fn = self._make_collates(
                per_sample_transform_on_device_overriden, collate_fn
            )

        worker_collate_fn = (
            worker_collate_fn.collate_fn if isinstance(worker_collate_fn, _Preprocessor) else worker_collate_fn
        )

        worker_preprocessor = _Preprocessor(
            self,
            worker_collate_fn,
            getattr(self, func_names["per_sample_transform"]),
            getattr(self, func_names["per_batch_transform"]),
            self.running_stage,
        )
        device_preprocessor = _Preprocessor(
            self,
            device_collate_fn,
            getattr(self, func_names["per_sample_transform_on_device"]),
            getattr(self, func_names["per_batch_transform_on_device"]),
            self.running_stage,
            apply_per_sample_transform=device_collate_fn != self._identity,
            on_device=True,
        )
        return worker_preprocessor, device_preprocessor
