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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.enums import LightningEnum
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data._utils.collate import default_collate

from flash.core.data.data_source import DefaultDataKeys
from flash.core.data.properties import Properties
from flash.core.data.states import CollateFn
from flash.core.data.transforms import ApplyToKeys
from flash.core.data.utils import _PREPROCESS_FUNCS, _STAGES_PREFIX
from flash.core.registry import FlashRegistry

TRANSFORM_TYPE = Optional[Union["PreTransform", Callable, Tuple[LightningEnum, Dict[str, Any]], LightningEnum]]


class PreTransformPlacement(LightningEnum):

    PER_SAMPLE_TRANSFORM = "per_sample_transform"
    PER_BATCH_TRANSFORM = "per_batch_transform"
    COLLATE = "collate"
    PER_SAMPLE_TRANSFORM_ON_DEVICE = "per_sample_transform_on_device"
    PER_BATCH_TRANSFORM_ON_DEVICE = "per_batch_transform_on_device"


class PreTransform(Properties):
    def get_state_dict(self) -> Dict[str, Any]:
        return {}

    def configure_transforms(self) -> Optional[Dict[str, Callable]]:
        """The default transforms to use.

        Will be overridden by transforms passed to the ``__init__``.
        """
        return None

    def __init__(
        self,
        running_stage: RunningStage,
        transform: Union[Callable, List, Dict[str, Callable]] = None,
        **tranform_kwargs,
    ):
        super().__init__()
        # used to keep track of provided transforms
        self._running_stage = running_stage
        self._collate_in_worker_from_transform: Optional[bool] = None

        self._tranform_kwargs = tranform_kwargs

        transform = transform or self._resolve_transforms(running_stage)
        self.transform = self._check_transforms(transform, running_stage)

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

    def per_sample_transform(self, sample: Any) -> Any:
        if isinstance(sample, list):
            return [self.current_transform(s) for s in sample]
        return self.current_transform(sample)

    def per_batch_transform(self, batch: Any) -> Any:
        """Transforms to apply to a whole batch (if possible use this for efficiency).

        .. note::     This option is mutually exclusive with :meth:`per_sample_transform_on_device`,     since if both
        are specified, uncollation has to be applied.
        """
        return self.current_transform(batch)

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

    def per_sample_transform_on_device(self, sample: Any) -> Any:
        """Transforms to apply to the data before the collation (per-sample basis).

        .. note::     This option is mutually exclusive with :meth:`per_batch_transform`,     since if both are
        specified, uncollation has to be applied. .. note::     This function won't be called within the dataloader
        workers, since to make that happen     each of the workers would have to create it's own CUDA-context which
        would pollute GPU memory (if on GPU).
        """
        return self.current_transform(sample)

    def per_batch_transform_on_device(self, batch: Any) -> Any:
        """Transforms to apply to a whole batch (if possible use this for efficiency).

        .. note::     This function won't be called within the dataloader workers, since to make that happen     each of
        the workers would have to create it's own CUDA-context which would pollute GPU memory (if on GPU).
        """
        return self.current_transform(batch)

    @classmethod
    def from_transform(
        cls,
        transform: TRANSFORM_TYPE,
        running_stage: RunningStage,
        transform_registry: Optional[FlashRegistry] = None,
    ) -> Optional["PreTransform"]:

        if isinstance(transform, PreTransform):
            transform.running_stage = running_stage
            return transform

        if isinstance(transform, Callable):
            return cls(running_stage, {PreTransformPlacement.PER_SAMPLE_TRANSFORM: transform})

        if isinstance(transform, tuple) or isinstance(transform, LightningEnum):
            enum, transform_kwargs = cls._sanetize_registry_transform(transform, transform_registry)
            transform_cls = transform_registry.get(enum)
            return transform_cls(running_stage, transform=None, **transform_kwargs)

        if not transform:
            return None

        raise MisconfigurationException(f"The format for the transform isn't correct. Found {transform}")

    def _resolve_transforms(self, running_stage: RunningStage) -> Optional[Dict[str, Callable]]:
        from flash.core.data.data_pipeline import DataPipeline

        resolved_function = getattr(
            self, DataPipeline._resolve_function_hierarchy("configure_transforms", self, running_stage, PreTransform)
        )
        transforms: Optional[Dict[str, Callable]] = resolved_function(**self._tranform_kwargs)
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

        if isinstance(transform, list):
            transform = {"pre_tensor_transform": ApplyToKeys(DefaultDataKeys.INPUT, torch.nn.Sequential(*transform))}
        elif callable(transform):
            transform = {"pre_tensor_transform": ApplyToKeys(DefaultDataKeys.INPUT, transform)}

        if not isinstance(transform, Dict):
            raise MisconfigurationException(
                "Transform should be a dict. "
                f"Here are the available keys for your transforms: {[v for v in PreTransformPlacement]}."
            )

        keys_diff = set(transform.keys()).difference([v for v in PreTransformPlacement])

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

    @classmethod
    def _sanetize_registry_transform(
        cls, transform: Tuple[LightningEnum, Any], transform_registry: Optional[FlashRegistry]
    ) -> Tuple[LightningEnum, Dict]:
        msg = "The transform should be provided as a tuple with the following types (LightningEnum, Dict[str, Any]) "
        msg += "when requesting transform from the registry."
        if not transform_registry:
            raise MisconfigurationException("You requested a transform from the registry, but it is empty.")
        if isinstance(transform, tuple) and len(transform) > 2:
            raise MisconfigurationException(msg)
        if isinstance(transform, LightningEnum):
            enum = transform
            transform_kwargs = {}
        else:
            enum, transform_kwargs = transform
        if not isinstance(enum, LightningEnum):
            raise MisconfigurationException(msg)
        if not isinstance(transform_kwargs, Dict):
            raise MisconfigurationException(msg)
        return enum, transform_kwargs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(running_stage={self.running_stage}, transform={self.transform})"

    def __getitem__(self, placement: PreTransformPlacement) -> Callable:
        return self.transform[placement]
