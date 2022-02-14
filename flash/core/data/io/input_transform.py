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
from dataclasses import dataclass
from functools import partial, wraps
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from pytorch_lightning.utilities.enums import LightningEnum
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data._utils.collate import default_collate

from flash.core.data.callback import ControlFlow, FlashCallback
from flash.core.data.io.input import DataKeys
from flash.core.data.properties import Properties
from flash.core.data.transforms import ApplyToKeys
from flash.core.data.utils import _INPUT_TRANSFORM_FUNCS, _STAGES_PREFIX
from flash.core.registry import FlashRegistry
from flash.core.utilities.stages import RunningStage
from flash.core.utilities.types import INPUT_TRANSFORM_TYPE


class InputTransformPlacement(LightningEnum):

    PER_SAMPLE_TRANSFORM = "per_sample_transform"
    PER_BATCH_TRANSFORM = "per_batch_transform"
    COLLATE = "collate"
    PER_SAMPLE_TRANSFORM_ON_DEVICE = "per_sample_transform_on_device"
    PER_BATCH_TRANSFORM_ON_DEVICE = "per_batch_transform_on_device"


class ApplyToKeyPrefix(LightningEnum):

    INPUT = "input"
    TARGET = "target"


def transform_context(func: Callable, current_fn: str) -> Callable:
    @wraps(func)
    def wrapper(self, *args, **kwargs) -> Any:
        self.current_fn = current_fn
        result = func(self, *args, **kwargs)
        self.current_fn = None
        return result

    return wrapper


# Credit to Torchvision Team:
# https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#Compose
class Compose:
    """Composes several transforms together.

    This transform does not support torchscript.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"{t}"
        format_string += "\n)"
        return format_string


@dataclass
class InputTransform(Properties):

    running_stage: RunningStage

    def __post_init__(self):
        # used to keep track of provided transforms
        self._collate_in_worker_from_transform: Optional[bool] = None
        self._transform = None
        self._transform = self._check_transforms(self._resolve_transforms(self.running_stage), self.running_stage)

        # Hack
        Properties.__init__(self, running_stage=self.running_stage)

    @property
    def current_transform(self) -> Callable:
        if self._transform:
            return self._get_transform(self._transform)
        return self._identity

    @property
    def transforms(self) -> Dict[str, Optional[Dict[str, Callable]]]:
        """The transforms currently being used by this
        :class:`~flash.core.data.io.input_transform.InputTransform`."""
        return {
            "transform": self._transform,
        }

    ########################
    # PER SAMPLE TRANSFORM #
    ########################

    def per_sample_transform(self) -> Callable:
        """Defines the transform to be applied on a single sample on cpu for all stages stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }

        You would need to use :class:`flash.core.data.transforms.ApplyToKeys` as follows:

        .. code-block:: python

            from flash.core.data.transforms import ApplyToKeys


            class MyInputTransform(InputTransform):
                def per_sample_transform(self) -> Callable:

                    return ApplyToKeys("input", my_func)
        """
        return self._identity

    def input_per_sample_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of each sample on
        device for all stages stage."""
        return self._identity

    def target_per_sample_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of each sample on
        device for all stages stage."""
        return self._identity

    def train_per_sample_transform(self) -> Callable:
        """Defines the transform to be applied on a single sample on cpu for the training stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }
        """
        return self._identity

    def train_input_per_sample_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of each single sample
        on cpu for the training stage."""
        return self._identity

    def train_target_per_sample_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of each single sample
        on cpu for the training stage."""
        return self._identity

    def val_per_sample_transform(self) -> Callable:
        """Defines the transform to be applied on a single sample on cpu for the validating stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }

        You would need to use :class:`flash.core.data.transforms.ApplyToKeys` as follows:

        .. code-block:: python

            from flash.core.data.transforms import ApplyToKeys


            class MyInputTransform(InputTransform):
                def per_sample_transform(self) -> Callable:

                    return ApplyToKeys("input", my_func)
        """
        return self._identity

    def val_input_per_sample_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of each single sample
        on cpu for the validating stage."""
        return self._identity

    def val_target_per_sample_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of each single sample
        on cpu for the validating stage."""
        return self._identity

    def test_per_sample_transform(self) -> Callable:
        """Defines the transform to be applied on a single sample on cpu for the testing stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }
        """
        return self._identity

    def test_input_per_sample_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of each single sample
        on cpu for the testing stage."""
        return self._identity

    def test_target_per_sample_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of each single sample
        on cpu for the testing stage."""
        return self._identity

    def predict_per_sample_transform(self) -> Callable:
        """Defines the transform to be applied on a single sample on cpu for the predicting stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }

        You would need to use :class:`flash.core.data.transforms.ApplyToKeys` as follows:

        .. code-block:: python

            from flash.core.data.transforms import ApplyToKeys


            class MyInputTransform(InputTransform):
                def per_sample_transform(self) -> Callable:

                    return ApplyToKeys("input", my_func)
        """
        return self._identity

    def predict_input_per_sample_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of each single sample
        on cpu for the predicting stage."""
        return self._identity

    def predict_target_per_sample_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of each single sample
        on cpu for the predicting stage."""
        return self._identity

    def serve_per_sample_transform(self) -> Callable:
        """Defines the transform to be applied on a single sample on cpu for the serving stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }

        You would need to use :class:`flash.core.data.transforms.ApplyToKeys` as follows:

        .. code-block:: python

            from flash.core.data.transforms import ApplyToKeys


            class MyInputTransform(InputTransform):
                def per_sample_transform(self) -> Callable:

                    return ApplyToKeys("input", my_func)
        """
        return self._identity

    def serve_input_per_sample_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of each single sample
        on cpu for the serving stage."""
        return self._identity

    def serve_target_per_sample_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of each single sample
        on cpu for the serving stage."""
        return self._identity

    ##################################
    # PER SAMPLE TRANSFORM ON DEVICE #
    ##################################

    def per_sample_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on a single sample on device for all stages stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }

        You would need to use :class:`flash.core.data.transforms.ApplyToKeys` as follows:

        .. code-block:: python

            from flash.core.data.transforms import ApplyToKeys


            class MyInputTransform(InputTransform):
                def per_sample_transform_on_device(self) -> Callable:

                    return ApplyToKeys("input", my_func)
        """
        return self._identity

    def input_per_sample_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of each sample on
        device for all stages stage."""
        return self._identity

    def target_per_sample_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of each sample on
        device for all stages stage."""
        return self._identity

    def train_per_sample_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on a single sample on device for the training stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }
        """
        return self._identity

    def train_input_per_sample_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of each single sample
        on device for the training stage."""
        return self._identity

    def train_target_per_sample_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of each single sample
        on device for the training stage."""
        return self._identity

    def val_per_sample_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on a single sample on device for the validating stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }

        You would need to use :class:`flash.core.data.transforms.ApplyToKeys` as follows:

        .. code-block:: python

            from flash.core.data.transforms import ApplyToKeys


            class MyInputTransform(InputTransform):
                def per_sample_transform_on_device(self) -> Callable:

                    return ApplyToKeys("input", my_func)
        """
        return self._identity

    def val_input_per_sample_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of each single sample
        on device for the validating stage."""
        return self._identity

    def val_target_per_sample_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of each single sample
        on device for the validating stage."""
        return self._identity

    def test_per_sample_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on a single sample on device for the testing stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }
        """
        return self._identity

    def test_input_per_sample_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of each single sample
        on device for the testing stage."""
        return self._identity

    def test_target_per_sample_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of each single sample
        on device for the testing stage."""
        return self._identity

    def predict_per_sample_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on a single sample on device for the predicting stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }

        You would need to use :class:`flash.core.data.transforms.ApplyToKeys` as follows:

        .. code-block:: python

            from flash.core.data.transforms import ApplyToKeys


            class MyInputTransform(InputTransform):
                def per_sample_transform_on_device(self) -> Callable:

                    return ApplyToKeys("input", my_func)
        """
        return self._identity

    def predict_input_per_sample_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of each single sample
        on device for the predicting stage."""
        return self._identity

    def predict_target_per_sample_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of each single sample
        on device for the predicting stage."""
        return self._identity

    #######################
    # PER BATCH TRANSFORM #
    #######################

    def per_batch_transform(self) -> Callable:
        """Defines the transform to be applied on a batch of data on cpu for all stages stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }

        You would need to use :class:`flash.core.data.transforms.ApplyToKeys` as follows:

        .. code-block:: python

            from flash.core.data.transforms import ApplyToKeys


            class MyInputTransform(InputTransform):
                def per_batch_transform(self) -> Callable:

                    return ApplyToKeys("input", my_func)
        """
        return self._identity

    def input_per_batch_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of batch on cpu for all
        stages stage."""
        return self._identity

    def target_per_batch_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of batch on cpu for
        all stages stage."""
        return self._identity

    def train_per_batch_transform(self) -> Callable:
        """Defines the transform to be applied on a batch of data on cpu for the training stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }
        """
        return self._identity

    def train_input_per_batch_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of each single sample
        on cpu for the training stage."""
        return self._identity

    def train_target_per_batch_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of each single sample
        on cpu for the training stage."""
        return self._identity

    def val_per_batch_transform(self) -> Callable:
        """Defines the transform to be applied on a batch of data on cpu for the validating stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }

        You would need to use :class:`flash.core.data.transforms.ApplyToKeys` as follows:

        .. code-block:: python

            from flash.core.data.transforms import ApplyToKeys


            class MyInputTransform(InputTransform):
                def per_batch_transform(self) -> Callable:

                    return ApplyToKeys("input", my_func)
        """
        return self._identity

    def val_input_per_batch_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of each single sample
        on cpu for the validating stage."""
        return self._identity

    def val_target_per_batch_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of each single sample
        on cpu for the validating stage."""
        return self._identity

    def test_per_batch_transform(self) -> Callable:
        """Defines the transform to be applied on a batch of data on cpu for the testing stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }
        """
        return self._identity

    def test_input_per_batch_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of each single sample
        on cpu for the testing stage."""
        return self._identity

    def test_target_per_batch_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of each single sample
        on cpu for the testing stage."""
        return self._identity

    def predict_per_batch_transform(self) -> Callable:
        """Defines the transform to be applied on a batch of data on cpu for the predicting stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }

        You would need to use :class:`flash.core.data.transforms.ApplyToKeys` as follows:

        .. code-block:: python

            from flash.core.data.transforms import ApplyToKeys


            class MyInputTransform(InputTransform):
                def per_batch_transform(self) -> Callable:

                    return ApplyToKeys("input", my_func)
        """
        return self._identity

    def predict_input_per_batch_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of each single sample
        on cpu for the predicting stage."""
        return self._identity

    def predict_target_per_batch_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of each single sample
        on cpu for the predicting stage."""
        return self._identity

    def serve_per_batch_transform(self) -> Callable:
        """Defines the transform to be applied on a batch of data on cpu for the serving stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }

        You would need to use :class:`flash.core.data.transforms.ApplyToKeys` as follows:

        .. code-block:: python

            from flash.core.data.transforms import ApplyToKeys


            class MyInputTransform(InputTransform):
                def per_batch_transform(self) -> Callable:

                    return ApplyToKeys("input", my_func)
        """
        return self._identity

    def serve_input_per_batch_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of each single sample
        on cpu for the serving stage."""
        return self._identity

    def serve_target_per_batch_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of each single sample
        on cpu for the serving stage."""
        return self._identity

    #################################
    # PER BATCH TRANSFORM ON DEVICE #
    #################################

    def per_batch_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on a batch of data on device for all stages stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }

        You would need to use :class:`flash.core.data.transforms.ApplyToKeys` as follows:

        .. code-block:: python

            from flash.core.data.transforms import ApplyToKeys


            class MyInputTransform(InputTransform):
                def per_batch_transform_on_device(self) -> Callable:

                    return ApplyToKeys("input", my_func)
        """
        return self._identity

    def input_per_batch_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of batch on device for
        all stages stage."""
        return self._identity

    def target_per_batch_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of batch on device for
        all stages stage."""
        return self._identity

    def train_per_batch_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on a batch of data on device for the training stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }
        """
        return self._identity

    def train_input_per_batch_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of each single sample
        on device for the training stage."""
        return self._identity

    def train_target_per_batch_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of each single sample
        on device for the training stage."""
        return self._identity

    def val_per_batch_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on a batch of data on device for the validating stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }

        You would need to use :class:`flash.core.data.transforms.ApplyToKeys` as follows:

        .. code-block:: python

            from flash.core.data.transforms import ApplyToKeys


            class MyInputTransform(InputTransform):
                def per_batch_transform_on_device(self) -> Callable:

                    return ApplyToKeys("input", my_func)
        """
        return self._identity

    def val_input_per_batch_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of each single sample
        on device for the validating stage."""
        return self._identity

    def val_target_per_batch_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of each single sample
        on device for the validating stage."""
        return self._identity

    def test_per_batch_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on a batch of data on device for the testing stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }
        """
        return self._identity

    def test_input_per_batch_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of each single sample
        on device for the testing stage."""
        return self._identity

    def test_target_per_batch_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of each single sample
        on device for the testing stage."""
        return self._identity

    def predict_per_batch_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on a batch of data on device for the predicting stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }

        You would need to use :class:`flash.core.data.transforms.ApplyToKeys` as follows:

        .. code-block:: python

            from flash.core.data.transforms import ApplyToKeys


            class MyInputTransform(InputTransform):
                def per_batch_transform_on_device(self) -> Callable:

                    return ApplyToKeys("input", my_func)
        """
        return self._identity

    def predict_input_per_batch_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of each single sample
        on device for the predicting stage."""
        return self._identity

    def predict_target_per_batch_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of each single sample
        on device for the predicting stage."""
        return self._identity

    ###########
    # COLLATE #
    ###########

    def train_collate(self) -> Callable:
        """Defines the transform to be applied on a list of training sample to create a training batch."""
        return default_collate

    def val_collate(self) -> Callable:
        """Defines the transform to be applied on a list of validating sample to create a validating batch."""
        return default_collate

    def test_collate(self) -> Callable:
        """Defines the transform to be applied on a list of testing sample to create a testing batch."""
        return default_collate

    def predict_collate(self) -> Callable:
        """Defines the transform to be applied on a list of predicting sample to create a predicting batch."""
        return default_collate

    def serve_collate(self) -> Callable:
        """Defines the transform to be applied on a list of serving sample to create a serving batch."""
        return default_collate

    def collate(self) -> Callable:
        """Defines the transform to be applied on a list of sample to create a batch for all stages."""
        return default_collate

    ########################################
    # HOOKS CALLED INTERNALLY WITHIN FLASH #
    ########################################

    @partial(transform_context, current_fn="per_sample_transform")
    def _per_sample_transform(self, sample: Any) -> Any:
        fn = self.current_transform
        if isinstance(sample, list):
            return [fn(s) for s in sample]
        return fn(sample)

    @partial(transform_context, current_fn="per_batch_transform")
    def _per_batch_transform(self, batch: Any) -> Any:
        """Transforms to apply to a whole batch (if possible use this for efficiency).

        .. note:: This option is mutually exclusive with :meth:`per_sample_transform_on_device`, since if both are
        specified, uncollation has to be applied.
        """
        return self.current_transform(batch)

    @partial(transform_context, current_fn="collate")
    def _collate(self, samples: Sequence, metadata=None) -> Any:
        """Transform to convert a sequence of samples to a collated batch."""
        collate_fn = self.current_transform
        parameters = inspect.signature(collate_fn).parameters
        if len(parameters) > 1 and DataKeys.METADATA in parameters:
            return collate_fn(samples, metadata)
        return collate_fn(samples)

    @partial(transform_context, current_fn="per_sample_transform_on_device")
    def _per_sample_transform_on_device(self, sample: Any) -> Any:
        """Transforms to apply to the data before the collation (per-sample basis).

        .. note::     This option is mutually exclusive with :meth:`per_batch_transform`,     since if both are
        specified, uncollation has to be applied. .. note::     This function won't be called within the dataloader
        workers, since to make that happen     each of the workers would have to create it's own CUDA-context which
        would pollute GPU memory (if on GPU).
        """
        fn = self.current_transform
        if isinstance(sample, list):
            return [fn(s) for s in sample]
        return fn(sample)

    @partial(transform_context, current_fn="per_batch_transform_on_device")
    def _per_batch_transform_on_device(self, batch: Any) -> Any:
        """Transforms to apply to a whole batch (if possible use this for efficiency).

        .. note::     This function won't be called within the dataloader workers, since to make that happen     each of
        the workers would have to create it's own CUDA-context which would pollute GPU memory (if on GPU).
        """
        return self.current_transform(batch)

    #############
    # UTILITIES #
    #############

    def _resolve_transforms(self, running_stage: RunningStage) -> Optional[Dict[str, Callable]]:
        from flash.core.data.data_pipeline import DataPipeline

        transforms_out = {}
        stage = _STAGES_PREFIX[running_stage]

        # iterate over all transforms hook name
        for transform_name in InputTransformPlacement:

            transforms = {}
            transform_name = transform_name.value

            # iterate over all prefixes
            for key in ApplyToKeyPrefix:

                # get the resolved hook name based on the current stage
                resolved_name = DataPipeline._resolve_function_hierarchy(
                    transform_name, self, running_stage, InputTransform
                )
                # check if the hook name is specialized
                is_specialized_name = resolved_name.startswith(stage)

                # get the resolved hook name for apply to key on the current stage
                resolved_apply_to_key_name = DataPipeline._resolve_function_hierarchy(
                    f"{key}_{transform_name}", self, running_stage, InputTransform
                )
                # check if resolved hook name for apply to key is specialized
                is_specialized_apply_to_key_name = resolved_apply_to_key_name.startswith(stage)

                # check if they are overridden by the user
                resolve_name_overridden = DataPipeline._is_overridden(resolved_name, self, InputTransform)
                resolved_apply_to_key_name_overridden = DataPipeline._is_overridden(
                    resolved_apply_to_key_name, self, InputTransform
                )

                if resolve_name_overridden and resolved_apply_to_key_name_overridden:
                    # if both are specialized or both aren't specialized, raise a exception
                    # It means there is priority to specialize hooks name.
                    if not (is_specialized_name ^ is_specialized_apply_to_key_name):
                        raise MisconfigurationException(
                            f"Only one of {resolved_name} or {resolved_apply_to_key_name} can be overridden."
                        )

                    method_name = resolved_name if is_specialized_name else resolved_apply_to_key_name
                else:
                    method_name = resolved_apply_to_key_name if resolved_apply_to_key_name_overridden else resolved_name

                # get associated transform
                try:
                    fn = getattr(self, method_name)()
                except AttributeError as e:
                    raise AttributeError(str(e) + ". Hint: Call super().__init__(...) after setting all attributes.")

                if not callable(fn):
                    raise MisconfigurationException(f"The hook {method_name} should return a function.")

                # if the default hook is used, it should return identity, skip it.
                if fn is self._identity:
                    continue

                # wrap apply to key hook into `ApplyToKeys` with the associated key.
                if method_name == resolved_apply_to_key_name:
                    fn = ApplyToKeys(key.value, fn)

                if method_name not in transforms:
                    transforms[method_name] = fn

            # store the transforms.
            if transforms:
                transforms = list(transforms.values())
                transforms_out[transform_name] = Compose(transforms) if len(transforms) > 1 else transforms[0]

        return transforms_out

    def _check_transforms(
        self, transform: Optional[Dict[str, Callable]], stage: RunningStage
    ) -> Optional[Dict[str, Callable]]:
        if transform is None:
            return transform

        keys_diff = set(transform.keys()).difference([v.value for v in InputTransformPlacement])

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

        self._collate_in_worker_from_transform = collate_in_worker
        return transform

    @staticmethod
    def _identity(x: Any) -> Any:
        return x

    def _get_transform(self, transform: Dict[str, Callable]) -> Callable:
        if self.current_fn in transform:
            return transform[self.current_fn]
        return self._identity

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(" + f"running_stage={self.running_stage}, transform={self._transform})"

    def __getitem__(self, placement: InputTransformPlacement) -> Callable:
        return self._transform[placement]


@dataclass
class LambdaInputTransform(InputTransform):

    transform: Callable = InputTransform._identity

    def per_sample_transform(self) -> Callable:
        return self.transform


def _sanitize_registry_transform(
    transform: Tuple[Union[LightningEnum, str], Any], input_transforms_registry: Optional[FlashRegistry]
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


def create_transform(
    transform: INPUT_TRANSFORM_TYPE,
    running_stage: RunningStage,
    input_transforms_registry: Optional[FlashRegistry] = None,
    transform_kwargs: Optional[Dict] = None,
) -> Optional["InputTransform"]:

    if not transform_kwargs:
        transform_kwargs = {}

    if isinstance(transform, InputTransform):
        return transform

    if inspect.isclass(transform) and issubclass(transform, InputTransform):
        return transform(running_stage=running_stage, **transform_kwargs)

    if isinstance(transform, Callable):
        return LambdaInputTransform(
            running_stage=running_stage,
            transform=transform,
            **transform_kwargs,
        )

    if isinstance(transform, tuple) or isinstance(transform, (LightningEnum, str)):
        enum, transform_kwargs = _sanitize_registry_transform(transform, input_transforms_registry)
        transform_cls = input_transforms_registry.get(enum)
        return transform_cls(running_stage, **transform_kwargs)

    if not transform:
        return None

    raise MisconfigurationException(f"The format for the transform isn't correct. Found {transform}")


def _make_collates(input_transform: "InputTransform", on_device: bool, collate: Callable) -> Tuple[Callable, Callable]:
    if on_device:
        return input_transform._identity, collate
    return collate, input_transform._identity


class _InputTransformProcessorV2:
    """
    This class is used to encapsulate the following functions of a InputTransformInputTransform Object:
    Inside a worker:
        per_sample_transform: Function to transform an individual sample
        collate: Function to merge sample into a batch
        per_batch_transform: Function to transform an individual batch

    Inside main process:
        per_sample_transform_on_device: Function to transform an individual sample
        collate: Function to merge sample into a batch
        per_batch_transform_on_device: Function to transform an individual batch
    """

    def __init__(
        self,
        input_transform: InputTransform,
        collate_fn: Callable,
        per_sample_transform: Callable,
        per_batch_transform: Callable,
        stage: RunningStage,
        apply_per_sample_transform: bool = True,
        on_device: bool = False,
        callbacks: Optional[List[FlashCallback]] = None,
    ):
        super().__init__()
        self.input_transform = input_transform
        self.callback = ControlFlow(callbacks or [])
        self.collate_fn = collate_fn
        self.per_sample_transform = per_sample_transform
        self.per_batch_transform = per_batch_transform
        self.apply_per_sample_transform = apply_per_sample_transform
        self.stage = stage
        self.on_device = on_device

    @staticmethod
    def _extract_metadata(
        samples: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
        metadata = [s.pop(DataKeys.METADATA, None) if isinstance(s, Mapping) else None for s in samples]
        return samples, metadata if any(m is not None for m in metadata) else None

    def __call__(self, samples: Sequence[Any]) -> Any:
        if not self.on_device:
            for sample in samples:
                self.callback.on_load_sample(sample, self.stage)

        if self.apply_per_sample_transform:
            if not isinstance(samples, list):
                list_samples = [samples]
            else:
                list_samples = samples

            transformed_samples = [self.per_sample_transform(sample) for sample in list_samples]

            for sample in transformed_samples:
                if self.on_device:
                    self.callback.on_per_sample_transform_on_device(sample, self.stage)
                else:
                    self.callback.on_per_sample_transform(sample, self.stage)

            extracted_samples, metadata = self._extract_metadata(transformed_samples)
            try:
                collated_samples = self.collate_fn(extracted_samples, metadata)
            except TypeError:
                collated_samples = self.collate_fn(extracted_samples)
            if metadata and isinstance(collated_samples, dict):
                collated_samples[DataKeys.METADATA] = metadata
            self.callback.on_collate(collated_samples, self.stage)
        else:
            collated_samples = samples

        transformed_collated_samples = self.per_batch_transform(collated_samples)
        if self.on_device:
            self.callback.on_per_batch_transform_on_device(transformed_collated_samples, self.stage)
        else:
            self.callback.on_per_batch_transform(transformed_collated_samples, self.stage)
        return transformed_collated_samples

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


def _create_collate_input_transform_processors(
    input_transform: "InputTransform", callbacks: List[FlashCallback]
) -> Tuple[_InputTransformProcessorV2, _InputTransformProcessorV2]:
    """This utility is used to create the 2 `_InputTransformProcessorV2` objects which contain the transforms used
    as the DataLoader `collate_fn` and the DataModule `on_after_batch_transfer` hook."""

    from flash.core.data.data_pipeline import DataPipeline

    prefix: str = _STAGES_PREFIX[input_transform.running_stage]

    per_batch_transform_overridden: bool = DataPipeline._is_overridden_recursive(
        "per_batch_transform", input_transform, InputTransform, prefix=prefix
    )

    per_sample_transform_on_device_overridden: bool = DataPipeline._is_overridden_recursive(
        "per_sample_transform_on_device", input_transform, InputTransform, prefix=prefix
    )

    is_per_overridden = per_batch_transform_overridden and per_sample_transform_on_device_overridden
    if input_transform._collate_in_worker_from_transform is None and is_per_overridden:
        raise MisconfigurationException(
            f"{input_transform.__class__.__name__}: `per_batch_transform` and `per_sample_transform_on_device` "
            f"are mutually exclusive for stage {input_transform.running_stage}"
        )

    if isinstance(input_transform._collate_in_worker_from_transform, bool):
        worker_collate_fn, device_collate_fn = _make_collates(
            input_transform, not input_transform._collate_in_worker_from_transform, input_transform._collate
        )
    else:
        worker_collate_fn, device_collate_fn = _make_collates(
            input_transform, per_sample_transform_on_device_overridden, input_transform._collate
        )

    worker_collate_fn = (
        worker_collate_fn.collate_fn if isinstance(worker_collate_fn, _InputTransformProcessorV2) else worker_collate_fn
    )

    worker_input_transform_processor = _InputTransformProcessorV2(
        input_transform,
        worker_collate_fn,
        input_transform._per_sample_transform,
        input_transform._per_batch_transform,
        input_transform.running_stage,
        callbacks=callbacks,
    )
    device_input_transform_processor = _InputTransformProcessorV2(
        input_transform,
        device_collate_fn,
        input_transform._per_sample_transform_on_device,
        input_transform._per_batch_transform_on_device,
        input_transform.running_stage,
        apply_per_sample_transform=device_collate_fn != input_transform._identity,
        on_device=True,
        callbacks=callbacks,
    )
    return worker_input_transform_processor, device_input_transform_processor
