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
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.enums import LightningEnum
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from flash.core.data.callback import ControlFlow
from flash.core.data.transforms import ApplyToKeys
from flash.core.data.utilities.collate import default_collate
from flash.core.data.utils import _INPUT_TRANSFORM_FUNCS, _STAGES_PREFIX
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


INVALID_STAGES_FOR_INPUT_TRANSFORMS = [RunningStage.SANITY_CHECKING, RunningStage.TUNING]


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
class _InputTransformPerStage:
    collate_in_worker_from_transform: Optional[bool] = None
    transforms: Optional[Dict[str, Callable]] = None


@dataclass
class InputTransform:
    def __post_init__(self):
        self.callbacks: Optional[List] = None

        # used to keep track of provided transforms
        self._transform: Dict[RunningStage, _InputTransformPerStage] = {}

        # For all the stages possible, set/load the transforms.
        for stage in RunningStage:
            if stage not in INVALID_STAGES_FOR_INPUT_TRANSFORMS:
                self._populate_transforms_for_stage(stage)

    def current_transform(self, stage: RunningStage, current_fn: str) -> Callable:
        if stage in [RunningStage.SANITY_CHECKING, RunningStage.TUNING]:
            raise KeyError(
                f"Transforms are only defined for stages:"
                f"\t{[stage for stage in RunningStage if stage not in INVALID_STAGES_FOR_INPUT_TRANSFORMS]}"
                f"But received {stage} instead."
            )

        # Check is transforms are present and the key is from the Enum defined above.
        if InputTransformPlacement.from_str(current_fn) is None:
            raise KeyError(
                f"{[fn for fn in InputTransformPlacement]} are the only allowed keys to retreive the transform."
                f"But received {current_fn} instead."
            )
        return self._transform[stage].transforms.get(current_fn, self._identity)

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
        pass

    def input_per_sample_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of each sample on
        device for all stages stage."""
        pass

    def target_per_sample_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of each sample on
        device for all stages stage."""
        pass

    def train_per_sample_transform(self) -> Callable:
        """Defines the transform to be applied on a single sample on cpu for the training stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }
        """
        pass

    def train_input_per_sample_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of each single sample
        on cpu for the training stage."""
        pass

    def train_target_per_sample_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of each single sample
        on cpu for the training stage."""
        pass

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
        pass

    def val_input_per_sample_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of each single sample
        on cpu for the validating stage."""
        pass

    def val_target_per_sample_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of each single sample
        on cpu for the validating stage."""
        pass

    def test_per_sample_transform(self) -> Callable:
        """Defines the transform to be applied on a single sample on cpu for the testing stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }
        """
        pass

    def test_input_per_sample_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of each single sample
        on cpu for the testing stage."""
        pass

    def test_target_per_sample_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of each single sample
        on cpu for the testing stage."""
        pass

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
        pass

    def predict_input_per_sample_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of each single sample
        on cpu for the predicting stage."""
        pass

    def predict_target_per_sample_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of each single sample
        on cpu for the predicting stage."""
        pass

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
        pass

    def serve_input_per_sample_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of each single sample
        on cpu for the serving stage."""
        pass

    def serve_target_per_sample_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of each single sample
        on cpu for the serving stage."""
        pass

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
        pass

    def input_per_sample_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of each sample on
        device for all stages stage."""
        pass

    def target_per_sample_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of each sample on
        device for all stages stage."""
        pass

    def train_per_sample_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on a single sample on device for the training stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }
        """
        pass

    def train_input_per_sample_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of each single sample
        on device for the training stage."""
        pass

    def train_target_per_sample_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of each single sample
        on device for the training stage."""
        pass

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
        pass

    def val_input_per_sample_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of each single sample
        on device for the validating stage."""
        pass

    def val_target_per_sample_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of each single sample
        on device for the validating stage."""
        pass

    def test_per_sample_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on a single sample on device for the testing stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }
        """
        pass

    def test_input_per_sample_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of each single sample
        on device for the testing stage."""
        pass

    def test_target_per_sample_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of each single sample
        on device for the testing stage."""
        pass

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
        pass

    def predict_input_per_sample_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of each single sample
        on device for the predicting stage."""
        pass

    def predict_target_per_sample_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of each single sample
        on device for the predicting stage."""
        pass

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
        pass

    def input_per_batch_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of batch on cpu for all
        stages stage."""
        pass

    def target_per_batch_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of batch on cpu for
        all stages stage."""
        pass

    def train_per_batch_transform(self) -> Callable:
        """Defines the transform to be applied on a batch of data on cpu for the training stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }
        """
        pass

    def train_input_per_batch_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of each single sample
        on cpu for the training stage."""
        pass

    def train_target_per_batch_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of each single sample
        on cpu for the training stage."""
        pass

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
        pass

    def val_input_per_batch_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of each single sample
        on cpu for the validating stage."""
        pass

    def val_target_per_batch_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of each single sample
        on cpu for the validating stage."""
        pass

    def test_per_batch_transform(self) -> Callable:
        """Defines the transform to be applied on a batch of data on cpu for the testing stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }
        """
        pass

    def test_input_per_batch_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of each single sample
        on cpu for the testing stage."""
        pass

    def test_target_per_batch_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of each single sample
        on cpu for the testing stage."""
        pass

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
        pass

    def predict_input_per_batch_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of each single sample
        on cpu for the predicting stage."""
        pass

    def predict_target_per_batch_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of each single sample
        on cpu for the predicting stage."""
        pass

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
        pass

    def serve_input_per_batch_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of each single sample
        on cpu for the serving stage."""
        pass

    def serve_target_per_batch_transform(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of each single sample
        on cpu for the serving stage."""
        pass

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
        pass

    def input_per_batch_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of batch on device for
        all stages stage."""
        pass

    def target_per_batch_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of batch on device for
        all stages stage."""
        pass

    def train_per_batch_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on a batch of data on device for the training stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }
        """
        pass

    def train_input_per_batch_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of each single sample
        on device for the training stage."""
        pass

    def train_target_per_batch_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of each single sample
        on device for the training stage."""
        pass

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
        pass

    def val_input_per_batch_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of each single sample
        on device for the validating stage."""
        pass

    def val_target_per_batch_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of each single sample
        on device for the validating stage."""
        pass

    def test_per_batch_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on a batch of data on device for the testing stage.

        The input data of the transform would have the following form::

            {
                DataKeys.INPUT: ...,
                DataKeys.TARGET: ...,
                DataKeys.METADATA: ...,
            }
        """
        pass

    def test_input_per_batch_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of each single sample
        on device for the testing stage."""
        pass

    def test_target_per_batch_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of each single sample
        on device for the testing stage."""
        pass

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
        pass

    def predict_input_per_batch_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "input" key of each single sample
        on device for the predicting stage."""
        pass

    def predict_target_per_batch_transform_on_device(self) -> Callable:
        """Defines the transform to be applied on the value associated with the "target" key of each single sample
        on device for the predicting stage."""
        pass

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

    def _per_sample_transform(self, sample: Any, stage: RunningStage) -> Any:
        fn = self.current_transform(stage=stage, current_fn="per_sample_transform")
        if isinstance(sample, list):
            return [fn(s) for s in sample]
        return fn(sample)

    def _per_batch_transform(self, batch: Any, stage: RunningStage) -> Any:
        """Transforms to apply to a whole batch (if possible use this for efficiency).

        .. note:: This option is mutually exclusive with :meth:`per_sample_transform_on_device`, since if both are
        specified, uncollation has to be applied.
        """
        return self.current_transform(stage=stage, current_fn="per_batch_transform")(batch)

    def _collate(self, samples: Sequence, stage: RunningStage) -> Any:
        """Transform to convert a sequence of samples to a collated batch."""
        return self.current_transform(stage=stage, current_fn="collate")(samples)

    def _per_sample_transform_on_device(self, sample: Any, stage: RunningStage) -> Any:
        """Transforms to apply to the data before the collation (per-sample basis).

        .. note::     This option is mutually exclusive with :meth:`per_batch_transform`,     since if both are
        specified, uncollation has to be applied. .. note::     This function won't be called within the dataloader
        workers, since to make that happen     each of the workers would have to create it's own CUDA-context which
        would pollute GPU memory (if on GPU).
        """
        fn = self.current_transform(stage=stage, current_fn="per_sample_transform_on_device")
        if isinstance(sample, list):
            return [fn(s) for s in sample]
        return fn(sample)

    def _per_batch_transform_on_device(self, batch: Any, stage: RunningStage) -> Any:
        """Transforms to apply to a whole batch (if possible use this for efficiency).

        .. note::     This function won't be called within the dataloader workers, since to make that happen     each of
        the workers would have to create it's own CUDA-context which would pollute GPU memory (if on GPU).
        """
        return self.current_transform(stage=stage, current_fn="per_batch_transform_on_device")(batch)

    #############
    # UTILITIES #
    #############

    def inject_collate_fn(self, collate_fn: Callable):
        # For all the stages possible, set collate function
        if collate_fn is not default_collate:
            for stage in RunningStage:
                if stage not in [RunningStage.SANITY_CHECKING, RunningStage.TUNING]:
                    self._transform[stage].transforms[InputTransformPlacement.COLLATE.value] = collate_fn

    def _populate_transforms_for_stage(self, running_stage: RunningStage):
        transform, collate_in_worker = self.__check_transforms(
            transform=self.__resolve_transforms(running_stage), stage=running_stage
        )
        if self._transform is None:
            self._transform = {}
        self._transform[running_stage] = _InputTransformPerStage(
            collate_in_worker_from_transform=collate_in_worker,
            transforms=transform,
        )

    def __resolve_transforms(self, running_stage: RunningStage) -> Optional[Dict[str, Callable]]:
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

                if fn is None:
                    continue

                if not callable(fn):
                    raise MisconfigurationException(f"The hook {method_name} should return a function.")

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

    def __check_transforms(
        self, transform: Optional[Dict[str, Callable]], stage: RunningStage
    ) -> Tuple[Optional[Dict[str, Callable]], Optional[bool]]:
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

        return transform, collate_in_worker

    @staticmethod
    def _identity(x: Any) -> Any:
        return x

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(" + f"running_stage={self.running_stage}, transform={self._transform})"

    def __getitem__(self, placement: InputTransformPlacement) -> Callable:
        return self._transform[placement]


@dataclass
class LambdaInputTransform(InputTransform):

    transform: Callable = InputTransform._identity

    def per_sample_transform(self) -> Callable:
        return self.transform


def create_or_configure_input_transform(
    transform: INPUT_TRANSFORM_TYPE,
    transform_kwargs: Optional[Dict] = None,
) -> Optional[InputTransform]:

    if not transform_kwargs:
        transform_kwargs = {}

    if isinstance(transform, InputTransform):
        return transform

    if inspect.isclass(transform) and issubclass(transform, InputTransform):
        # Deprecation Warning
        rank_zero_warn(
            "Please pass an instantiated object of the `InputTransform` class. Passing the Class and keyword arguments"
            " separately has been deprecated since v0.8.0 and will be removed in v0.9.0.",
            stacklevel=8,
            category=FutureWarning,
        )
        return transform(**transform_kwargs)

    if isinstance(transform, partial):
        return transform(**transform_kwargs)

    if isinstance(transform, Callable):
        return LambdaInputTransform(
            transform=transform,
            **transform_kwargs,
        )

    if not transform:
        return None

    raise MisconfigurationException(f"The format for the transform isn't correct. Found {transform}")


class _InputTransformProcessor:
    """
    This class is used to encapsulate the following functions of an `InputTransform` Object:
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
    ):
        super().__init__()
        self.input_transform = input_transform
        self.callback = ControlFlow(self.input_transform.callbacks or [])
        self.collate_fn = collate_fn
        self.per_sample_transform = per_sample_transform
        self.per_batch_transform = per_batch_transform
        self.apply_per_sample_transform = apply_per_sample_transform
        self.stage = stage
        self.on_device = on_device

    def __call__(self, samples: Sequence[Any]) -> Any:
        if not self.on_device:
            for sample in samples:
                self.callback.on_load_sample(sample, self.stage)

        if self.apply_per_sample_transform:
            if not isinstance(samples, list):
                list_samples = [samples]
            else:
                list_samples = samples

            transformed_samples = [self.per_sample_transform(sample, self.stage) for sample in list_samples]

            for sample in transformed_samples:
                if self.on_device:
                    self.callback.on_per_sample_transform_on_device(sample, self.stage)
                else:
                    self.callback.on_per_sample_transform(sample, self.stage)

            collated_samples = self.collate_fn(transformed_samples, self.stage)
            self.callback.on_collate(collated_samples, self.stage)
        else:
            collated_samples = samples

        transformed_collated_samples = self.per_batch_transform(collated_samples, self.stage)
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


def __make_collates(input_transform: InputTransform, on_device: bool, collate: Callable) -> Tuple[Callable, Callable]:
    """Returns the appropriate collate functions based on whether the transforms happen in a DataLoader worker or
    on the device (main process)."""
    if on_device:
        return input_transform._identity, collate
    return collate, input_transform._identity


def __configure_worker_and_device_collate_fn(
    running_stage: RunningStage, input_transform: InputTransform
) -> Tuple[Callable, Callable]:

    from flash.core.data.data_pipeline import DataPipeline

    prefix: str = _STAGES_PREFIX[running_stage]
    transform_for_stage: _InputTransformPerStage = input_transform._transform[running_stage]

    per_batch_transform_overridden: bool = DataPipeline._is_overridden_recursive(
        "per_batch_transform", input_transform, InputTransform, prefix=prefix
    )

    per_sample_transform_on_device_overridden: bool = DataPipeline._is_overridden_recursive(
        "per_sample_transform_on_device", input_transform, InputTransform, prefix=prefix
    )

    is_per_overridden = per_batch_transform_overridden and per_sample_transform_on_device_overridden
    if transform_for_stage.collate_in_worker_from_transform is None and is_per_overridden:
        raise MisconfigurationException(
            f"{input_transform.__class__.__name__}: `per_batch_transform` and `per_sample_transform_on_device` "
            f"are mutually exclusive for stage {running_stage}"
        )

    if isinstance(transform_for_stage.collate_in_worker_from_transform, bool):
        worker_collate_fn, device_collate_fn = __make_collates(
            input_transform, not transform_for_stage.collate_in_worker_from_transform, input_transform._collate
        )
    else:
        worker_collate_fn, device_collate_fn = __make_collates(
            input_transform, per_sample_transform_on_device_overridden, input_transform._collate
        )

    worker_collate_fn = (
        worker_collate_fn.collate_fn if isinstance(worker_collate_fn, _InputTransformProcessor) else worker_collate_fn
    )

    return worker_collate_fn, device_collate_fn


def create_worker_input_transform_processor(
    running_stage: RunningStage, input_transform: InputTransform
) -> _InputTransformProcessor:
    """This utility is used to create the 2 `_InputTransformProcessor` objects which contain the transforms used as
    the DataLoader `collate_fn`."""
    worker_collate_fn, _ = __configure_worker_and_device_collate_fn(
        running_stage=running_stage, input_transform=input_transform
    )
    worker_input_transform_processor = _InputTransformProcessor(
        input_transform,
        worker_collate_fn,
        input_transform._per_sample_transform,
        input_transform._per_batch_transform,
        running_stage,
    )
    return worker_input_transform_processor


def create_device_input_transform_processor(
    running_stage: RunningStage, input_transform: InputTransform
) -> _InputTransformProcessor:
    """This utility is used to create a `_InputTransformProcessor` object which contain the transforms used as the
    DataModule `on_after_batch_transfer` hook."""
    _, device_collate_fn = __configure_worker_and_device_collate_fn(
        running_stage=running_stage, input_transform=input_transform
    )
    device_input_transform_processor = _InputTransformProcessor(
        input_transform,
        device_collate_fn,
        input_transform._per_sample_transform_on_device,
        input_transform._per_batch_transform_on_device,
        running_stage,
        apply_per_sample_transform=device_collate_fn != input_transform._identity,
        on_device=True,
    )
    return device_input_transform_processor
