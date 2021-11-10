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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from flash.core.data.base_viz import BaseVisualization
from flash.core.data.callback import BaseDataFetcher
from flash.core.data.data_module import DataModule
from flash.core.data.io.input import DataKeys, InputFormat, LabelsState, NumpyInput
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.transforms import ApplyToKeys
from flash.core.utilities.imports import _SKLEARN_AVAILABLE
from flash.core.utilities.stages import RunningStage

if _SKLEARN_AVAILABLE:
    from sklearn.utils import Bunch
else:
    Bunch = object


class TemplateNumpyInput(NumpyInput):
    """An example data source that records ``num_features`` on the dataset.

    We extend
    :class:`~flash.core.data.io.input.NumpyInput` so that we can use ``super().load_data``.
    """

    def load_data(self, data: Tuple[np.ndarray, Sequence[Any]], dataset: Any) -> Sequence[Mapping[str, Any]]:
        """Sets the ``num_features`` attribute and calls ``super().load_data``.

        Args:
            data: The tuple of ``np.ndarray`` (num_examples x num_features) and associated targets.
            dataset: The object that we can set attributes (such as ``num_features``) on.

        Returns:
            A sequence of samples / sample metadata.
        """
        dataset.num_features = data[0].shape[1]
        return super().load_data(data, dataset)


class TemplateSKLearnInput(TemplateNumpyInput):
    """An example data source that loads data from an sklearn data ``Bunch``."""

    def load_data(self, data: Bunch, dataset: Any) -> Sequence[Mapping[str, Any]]:
        """Gets the ``data`` and ``target`` attributes from the ``Bunch`` and passes them to ``super().load_data``.

        Args:
            data: The scikit-learn data ``Bunch``.
            dataset: The object that we can set attributes (such as ``num_classes``) on.

        Returns:
            A sequence of samples / sample metadata.
        """
        dataset.num_classes = len(data.target_names)
        self.set_state(LabelsState(data.target_names))
        return super().load_data((data.data, data.target), dataset=dataset)

    def predict_load_data(self, data: Bunch) -> Sequence[Mapping[str, Any]]:
        """Avoid including targets when predicting.

        Args:
            data: The scikit-learn data ``Bunch``.

        Returns:
            A sequence of samples / sample metadata.
        """
        return super().predict_load_data(data.data)


class TemplateInputTransform(InputTransform):
    """An example :class:`~flash.core.data.io.input_transform.InputTransform`.

    Args:
        train_transform: The user-specified transforms to apply during training.
        val_transform: The user-specified transforms to apply during validation.
        test_transform: The user-specified transforms to apply during testing.
        predict_transform: The user-specified transforms to apply during prediction.
    """

    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
    ):
        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            inputs={
                InputFormat.NUMPY: TemplateNumpyInput(),
                "sklearn": TemplateSKLearnInput(),
            },
            default_input=InputFormat.NUMPY,
        )

    def get_state_dict(self) -> Dict[str, Any]:
        """For serialization, you have control over what to save with the ``get_state_dict`` method.

        It's usually a good idea to save the transforms. So we just return them here. If you had any other attributes
        you wanted to save, this is where you would return them.
        """
        return self.transforms

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool = False):
        """This methods gets whatever we returned from ``get_state_dict`` as an input.

        Now we re-create the class with the transforms we saved.
        """
        return cls(**state_dict)

    @staticmethod
    def input_to_tensor(input: np.ndarray):
        """Transform which creates a tensor from the given numpy ``ndarray`` and converts it to ``float``"""
        return torch.from_numpy(input).float()

    def default_transforms(self) -> Optional[Dict[str, Callable]]:
        """Configures the default ``to_tensor_transform``.

        Returns:
            Our dictionary of transforms.
        """
        return {
            "to_tensor_transform": nn.Sequential(
                ApplyToKeys(DataKeys.INPUT, self.input_to_tensor),
                ApplyToKeys(DataKeys.TARGET, torch.as_tensor),
            ),
        }

    # If we wanted to apply different transforms at a particular stage (e.g. during training), we can prepend: `train`,
    # `val`, `test`, or `predict`, and provide some different defaults linke this:
    # def train_default_transforms(self) -> Optional[Dict[str, Callable]]:


class TemplateData(DataModule):
    """Creating our :class:`~flash.core.data.data_module.DataModule` is as easy as setting the
    ``input_transform_cls`` attribute.

    We get the ``from_numpy`` method for free as we've configured a ``InputFormat.NUMPY`` data source. We'll also add a
    ``from_sklearn`` method so that we can use our ``TemplateSKLearnInput. Finally, we define the ``num_features``
    property for convenience.
    """

    input_transform_cls = TemplateInputTransform

    @classmethod
    def from_sklearn(
        cls,
        train_bunch: Optional[Bunch] = None,
        val_bunch: Optional[Bunch] = None,
        test_bunch: Optional[Bunch] = None,
        predict_bunch: Optional[Bunch] = None,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        data_fetcher: Optional[BaseDataFetcher] = None,
        input_transform: Optional[InputTransform] = None,
        val_split: Optional[float] = None,
        batch_size: int = 4,
        num_workers: int = 0,
        **input_transform_kwargs: Any,
    ):
        """This is our custom ``from_*`` method. It expects scikit-learn ``Bunch`` objects as input and passes them
        through to the :meth:`~flash.core.data.data_module.DataModule.from_` method underneath.

        Args:
            train_bunch: The scikit-learn ``Bunch`` containing the train data.
            val_bunch: The scikit-learn ``Bunch`` containing the validation data.
            test_bunch: The scikit-learn ``Bunch`` containing the test data.
            predict_bunch: The scikit-learn ``Bunch`` containing the predict data.
            train_transform: The dictionary of transforms to use during training which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            val_transform: The dictionary of transforms to use during validation which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            test_transform: The dictionary of transforms to use during testing which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            predict_transform: The dictionary of transforms to use during predicting which maps
                :class:`~flash.core.data.io.input_transform.InputTransform` hook names to callable transforms.
            data_fetcher: The :class:`~flash.core.data.callback.BaseDataFetcher` to pass to the
                :class:`~flash.core.data.data_module.DataModule`.
            input_transform: The :class:`~flash.core.data.data.InputTransform` to pass to the
                :class:`~flash.core.data.data_module.DataModule`. If ``None``, ``cls.input_transform_cls`` will be
                constructed and used.
            val_split: The ``val_split`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            batch_size: The ``batch_size`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            num_workers: The ``num_workers`` argument to pass to the :class:`~flash.core.data.data_module.DataModule`.
            input_transform_kwargs: Additional keyword arguments to use when constructing the input_transform.
                Will only be used if ``input_transform = None``.

        Returns:
            The constructed data module.
        """
        return super().from_input(
            "sklearn",
            train_bunch,
            val_bunch,
            test_bunch,
            predict_bunch,
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_fetcher=data_fetcher,
            input_transform=input_transform,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=num_workers,
            **input_transform_kwargs,
        )

    @property
    def num_features(self) -> Optional[int]:
        """Tries to get the ``num_features`` from each dataset in turn and returns the output."""
        n_fts_train = getattr(self.train_dataset, "num_features", None)
        n_fts_val = getattr(self.val_dataset, "num_features", None)
        n_fts_test = getattr(self.test_dataset, "num_features", None)
        return n_fts_train or n_fts_val or n_fts_test

    # OPTIONAL - Everything from this point onwards is an optional extra

    @staticmethod
    def configure_data_fetcher(*args, **kwargs) -> BaseDataFetcher:
        """We can, *optionally*, provide a data visualization callback using the ``configure_data_fetcher``
        method."""
        return TemplateVisualization(*args, **kwargs)


class TemplateVisualization(BaseVisualization):
    """The ``TemplateVisualization`` class is a :class:`~flash.core.data.callbacks.BaseVisualization` that just
    prints the data.

    If you want to provide a visualization with your task, you can override these hooks.
    """

    def show_load_sample(self, samples: List[Any], running_stage: RunningStage):
        print(samples)

    def show_pre_tensor_transform(self, samples: List[Any], running_stage: RunningStage):
        print(samples)

    def show_to_tensor_transform(self, samples: List[Any], running_stage: RunningStage):
        print(samples)

    def show_post_tensor_transform(self, samples: List[Any], running_stage: RunningStage):
        print(samples)

    def show_per_batch_transform(self, batch: List[Any], running_stage):
        print(batch)
