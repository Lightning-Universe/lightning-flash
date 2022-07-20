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
from typing import Any, Callable, Collection, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np
import torch

from flash.core.data.base_viz import BaseVisualization
from flash.core.data.callback import BaseDataFetcher
from flash.core.data.data_module import DataModule
from flash.core.data.io.classification_input import ClassificationInputMixin
from flash.core.data.io.input import DataKeys, Input
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.utilities.classification import TargetFormatter
from flash.core.data.utilities.samples import to_samples
from flash.core.utilities.imports import _SKLEARN_AVAILABLE
from flash.core.utilities.stages import RunningStage
from flash.core.utilities.types import INPUT_TRANSFORM_TYPE

if _SKLEARN_AVAILABLE:
    from sklearn.utils import Bunch
else:
    Bunch = object


class TemplateNumpyClassificationInput(Input, ClassificationInputMixin):
    """An example data source that records ``num_features`` on the dataset."""

    num_features: int

    def load_data(
        self,
        examples: Collection[np.ndarray],
        targets: Optional[Sequence[Any]] = None,
        target_formatter: Optional[TargetFormatter] = None,
    ) -> Sequence[Dict[str, Any]]:
        """Sets the ``num_features`` attribute and calls ``super().load_data``.

        Args:
            examples: The ``np.ndarray`` (num_examples x num_features).
            targets: Associated targets.
            target_formatter: Optionally provide a ``TargetFormatter`` to control how targets are formatted.

        Returns:
            A sequence of samples / sample metadata.
        """
        if not self.predicting and isinstance(examples, np.ndarray):
            self.num_features = examples.shape[1]
        if targets is not None:
            self.load_target_metadata(targets, target_formatter=target_formatter)
        return to_samples(examples, targets)

    def load_sample(self, sample: Dict[str, Any]) -> Any:
        if DataKeys.TARGET in sample:
            sample[DataKeys.TARGET] = self.format_target(sample[DataKeys.TARGET])
        return sample


class TemplateSKLearnClassificationInput(TemplateNumpyClassificationInput):
    """An example data source that loads data from an sklearn data ``Bunch``."""

    def load_data(self, data: Bunch, target_formatter: Optional[TargetFormatter] = None) -> Sequence[Dict[str, Any]]:
        """Gets the ``data`` and ``target`` attributes from the ``Bunch`` and passes them to ``super().load_data``.

        Args:
            data: The scikit-learn data ``Bunch``.
            target_formatter: Optionally provide a ``TargetFormatter`` to control how targets are formatted.

        Returns:
            A sequence of samples / sample metadata.
        """
        return super().load_data(data.data, data.target, target_formatter=target_formatter)

    def predict_load_data(self, data: Bunch) -> Sequence[Dict[str, Any]]:
        """Avoid including targets when predicting.

        Args:
            data: The scikit-learn data ``Bunch``.

        Returns:
            A sequence of samples / sample metadata.
        """
        return super().load_data(data.data)


class TemplateInputTransform(InputTransform):
    """An example :class:`~flash.core.data.io.input_transform.InputTransform`."""

    @staticmethod
    def to_tensor(sample: Dict[str, Any]):
        """Transform which converts the sample to a tensor."""
        sample[DataKeys.INPUT] = torch.from_numpy(sample[DataKeys.INPUT]).float()

        if DataKeys.TARGET in sample:
            sample[DataKeys.TARGET] = torch.as_tensor(sample[DataKeys.TARGET]).long()

        return sample

    def per_sample_transform(self) -> Callable:
        return self.to_tensor


class TemplateData(DataModule):
    """To create our :class:`~flash.core.data.data_module.DataModule` we first set the ``input_transform_cls``
    attribute.

    Next, we add a ``from_numpy`` method and a ``from_sklearn`` method. Finally, we define the ``num_features`` property
    for convenience.
    """

    input_transform_cls = TemplateInputTransform

    @classmethod
    def from_numpy(
        cls,
        train_data: Optional[Collection[np.ndarray]] = None,
        train_targets: Optional[Collection[Any]] = None,
        val_data: Optional[Collection[np.ndarray]] = None,
        val_targets: Optional[Sequence[Any]] = None,
        test_data: Optional[Collection[np.ndarray]] = None,
        test_targets: Optional[Sequence[Any]] = None,
        predict_data: Optional[Collection[np.ndarray]] = None,
        input_cls: Type[Input] = TemplateNumpyClassificationInput,
        transform: INPUT_TRANSFORM_TYPE = TemplateInputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "TemplateData":
        """This is our custom ``from_*`` method. It expects numpy ``Array`` objects and targets as input and
        creates the ``TemplateData`` with them.

        Args:
            train_data: The numpy ``Array`` containing the train data.
            train_targets: The sequence of train targets.
            val_data: The numpy ``Array`` containing the validation data.
            val_targets: The sequence of validation targets.
            test_data: The numpy ``Array`` containing the test data.
            test_targets: The sequence of test targets.
            predict_data: The numpy ``Array`` containing the predict data.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
              :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed data module.
        """

        ds_kw = dict()

        train_input = input_cls(RunningStage.TRAINING, train_data, train_targets, **ds_kw)
        target_formatter = getattr(train_input, "target_formatter", None)

        return cls(
            train_input,
            input_cls(RunningStage.VALIDATING, val_data, val_targets, target_formatter=target_formatter, **ds_kw),
            input_cls(RunningStage.TESTING, test_data, test_targets, target_formatter=target_formatter, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_data, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @classmethod
    def from_sklearn(
        cls,
        train_bunch: Optional[Bunch] = None,
        val_bunch: Optional[Bunch] = None,
        test_bunch: Optional[Bunch] = None,
        predict_bunch: Optional[Bunch] = None,
        input_cls: Type[Input] = TemplateSKLearnClassificationInput,
        transform: INPUT_TRANSFORM_TYPE = TemplateInputTransform,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs: Any,
    ) -> "TemplateData":
        """This is our custom ``from_*`` method. It expects scikit-learn ``Bunch`` objects as input and creates the
        ``TemplateData`` with them.

        Args:
            train_bunch: The scikit-learn ``Bunch`` containing the train data.
            val_bunch: The scikit-learn ``Bunch`` containing the validation data.
            test_bunch: The scikit-learn ``Bunch`` containing the test data.
            predict_bunch: The scikit-learn ``Bunch`` containing the predict data.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
              :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed data module.
        """
        ds_kw = dict()

        train_input = input_cls(RunningStage.TRAINING, train_bunch, **ds_kw)
        target_formatter = getattr(train_input, "target_formatter", None)

        return cls(
            train_input,
            input_cls(RunningStage.VALIDATING, val_bunch, target_formatter=target_formatter, **ds_kw),
            input_cls(RunningStage.TESTING, test_bunch, target_formatter=target_formatter, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_bunch, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
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

    def show_load_sample(
        self,
        samples: List[Any],
        running_stage: RunningStage,
        limit_nb_samples: int = None,
        figsize: Tuple[int, int] = (6.4, 4.8),
    ):
        print(samples)

    def show_per_sample_transform(
        self,
        samples: List[Any],
        running_stage: RunningStage,
        limit_nb_samples: int = None,
        figsize: Tuple[int, int] = (6.4, 4.8),
    ):
        print(samples)
