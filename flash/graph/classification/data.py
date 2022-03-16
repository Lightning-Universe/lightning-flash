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
from typing import Dict, Optional, Type

from torch.utils.data import Dataset

from flash.core.data.data_module import DataModule
from flash.core.data.io.input import Input
from flash.core.data.utilities.classification import TargetFormatter
from flash.core.utilities.imports import _GRAPH_TESTING
from flash.core.utilities.stages import RunningStage
from flash.core.utilities.types import INPUT_TRANSFORM_TYPE
from flash.graph.classification.input import GraphClassificationDatasetInput
from flash.graph.classification.input_transform import GraphClassificationInputTransform

# Skip doctests if requirements aren't available
if not _GRAPH_TESTING:
    __doctest_skip__ = ["GraphClassificationData", "GraphClassificationData.*"]


class GraphClassificationData(DataModule):
    """The ``GraphClassificationData`` class is a :class:`~flash.core.data.data_module.DataModule` with a set of
    classmethods for loading data for graph classification."""

    input_transform_cls = GraphClassificationInputTransform

    @classmethod
    def from_datasets(
        cls,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        predict_dataset: Optional[Dataset] = None,
        input_cls: Type[Input] = GraphClassificationDatasetInput,
        transform: INPUT_TRANSFORM_TYPE = GraphClassificationInputTransform,
        transform_kwargs: Optional[Dict] = None,
        target_formatter: Optional[TargetFormatter] = None,
        **data_module_kwargs,
    ) -> "GraphClassificationData":
        """Load the :class:`~flash.graph.classification.data.GraphClassificationData` from PyTorch Dataset objects.

        The Dataset objects should be one of the following:

        * A PyTorch Dataset where the ``__getitem__`` returns a tuple: ``(PyTorch Geometric Data object, target)``
        * A PyTorch Dataset where the ``__getitem__`` returns a dict:
            ``{"input": PyTorch Geometric Data object, "target": target}``

        To learn how to customize the transforms applied for each stage, read our
        :ref:`customizing transforms guide <customizing_transforms>`.

        Args:
            train_dataset: The Dataset to use when training.
            val_dataset: The Dataset to use when validating.
            test_dataset: The Dataset to use when testing.
            predict_dataset: The Dataset to use when predicting.
            target_formatter: Optionally provide a :class:`~flash.core.data.utilities.classification.TargetFormatter` to
                control how targets are handled. If ``None`` then no formatting will be applied to targets.
            input_cls: The :class:`~flash.core.data.io.input.Input` type to use for loading the data.
            transform: The :class:`~flash.core.data.io.input_transform.InputTransform` type to use.
            transform_kwargs: Dict of keyword arguments to be provided when instantiating the transforms.
            data_module_kwargs: Additional keyword arguments to provide to the
                :class:`~flash.core.data.data_module.DataModule` constructor.

        Returns:
            The constructed :class:`~flash.graph.classification.data.GraphClassificationData`.

        Examples
        ________

        A PyTorch Dataset where the ``__getitem__`` returns a tuple: ``(PyTorch Geometric Data object, target)``:

        .. doctest::

            >>> import torch
            >>> from torch.utils.data import Dataset
            >>> from torch_geometric.data import Data
            >>> from flash import Trainer
            >>> from flash.graph import GraphClassificationData, GraphClassifier
            >>> from flash.core.data.utilities.classification import SingleLabelTargetFormatter
            >>>
            >>> class CustomDataset(Dataset):
            ...     def __init__(self, targets=None):
            ...         self.targets = targets
            ...     def __getitem__(self, index):
            ...         edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
            ...         x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
            ...         data = Data(x=x, edge_index=edge_index)
            ...         if self.targets is not None:
            ...             return data, self.targets[index]
            ...         return data
            ...     def __len__(self):
            ...         return len(self.targets) if self.targets is not None else 3
            ...
            >>> datamodule = GraphClassificationData.from_datasets(
            ...     train_dataset=CustomDataset(["cat", "dog", "cat"]),
            ...     predict_dataset=CustomDataset(),
            ...     target_formatter=SingleLabelTargetFormatter(labels=["cat", "dog"]),
            ...     batch_size=2,
            ... )
            >>> datamodule.num_features
            1
            >>> datamodule.num_classes
            2
            >>> datamodule.labels
            ['cat', 'dog']
            >>> model = GraphClassifier(num_features=datamodule.num_features, num_classes=datamodule.num_classes)
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...

        A PyTorch Dataset where the ``__getitem__`` returns a dict:
        ``{"input": PyTorch Geometric Data object, "target": target}``:

        .. doctest::

            >>> import torch  # noqa: F811
            >>> from torch.utils.data import Dataset
            >>> from torch_geometric.data import Data  # noqa: F811
            >>> from flash import Trainer
            >>> from flash.graph import GraphClassificationData, GraphClassifier
            >>> from flash.core.data.utilities.classification import SingleLabelTargetFormatter
            >>>
            >>> class CustomDataset(Dataset):
            ...     def __init__(self, targets=None):
            ...         self.targets = targets
            ...     def __getitem__(self, index):
            ...         edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
            ...         x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
            ...         data = Data(x=x, edge_index=edge_index)
            ...         if self.targets is not None:
            ...             return {"input": data, "target": self.targets[index]}
            ...         return {"input": data}
            ...     def __len__(self):
            ...         return len(self.targets) if self.targets is not None else 3
            ...
            >>> datamodule = GraphClassificationData.from_datasets(
            ...     train_dataset=CustomDataset(["cat", "dog", "cat"]),
            ...     predict_dataset=CustomDataset(),
            ...     target_formatter=SingleLabelTargetFormatter(labels=["cat", "dog"]),
            ...     batch_size=2,
            ... )
            >>> datamodule.num_features
            1
            >>> datamodule.num_classes
            2
            >>> datamodule.labels
            ['cat', 'dog']
            >>> model = GraphClassifier(num_features=datamodule.num_features, num_classes=datamodule.num_classes)
            >>> trainer = Trainer(fast_dev_run=True)
            >>> trainer.fit(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Training...
            >>> trainer.predict(model, datamodule=datamodule)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
            Predicting...
        """

        ds_kw = dict(
            target_formatter=target_formatter,
        )

        train_input = input_cls(RunningStage.TRAINING, train_dataset, **ds_kw)
        ds_kw["target_formatter"] = getattr(train_input, "target_formatter", None)

        return cls(
            train_input,
            input_cls(RunningStage.VALIDATING, val_dataset, **ds_kw),
            input_cls(RunningStage.TESTING, test_dataset, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_dataset, **ds_kw),
            transform=transform,
            transform_kwargs=transform_kwargs,
            **data_module_kwargs,
        )

    @property
    def num_features(self):
        """The number of features per node in the graphs contained in this ``GraphClassificationData``."""
        n_cls_train = getattr(self.train_dataset, "num_features", None)
        n_cls_val = getattr(self.val_dataset, "num_features", None)
        n_cls_test = getattr(self.test_dataset, "num_features", None)
        return n_cls_train or n_cls_val or n_cls_test
