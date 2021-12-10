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

from flash.core.data.data_pipeline import DataPipelineState
from flash.core.data.io.input import Input
from flash.core.data.new_data_module import DataModule
from flash.core.utilities.stages import RunningStage
from flash.core.utilities.types import INPUT_TRANSFORM_TYPE
from flash.graph.classification.input import GraphClassificationDatasetInput
from flash.graph.classification.input_transform import GraphClassificationInputTransform


class GraphClassificationData(DataModule):
    """Data module for graph classification tasks."""

    input_transform_cls = GraphClassificationInputTransform

    @classmethod
    def from_datasets(
        cls,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        predict_dataset: Optional[Dataset] = None,
        train_transform: INPUT_TRANSFORM_TYPE = GraphClassificationInputTransform,
        val_transform: INPUT_TRANSFORM_TYPE = GraphClassificationInputTransform,
        test_transform: INPUT_TRANSFORM_TYPE = GraphClassificationInputTransform,
        predict_transform: INPUT_TRANSFORM_TYPE = GraphClassificationInputTransform,
        input_cls: Type[Input] = GraphClassificationDatasetInput,
        transform_kwargs: Optional[Dict] = None,
        **data_module_kwargs,
    ) -> "GraphClassificationData":

        ds_kw = dict(
            data_pipeline_state=DataPipelineState(),
            transform_kwargs=transform_kwargs,
            input_transforms_registry=cls.input_transforms_registry,
        )

        return cls(
            input_cls(RunningStage.TRAINING, train_dataset, transform=train_transform, **ds_kw),
            input_cls(RunningStage.VALIDATING, val_dataset, transform=val_transform, **ds_kw),
            input_cls(RunningStage.TESTING, test_dataset, transform=test_transform, **ds_kw),
            input_cls(RunningStage.PREDICTING, predict_dataset, transform=predict_transform, **ds_kw),
            **data_module_kwargs,
        )

    @property
    def num_features(self):
        n_cls_train = getattr(self.train_dataset, "num_features", None)
        n_cls_val = getattr(self.val_dataset, "num_features", None)
        n_cls_test = getattr(self.test_dataset, "num_features", None)
        return n_cls_train or n_cls_val or n_cls_test
