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
from functools import partial
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from flash.core.data.io.classification_input import ClassificationInputMixin
from flash.core.data.io.input import DataKeys, Input
from flash.core.data.utilities.classification import MultiBinaryTargetFormatter, TargetFormatter
from flash.core.data.utilities.loading import load_data_frame
from flash.core.data.utilities.paths import PATH_TYPE
from flash.core.utilities.imports import _TEXT_AVAILABLE, requires

if _TEXT_AVAILABLE:
    from datasets import Dataset, load_dataset
else:
    Dataset = object


class TextClassificationInput(Input, ClassificationInputMixin):
    @staticmethod
    def _resolve_target(target_keys: Union[str, List[str]], element: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(target_keys, List):
            element[DataKeys.TARGET] = element.pop(target_keys)
        else:
            element[DataKeys.TARGET] = [element[target_key] for target_key in target_keys]
        return element

    @requires("text")
    def load_data(
        self,
        hf_dataset: Dataset,
        input_key: str,
        target_keys: Optional[Union[str, List[str]]] = None,
        target_formatter: Optional[TargetFormatter] = None,
    ) -> Dataset:
        """Loads data into HuggingFace datasets.Dataset."""
        if not self.predicting:
            hf_dataset = hf_dataset.map(partial(self._resolve_target, target_keys))
            targets = hf_dataset.to_dict()[DataKeys.TARGET]
            self.load_target_metadata(targets, target_formatter=target_formatter)

            # If we had binary multi-class targets then we also know the labels (column names)
            if isinstance(self.target_formatter, MultiBinaryTargetFormatter) and isinstance(target_keys, List):
                self.labels = target_keys

        # remove extra columns
        extra_columns = set(hf_dataset.column_names) - {input_key, DataKeys.TARGET}
        hf_dataset = hf_dataset.remove_columns(extra_columns)

        if input_key != DataKeys.INPUT:
            hf_dataset = hf_dataset.rename_column(input_key, DataKeys.INPUT)

        return hf_dataset

    def load_sample(self, sample: Dict[str, Any]) -> Any:
        if DataKeys.TARGET in sample:
            sample[DataKeys.TARGET] = self.format_target(sample[DataKeys.TARGET])
        return sample


class TextClassificationCSVInput(TextClassificationInput):
    @requires("text")
    def load_data(
        self,
        csv_file: PATH_TYPE,
        input_key: str,
        target_keys: Optional[Union[str, List[str]]] = None,
        target_formatter: Optional[TargetFormatter] = None,
    ) -> Dataset:
        return super().load_data(
            Dataset.from_pandas(load_data_frame(csv_file)), input_key, target_keys, target_formatter=target_formatter
        )


class TextClassificationJSONInput(TextClassificationInput):
    @requires("text")
    def load_data(
        self,
        json_file: PATH_TYPE,
        field: str,
        input_key: str,
        target_keys: Optional[Union[str, List[str]]] = None,
        target_formatter: Optional[TargetFormatter] = None,
    ) -> Dataset:
        dataset_dict = load_dataset("json", data_files={"data": str(json_file)}, field=field)
        return super().load_data(dataset_dict["data"], input_key, target_keys, target_formatter=target_formatter)


class TextClassificationDataFrameInput(TextClassificationInput):
    @requires("text")
    def load_data(
        self,
        data_frame: pd.DataFrame,
        input_key: str,
        target_keys: Optional[Union[str, List[str]]] = None,
        target_formatter: Optional[TargetFormatter] = None,
    ) -> Dataset:
        return super().load_data(
            Dataset.from_pandas(data_frame), input_key, target_keys, target_formatter=target_formatter
        )


class TextClassificationParquetInput(TextClassificationInput):
    @requires("text")
    def load_data(
        self,
        parquet_file: PATH_TYPE,
        input_key: str,
        target_keys: Optional[Union[str, List[str]]] = None,
        target_formatter: Optional[TargetFormatter] = None,
    ) -> Dataset:
        return super().load_data(
            Dataset.from_parquet(str(parquet_file)), input_key, target_keys, target_formatter=target_formatter
        )


class TextClassificationListInput(TextClassificationInput):
    @requires("text")
    def load_data(
        self,
        inputs: List[str],
        targets: Optional[List[Any]] = None,
        target_formatter: Optional[TargetFormatter] = None,
    ) -> Dataset:
        if targets is not None:
            hf_dataset = Dataset.from_dict({DataKeys.INPUT: inputs, DataKeys.TARGET: targets})
        else:
            hf_dataset = Dataset.from_dict({DataKeys.INPUT: inputs})
        return super().load_data(hf_dataset, DataKeys.INPUT, DataKeys.TARGET, target_formatter=target_formatter)
