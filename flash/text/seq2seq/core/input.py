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
from typing import Any, Dict, List, Optional, Union

import flash
from flash.core.data.io.input import DataKeys, Input
from flash.core.data.utilities.paths import PATH_TYPE
from flash.core.integrations.transformers.states import TransformersBackboneState
from flash.core.utilities.imports import _TEXT_AVAILABLE, requires

if _TEXT_AVAILABLE:
    from datasets import Dataset, load_dataset
else:
    Dataset = object


class Seq2SeqInputBase(Input):
    @requires("text")
    def load_data(
        self,
        hf_dataset: Dataset,
        input_key: str,
        target_key: Optional[str] = None,
        max_source_length: int = 128,
        max_target_length: int = 128,
        padding: Union[str, bool] = "max_length",
    ) -> Dataset:
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.padding = padding

        # remove extra columns
        extra_columns = set(hf_dataset.column_names) - {input_key, target_key}
        hf_dataset = hf_dataset.remove_columns(extra_columns)

        if input_key != DataKeys.INPUT:
            hf_dataset = hf_dataset.rename_column(input_key, DataKeys.INPUT)

        if target_key in hf_dataset.column_names and target_key != DataKeys.TARGET:
            hf_dataset = hf_dataset.rename_column(target_key, DataKeys.TARGET)

        if flash._IS_TESTING:
            # NOTE: must subset in this way to return a Dataset
            hf_dataset = hf_dataset.select(range(20))

        return hf_dataset

    def load_sample(self, sample: Dict[str, Any]) -> Any:
        tokenizer = self.get_state(TransformersBackboneState).tokenizer
        tokenized_sample = tokenizer(
            sample[DataKeys.INPUT],
            max_length=self.max_source_length,
            padding=self.padding,
            add_special_tokens=True,
            truncation=True,
        )
        tokenized_sample = tokenized_sample.data
        if DataKeys.TARGET in sample:
            with tokenizer.as_target_tokenizer():
                tokenized_sample[DataKeys.TARGET] = tokenizer(
                    sample[DataKeys.TARGET],
                    max_length=self.max_target_length,
                    padding=self.padding,
                    add_special_tokens=True,
                    truncation=True,
                )["input_ids"]
        return tokenized_sample


class Seq2SeqCSVInput(Seq2SeqInputBase):
    @requires("text")
    def load_data(
        self,
        csv_file: PATH_TYPE,
        input_key: str,
        target_key: Optional[str] = None,
        max_source_length: int = 128,
        max_target_length: int = 128,
        padding: Union[str, bool] = "max_length",
    ) -> Dataset:
        dataset_dict = load_dataset("csv", data_files={"data": str(csv_file)})
        return super().load_data(
            dataset_dict["data"],
            input_key,
            target_key,
            max_source_length,
            max_target_length,
            padding,
        )


class Seq2SeqJSONInput(Seq2SeqInputBase):
    @requires("text")
    def load_data(
        self,
        json_file: PATH_TYPE,
        field: str,
        input_key: str,
        target_key: Optional[str] = None,
        max_source_length: int = 128,
        max_target_length: int = 128,
        padding: Union[str, bool] = "max_length",
    ) -> Dataset:
        dataset_dict = load_dataset("json", data_files={"data": str(json_file)}, field=field)
        return super().load_data(
            dataset_dict["data"],
            input_key,
            target_key,
            max_source_length,
            max_target_length,
            padding,
        )


class Seq2SeqListInput(Seq2SeqInputBase):
    @requires("text")
    def load_data(
        self,
        inputs: List[str],
        targets: Optional[List[str]] = None,
        max_source_length: int = 128,
        max_target_length: int = 128,
        padding: Union[str, bool] = "max_length",
    ) -> Dataset:
        if targets is not None:
            hf_dataset = Dataset.from_dict({DataKeys.INPUT: inputs, DataKeys.TARGET: targets})
        else:
            hf_dataset = Dataset.from_dict({DataKeys.INPUT: inputs})
        return super().load_data(
            hf_dataset,
            DataKeys.INPUT,
            DataKeys.TARGET,
            max_source_length,
            max_target_length,
            padding,
        )
