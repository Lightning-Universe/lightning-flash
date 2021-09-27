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
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch import Tensor

import flash
from flash.core.data.data_module import DataModule
from flash.core.data.data_source import DataSource, DefaultDataSources
from flash.core.data.process import Postprocess, Preprocess
from flash.core.data.properties import ProcessState
from flash.core.utilities.imports import _TEXT_AVAILABLE, requires
from flash.text.classification.data import TextDeserializer

if _TEXT_AVAILABLE:
    import datasets
    from datasets import DatasetDict, load_dataset
    from transformers import AutoTokenizer, default_data_collator


class Seq2SeqDataSource(DataSource):
    @requires("text")
    def __init__(
        self,
        backbone: str,
        max_source_length: int = 128,
        max_target_length: int = 128,
        padding: Union[str, bool] = "max_length",
        **backbone_kwargs,
    ):
        super().__init__()

        self.backbone = backbone
        self.backbone_kwargs = backbone_kwargs
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone, use_fast=True, **backbone_kwargs)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.padding = padding

    def _tokenize_fn(
        self,
        ex: Union[Dict[str, str], str],
        input: Optional[str] = None,
        target: Optional[str] = None,
    ) -> Callable:
        if isinstance(ex, dict):
            ex_input = ex[input]
            ex_target = ex[target] if target else None
        else:
            ex_input = ex
            ex_target = None

        model_inputs = self.tokenizer(
            ex_input,
            max_length=self.max_source_length,
            padding=self.padding,
            add_special_tokens=True,
            truncation=True,
        )
        if ex_target is not None:
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    ex_target,
                    max_length=self.max_target_length,
                    padding=self.padding,
                    add_special_tokens=True,
                    truncation=True,
                )
            model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def __getstate__(self):  # TODO: Find out why this is being pickled
        state = self.__dict__.copy()
        state.pop("tokenizer")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone, use_fast=True, **self.backbone_kwargs)


class Seq2SeqFileDataSource(Seq2SeqDataSource):
    def __init__(
        self,
        filetype: str,
        backbone: str,
        max_source_length: int = 128,
        max_target_length: int = 128,
        padding: Union[str, bool] = "max_length",
        **backbone_kwargs,
    ):
        super().__init__(backbone, max_source_length, max_target_length, padding, **backbone_kwargs)

        self.filetype = filetype

    def load_data(self, data: Any, columns: List[str] = None) -> "datasets.Dataset":
        if columns is None:
            columns = ["input_ids", "attention_mask", "labels"]
        if self.filetype == "json":
            file, input, target, field = data
        else:
            file, input, target = data
        data_files = {}
        stage = self._running_stage.value
        data_files[stage] = str(file)

        # FLASH_TESTING is set in the CI to run faster.
        if flash._IS_TESTING:
            try:
                if self.filetype == "json" and field is not None:
                    dataset_dict = DatasetDict(
                        {
                            stage: load_dataset(
                                self.filetype, data_files=data_files, split=[f"{stage}[:20]"], field=field
                            )[0]
                        }
                    )
                else:
                    dataset_dict = DatasetDict(
                        {stage: load_dataset(self.filetype, data_files=data_files, split=[f"{stage}[:20]"])[0]}
                    )
            except Exception:
                if self.filetype == "json" and field is not None:
                    dataset_dict = load_dataset(self.filetype, data_files=data_files, field=field)
                else:
                    dataset_dict = load_dataset(self.filetype, data_files=data_files)
        else:
            if self.filetype == "json" and field is not None:
                dataset_dict = load_dataset(self.filetype, data_files=data_files, field=field)
            else:
                dataset_dict = load_dataset(self.filetype, data_files=data_files)

        dataset_dict = dataset_dict.map(partial(self._tokenize_fn, input=input, target=target), batched=True)
        dataset_dict.set_format(columns=columns)
        return dataset_dict[stage]

    def predict_load_data(self, data: Any) -> Union["datasets.Dataset", List[Dict[str, torch.Tensor]]]:
        return self.load_data(data, columns=["input_ids", "attention_mask"])

    def __getstate__(self):  # TODO: Find out why this is being pickled
        state = self.__dict__.copy()
        state.pop("tokenizer")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone, use_fast=True, **self.backbone_kwargs)


class Seq2SeqCSVDataSource(Seq2SeqFileDataSource):
    def __init__(
        self,
        backbone: str,
        max_source_length: int = 128,
        max_target_length: int = 128,
        padding: Union[str, bool] = "max_length",
        **backbone_kwargs,
    ):
        super().__init__(
            "csv",
            backbone,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            padding=padding,
            **backbone_kwargs,
        )

    def __getstate__(self):  # TODO: Find out why this is being pickled
        state = self.__dict__.copy()
        state.pop("tokenizer")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone, use_fast=True, **self.backbone_kwargs)


class Seq2SeqJSONDataSource(Seq2SeqFileDataSource):
    def __init__(
        self,
        backbone: str,
        max_source_length: int = 128,
        max_target_length: int = 128,
        padding: Union[str, bool] = "max_length",
        **backbone_kwargs,
    ):
        super().__init__(
            "json",
            backbone,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            padding=padding,
            **backbone_kwargs,
        )

    def __getstate__(self):  # TODO: Find out why this is being pickled
        state = self.__dict__.copy()
        state.pop("tokenizer")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone, use_fast=True, **self.backbone_kwargs)


class Seq2SeqSentencesDataSource(Seq2SeqDataSource):
    def load_data(
        self,
        data: Union[str, List[str]],
        dataset: Optional[Any] = None,
    ) -> List[Any]:

        if isinstance(data, str):
            data = [data]
        return [self._tokenize_fn(s) for s in data]

    def __getstate__(self):  # TODO: Find out why this is being pickled
        state = self.__dict__.copy()
        state.pop("tokenizer")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone, use_fast=True, **self.backbone_kwargs)


@dataclass(unsafe_hash=True, frozen=True)
class Seq2SeqBackboneState(ProcessState):
    """The ``Seq2SeqBackboneState`` stores the backbone in use by the
    :class:`~flash.text.seq2seq.core.data.Seq2SeqPreprocess`
    """

    backbone: str
    backbone_kwargs: Dict[str, Any] = field(default_factory=dict)


class Seq2SeqPreprocess(Preprocess):
    @requires("text")
    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        backbone: str = "sshleifer/tiny-mbart",
        max_source_length: int = 128,
        max_target_length: int = 128,
        padding: Union[str, bool] = "max_length",
        **backbone_kwargs,
    ):
        self.backbone = backbone
        self.max_target_length = max_target_length
        self.max_source_length = max_source_length
        self.padding = padding

        super().__init__(
            train_transform=train_transform,
            val_transform=val_transform,
            test_transform=test_transform,
            predict_transform=predict_transform,
            data_sources={
                DefaultDataSources.CSV: Seq2SeqCSVDataSource(
                    self.backbone,
                    max_source_length=max_source_length,
                    max_target_length=max_target_length,
                    padding=padding,
                    **backbone_kwargs,
                ),
                DefaultDataSources.JSON: Seq2SeqJSONDataSource(
                    self.backbone,
                    max_source_length=max_source_length,
                    max_target_length=max_target_length,
                    padding=padding,
                    **backbone_kwargs,
                ),
                "sentences": Seq2SeqSentencesDataSource(
                    self.backbone,
                    max_source_length=max_source_length,
                    max_target_length=max_target_length,
                    padding=padding,
                    **backbone_kwargs,
                ),
            },
            default_data_source="sentences",
            deserializer=TextDeserializer(backbone, max_source_length),
        )

        self.set_state(Seq2SeqBackboneState(self.backbone, backbone_kwargs))

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            **self.transforms,
            "backbone": self.backbone,
            "max_source_length": self.max_source_length,
            "max_target_length": self.max_target_length,
            "padding": self.padding,
        }

    @classmethod
    def load_state_dict(cls, state_dict: Dict[str, Any], strict: bool):
        return cls(**state_dict)

    def collate(self, samples: Any) -> Tensor:
        """Override to convert a set of samples to a batch."""
        return default_data_collator(samples)


class Seq2SeqPostprocess(Postprocess):
    @requires("text")
    def __init__(self):
        super().__init__()

        self._backbone = None
        self._tokenizer = None

    @property
    def backbone_state(self):
        return self.get_state(Seq2SeqBackboneState)

    @property
    def tokenizer(self):
        if self.backbone_state is not None and self.backbone_state.backbone != self._backbone:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.backbone_state.backbone, use_fast=True, **self.backbone_state.backbone_kwargs
            )
            self._backbone = self.backbone_state.backbone
        return self._tokenizer

    def uncollate(self, generated_tokens: Any) -> Any:
        pred_str = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        pred_str = [str.strip(s) for s in pred_str]
        return pred_str

    def __getstate__(self):  # TODO: Find out why this is being pickled
        state = self.__dict__.copy()
        state.pop("_tokenizer")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._backbone = None
        _ = self.tokenizer


class Seq2SeqData(DataModule):
    """Data module for Seq2Seq tasks."""

    preprocess_cls = Seq2SeqPreprocess
    postprocess_cls = Seq2SeqPostprocess
