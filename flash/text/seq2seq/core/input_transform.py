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
from dataclasses import dataclass
from typing import Callable, Optional

from pytorch_lightning.utilities.warnings import rank_zero_warn

from flash.core.data.io.input import DataKeys
from flash.core.integrations.transformers.input_transform import TransformersTextInputTransform
from flash.core.utilities.imports import _TRANSFORMERS_AVAILABLE

if _TRANSFORMERS_AVAILABLE:
    from transformers import DataCollatorForSeq2Seq
else:
    DataCollatorForSeq2Seq = object


@dataclass(unsafe_hash=True)
class Seq2SeqInputTransform(TransformersTextInputTransform):
    backbone: str = "t5-small"
    max_source_length: int = 128
    max_target_length: int = 128
    ignore_pad_token_for_loss: bool = True
    prefix: Optional[str] = None

    def __attach_prefix(self, input_text: str) -> str:
        if self.prefix is None and self.backbone in [
            "t5-small",
            "t5-base",
            "t5-large",
            "t5-3b",
            "t5-11b",
        ]:
            rank_zero_warn(
                "You're running a t5 model but didn't provide a prefix, which is the expected."
                "Eg: `prefix 'summarize: ' `",
                UserWarning,
            )

        prefix = self.prefix if self.prefix is not None else ""
        return prefix + input_text

    def tokenize_per_sample(self, sample):
        tokenized_sample = self._tokenize_call(
            text=self.__attach_prefix(sample[DataKeys.INPUT]),
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_source_length,
        )

        if DataKeys.TARGET in sample:
            with self.tokenizer.as_target_tokenizer():
                labels = self._tokenize_call(
                    text=sample[DataKeys.TARGET],
                    padding=self.padding,
                    truncation=self.truncation,
                    max_length=self.max_target_length,
                )

                if self._is_pad_to_max_length() and self.ignore_pad_token_for_loss:
                    # Since a single sample is being transformed.
                    # We have only one loop unlike HF example which transforms a batch.
                    label = labels["input_ids"]
                    labels["input_ids"] = [
                        token_id if token_id != self.tokenizer.pad_token_id else -100 for token_id in label
                    ]

                tokenized_sample[DataKeys.TARGET] = labels["input_ids"]
        return tokenized_sample

    def per_sample_transform(self) -> Callable:
        return self.tokenize_per_sample

    def collate(self) -> Callable:
        _collate_fn = DataCollatorForSeq2Seq(
            self.tokenizer,
            label_pad_token_id=self.tokenizer.pad_token_id,
            pad_to_multiple_of=None,
        )
        return _collate_fn
