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

# Adapted from:
# https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_qa_no_trainer.py
# https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/utils_qa.py

from dataclasses import dataclass
from typing import Any, Callable

import torch

from flash.core.data.io.input import DataKeys
from flash.core.integrations.transformers.input_transform import TransformersTextInputTransform
from flash.core.utilities.imports import _TRANSFORMERS_AVAILABLE

if _TRANSFORMERS_AVAILABLE:
    from transformers import DataCollatorWithPadding, default_data_collator


@dataclass(unsafe_hash=True)
class QuestionAnsweringInputTransform(TransformersTextInputTransform):

    stride: int = 128
    backbone: str = "sshleifer/tiny-distilbert-base-cased-distilled-squad"
    max_source_length: int = 384
    max_target_length: int = 30

    @staticmethod
    def _set_start_and_end_positions(tokenizer, sample: Any, tokenized_sample: Any, pad_on_right: bool):
        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_sample.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_sample.pop("offset_mapping")

        # Let's label those examples!
        tokenized_sample["start_positions"] = []
        tokenized_sample["end_positions"] = []

        # for i, offsets in enumerate(offset_mapping):
        offsets = offset_mapping[0]
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_sample["input_ids"][0]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_sample.sequence_ids(0)

        # One example can give several spans, this is the index of the example containing this span of text.
        # sample_index = sample_mapping[0]
        answers = sample["answer"]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_sample["start_positions"].append(cls_index)
            tokenized_sample["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_sample["start_positions"].append(cls_index)
                tokenized_sample["end_positions"].append(cls_index)
            else:
                # Otherwise, move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_sample["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_sample["end_positions"].append(token_end_index + 1)

        tokenized_sample["overflow_to_sample_mapping"] = sample_mapping
        tokenized_sample["offset_mapping"] = offset_mapping
        return tokenized_sample

    @staticmethod
    def _generate_metadata(sample: Any, tokenized_sample: Any, pad_on_right: bool):
        # NOTE: Since this is a single sample transform, we don't need the `sample_mapping` variable
        #       But, later we might support long input chunking and this might be needed. Hence commenting this part.
        # # Since one example might give us several features if it has a long context, we need a map from a feature to
        # # its corresponding example. This key gives us just that.
        # sample_mapping = tokenized_sample.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id, and we will store the offset mappings.

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_sample.sequence_ids(0)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        tokenized_sample["example_id"] = sample["id"]
        tokenized_sample["context"] = sample["context"]
        if "answer" in sample:
            tokenized_sample["answer"] = sample["answer"]

        # Set to None the offset_mapping that are not part of the context, so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_sample["offset_mapping"] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_sample["offset_mapping"][0])
        ]

        offset_mapping = tokenized_sample.pop("offset_mapping")
        example_id = tokenized_sample.pop("example_id")
        context = tokenized_sample.pop("context")

        tokenized_sample[DataKeys.METADATA] = {
            "context": context,
            "offset_mapping": offset_mapping,
            "example_id": example_id,
        }
        if "answer" in tokenized_sample:
            answer = tokenized_sample.pop("answer")
            tokenized_sample[DataKeys.METADATA]["answer"] = answer
            del answer

        del offset_mapping
        del example_id
        del context

        return tokenized_sample

    def _sanitize_input_and_tokenize_input(self, sample: Any, pad_on_right: bool):
        sample["question"] = sample["question"].lstrip()
        tokenized_sample = self._tokenize_call(
            text=sample["question" if pad_on_right else "context"],
            text_pair=sample["context" if pad_on_right else "question"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=self.max_source_length,
            padding=self.padding,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )
        return tokenized_sample

    def train_val_test_tokenize_per_sample(self, sample):
        pad_on_right = self.tokenizer.padding_side == "right"
        tokenized_sample = self._sanitize_input_and_tokenize_input(sample, pad_on_right)
        tokenized_sample = QuestionAnsweringInputTransform._set_start_and_end_positions(
            tokenizer=self.tokenizer,
            sample=sample,
            tokenized_sample=tokenized_sample,
            pad_on_right=pad_on_right,
        )
        tokenized_sample = QuestionAnsweringInputTransform._generate_metadata(
            sample=sample,
            tokenized_sample=tokenized_sample,
            pad_on_right=pad_on_right,
        )
        return tokenized_sample

    def predict_tokenize_per_sample(self, sample):
        pad_on_right = self.tokenizer.padding_side == "right"
        tokenized_sample = self._sanitize_input_and_tokenize_input(sample, pad_on_right)
        tokenized_sample = QuestionAnsweringInputTransform._generate_metadata(
            sample=sample,
            tokenized_sample=tokenized_sample,
            pad_on_right=pad_on_right,
        )
        return tokenized_sample

    def per_sample_transform(self) -> Callable:
        return self.train_val_test_tokenize_per_sample

    def predict_per_sample_transform(self) -> Callable:
        return self.predict_tokenize_per_sample

    def predict_squeeze_inputs_for_batch(self, batch: Any):
        batch["input_ids"] = torch.squeeze(batch["input_ids"])
        batch["attention_mask"] = torch.squeeze(batch["attention_mask"])
        return batch

    def train_val_test_squeeze_inputs_for_batch(self, batch: Any):
        batch = self.predict_squeeze_inputs_for_batch(batch)
        batch["start_positions"] = torch.squeeze(batch["start_positions"])
        batch["end_positions"] = torch.squeeze(batch["end_positions"])
        return batch

    def per_batch_transform(self) -> Callable:
        return self.train_val_test_squeeze_inputs_for_batch

    def predict_per_batch_transform(self) -> Callable:
        return self.predict_squeeze_inputs_for_batch

    def collate(self) -> Callable:
        if self._is_pad_to_max_length():
            _collate_fn = default_data_collator
        else:
            _collate_fn = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=None)
        return _collate_fn
