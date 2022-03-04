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
from typing import Any, Optional, Union

from flash.core.data.io.input import DataKeys
from flash.core.integrations.transformers.collate import TransformersCollate
from flash.core.model import Task


@dataclass(unsafe_hash=True)
class TextQuestionAnsweringCollate(TransformersCollate):
    max_source_length: int = 384
    max_target_length: int = 30
    padding: Union[str, bool] = "max_length"
    doc_stride: int = 128
    model: Optional[Task] = None

    def _prepare_train_features(self, tokenizer, samples: Any, tokenized_samples: Any, pad_on_right: bool):
        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_samples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_samples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_samples["start_positions"] = []
        tokenized_samples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_samples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_samples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = samples["answer"][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_samples["start_positions"].append(cls_index)
                tokenized_samples["end_positions"].append(cls_index)
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
                    tokenized_samples["start_positions"].append(cls_index)
                    tokenized_samples["end_positions"].append(cls_index)
                else:
                    # Otherwise, move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_samples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_samples["end_positions"].append(token_end_index + 1)

        return tokenized_samples, sample_mapping, offset_mapping

    def _prepare_val_features(self, samples: Any, tokenized_samples: Any, pad_on_right: bool):
        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_samples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id, and we will store the offset mappings.
        tokenized_samples["example_id"] = []
        tokenized_samples["context"] = []
        tokenized_samples["answer"] = []

        for i in range(len(tokenized_samples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_samples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_samples["example_id"].append(samples["id"][sample_index])
            tokenized_samples["context"].append(samples["context"][sample_index])
            if "answer" in samples:
                tokenized_samples["answer"].append(samples["answer"][sample_index])

            # Set to None the offset_mapping that are not part of the context, so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_samples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_samples["offset_mapping"][i])
            ]

        return tokenized_samples

    def tokenize(self, samples: Any):
        pad_on_right = self.tokenizer.padding_side == "right"

        samples["question"] = [q.lstrip() for q in samples["question"]]

        tokenized_samples = self.tokenizer(
            samples["question" if pad_on_right else "context"],
            samples["context" if pad_on_right else "question"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=self.max_source_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=self.padding,
        )

        if "answer" in samples:
            tokenized_samples, _sample_mapping, _offset_mapping = self._prepare_train_features(
                self.tokenizer, samples, tokenized_samples, pad_on_right
            )

        if not self.model.training:
            if "answer" in samples:
                tokenized_samples["overflow_to_sample_mapping"] = _sample_mapping
                tokenized_samples["offset_mapping"] = _offset_mapping

            # InputTransform function for eval or predict
            tokenized_samples = self._prepare_val_features(samples, tokenized_samples, pad_on_right)

            offset_mappings = tokenized_samples.pop("offset_mapping")
            example_ids = tokenized_samples.pop("example_id")
            contexts = tokenized_samples.pop("context")

            tokenized_samples[DataKeys.METADATA] = []
            for offset_mapping, example_id, context in zip(offset_mappings, example_ids, contexts):
                tokenized_samples[DataKeys.METADATA].append(
                    {"context": context, "offset_mapping": offset_mapping, "example_id": example_id}
                )
            if "answer" in tokenized_samples:
                answers = tokenized_samples.pop("answer")
                for index, answer in enumerate(answers):
                    tokenized_samples[DataKeys.METADATA][index]["answer"] = answer

            del offset_mappings
            del example_ids
            del contexts
            del answers

        return tokenized_samples.data
