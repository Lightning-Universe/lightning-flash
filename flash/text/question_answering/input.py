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
import json
from pathlib import Path
from typing import Any, Callable, Dict, Union

import flash
from flash.core.data.batch import default_uncollate
from flash.core.data.io.input import DataKeys, Input
from flash.core.data.utilities.paths import PATH_TYPE
from flash.core.integrations.transformers.states import TransformersBackboneState
from flash.core.utilities.imports import _TEXT_AVAILABLE, requires

if _TEXT_AVAILABLE:
    from datasets import Dataset, load_dataset
else:
    Dataset = object


class QuestionAnsweringInputBase(Input):
    def _tokenize_fn(self, samples: Any) -> Callable:
        tokenizer = self.get_state(TransformersBackboneState).tokenizer
        pad_on_right = tokenizer.padding_side == "right"

        samples[self.question_column_name] = [q.lstrip() for q in samples[self.question_column_name]]

        tokenized_samples = tokenizer(
            samples[self.question_column_name if pad_on_right else self.context_column_name],
            samples[self.context_column_name if pad_on_right else self.question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=self.max_source_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=self.padding,
        )

        if self.training:
            # InputTransform function for training
            tokenized_samples, _, _ = self._prepare_train_features(tokenizer, samples, tokenized_samples, pad_on_right)
        else:
            if self.validating or self.testing:
                tokenized_samples, _sample_mapping, _offset_mapping = self._prepare_train_features(
                    tokenizer, samples, tokenized_samples, pad_on_right
                )

                tokenized_samples["overflow_to_sample_mapping"] = _sample_mapping
                tokenized_samples["offset_mapping"] = _offset_mapping

            # InputTransform function for eval or predict
            tokenized_samples = self._prepare_val_features(samples, tokenized_samples, pad_on_right)

            offset_mappings = tokenized_samples.pop("offset_mapping")
            example_ids = tokenized_samples.pop("example_id")
            contexts = tokenized_samples.pop("context")
            answers = tokenized_samples.pop("answer")

            tokenized_samples[DataKeys.METADATA] = []
            for offset_mapping, example_id, context in zip(offset_mappings, example_ids, contexts):
                tokenized_samples[DataKeys.METADATA].append(
                    {"context": context, "offset_mapping": offset_mapping, "example_id": example_id}
                )
            if self.validating or self.testing:
                for index, answer in enumerate(answers):
                    tokenized_samples[DataKeys.METADATA][index]["answer"] = answer

            del offset_mappings
            del example_ids
            del contexts
            del answers

        return tokenized_samples

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
            answers = samples[self.answer_column_name][sample_index]
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
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
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
        # corresponding example_id and we will store the offset mappings.
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
            if self.validating or self.testing:
                tokenized_samples["answer"].append(samples["answer"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_samples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_samples["offset_mapping"][i])
            ]

        return tokenized_samples

    def _reshape_answer_column(self, sample: Any):
        text = sample.pop("answer_text")
        start = sample.pop("answer_start")
        if isinstance(text, str):
            text = [text]
        if isinstance(start, int):
            start = [start]
        sample["answer"] = {"text": text, "answer_start": start}
        return sample

    @requires("text")
    def load_data(
        self,
        hf_dataset: Dataset,
        max_source_length: int = 384,
        max_target_length: int = 30,
        padding: Union[str, bool] = "max_length",
        question_column_name: str = "question",
        context_column_name: str = "context",
        answer_column_name: str = "answer",
        doc_stride: int = 128,
    ) -> Dataset:
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.padding = padding
        self.question_column_name = question_column_name
        self.context_column_name = context_column_name
        self.answer_column_name = answer_column_name
        self.doc_stride = doc_stride

        if self.training or self.validating or self.testing:
            if self.answer_column_name == "answer":
                column_names = hf_dataset.column_names
                if "answer" not in column_names:
                    if "answer_text" in column_names and "answer_start" in column_names:
                        hf_dataset = hf_dataset.map(self._reshape_answer_column, batched=False)
                    else:
                        raise KeyError(
                            """Dataset must contain either \"answer\" key as dict type or "answer_text" and
                            "answer_start" as string and integer types."""
                        )
            if not isinstance(hf_dataset[self.answer_column_name][0], Dict):
                raise TypeError(
                    f'{self.answer_column_name} column should be of type dict with keys "text" and "answer_start"'
                )

        if flash._IS_TESTING:
            # NOTE: must subset in this way to return a Dataset
            hf_dataset = hf_dataset.select(range(20))

        return hf_dataset

    def load_sample(self, sample: Dict[str, Any]) -> Any:
        sample = {key: [value] for key, value in sample.items()}
        tokenized_sample = self._tokenize_fn(sample).data

        # The tokenize function can return multiple outputs for each input. So we uncollate them here
        return default_uncollate(tokenized_sample)


class QuestionAnsweringCSVInput(QuestionAnsweringInputBase):
    @requires("text")
    def load_data(
        self,
        csv_file: PATH_TYPE,
        max_source_length: int = 384,
        max_target_length: int = 30,
        padding: Union[str, bool] = "max_length",
        question_column_name: str = "question",
        context_column_name: str = "context",
        answer_column_name: str = "answer",
        doc_stride: int = 128,
    ) -> Dataset:
        dataset_dict = load_dataset("csv", data_files={"data": str(csv_file)})
        return super().load_data(
            dataset_dict["data"],
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            padding=padding,
            question_column_name=question_column_name,
            context_column_name=context_column_name,
            answer_column_name=answer_column_name,
            doc_stride=doc_stride,
        )


class QuestionAnsweringJSONInput(QuestionAnsweringInputBase):
    @requires("text")
    def load_data(
        self,
        json_file: PATH_TYPE,
        field: str,
        max_source_length: int = 384,
        max_target_length: int = 30,
        padding: Union[str, bool] = "max_length",
        question_column_name: str = "question",
        context_column_name: str = "context",
        answer_column_name: str = "answer",
        doc_stride: int = 128,
    ) -> Dataset:
        dataset_dict = load_dataset("json", data_files={"data": str(json_file)}, field=field)
        return super().load_data(
            dataset_dict["data"],
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            padding=padding,
            question_column_name=question_column_name,
            context_column_name=context_column_name,
            answer_column_name=answer_column_name,
            doc_stride=doc_stride,
        )


class QuestionAnsweringDictionaryInput(QuestionAnsweringInputBase):
    def load_data(
        self,
        data: Dict[str, Any],
        max_source_length: int = 384,
        max_target_length: int = 30,
        padding: Union[str, bool] = "max_length",
        question_column_name: str = "question",
        context_column_name: str = "context",
        answer_column_name: str = "answer",
        doc_stride: int = 128,
    ) -> Dataset:
        return super().load_data(
            Dataset.from_dict(data),
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            padding=padding,
            question_column_name=question_column_name,
            context_column_name=context_column_name,
            answer_column_name=answer_column_name,
            doc_stride=doc_stride,
        )


class QuestionAnsweringSQuADInput(QuestionAnsweringDictionaryInput):
    def load_data(
        self,
        json_file: PATH_TYPE,
        max_source_length: int = 384,
        max_target_length: int = 30,
        padding: Union[str, bool] = "max_length",
        question_column_name: str = "question",
        context_column_name: str = "context",
        answer_column_name: str = "answer",
        doc_stride: int = 128,
    ) -> Dataset:
        path = Path(json_file)
        with open(path, "rb") as f:
            squad_v_2_dict = json.load(f)

        ids = []
        titles = []
        contexts = []
        questions = []
        answers = []
        for topic in squad_v_2_dict["data"]:
            title = topic["title"]
            for comprehension in topic["paragraphs"]:
                context = comprehension["context"]
                for qa in comprehension["qas"]:
                    question = qa["question"]
                    id = qa["id"]

                    ids.append(id)
                    titles.append(title)
                    contexts.append(context)
                    questions.append(question)

                    if not self.predicting:
                        _answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                        _answers = [answer["text"] for answer in qa["answers"]]
                        answers.append(dict(text=_answers, answer_start=_answer_starts))

        data = {"id": ids, "title": titles, "context": contexts, "question": questions}
        if not self.predicting:
            data["answer"] = answers

        return super().load_data(
            data,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            padding=padding,
            question_column_name=question_column_name,
            context_column_name=context_column_name,
            answer_column_name=answer_column_name,
            doc_stride=doc_stride,
        )
