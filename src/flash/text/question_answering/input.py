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
from typing import Any, Dict

import flash
from flash.core.data.io.input import Input
from flash.core.data.utilities.loading import load_data_frame
from flash.core.data.utilities.paths import PATH_TYPE
from flash.core.utilities.imports import _TEXT_AVAILABLE, requires

if _TEXT_AVAILABLE:
    from datasets import Dataset, load_dataset
else:
    Dataset = object


class QuestionAnsweringInputBase(Input):
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
        question_column_name: str = "question",
        context_column_name: str = "context",
        answer_column_name: str = "answer",
    ) -> Dataset:
        column_names = hf_dataset.column_names

        if self.training or self.validating or self.testing:
            if answer_column_name == "answer":
                if "answer" not in column_names:
                    if "answer_text" in column_names and "answer_start" in column_names:
                        hf_dataset = hf_dataset.map(self._reshape_answer_column, batched=False)
                    else:
                        raise KeyError(
                            """Dataset must contain either \"answer\" key as dict type or "answer_text" and
                            "answer_start" as string and integer types."""
                        )
            if not isinstance(hf_dataset[answer_column_name][0], Dict):
                raise TypeError(
                    f'{answer_column_name} column should be of type dict with keys "text" and "answer_start"'
                )

            if answer_column_name in column_names and answer_column_name != "answer":
                hf_dataset = hf_dataset.rename_column(answer_column_name, "answer")

        if question_column_name in column_names and question_column_name != "question":
            hf_dataset = hf_dataset.rename_column(question_column_name, "question")

        if context_column_name in column_names and context_column_name != "context":
            hf_dataset = hf_dataset.rename_column(context_column_name, "context")

        if flash._IS_TESTING:
            # NOTE: must subset in this way to return a Dataset
            hf_dataset = [sample for sample in hf_dataset][:40]

        return hf_dataset


class QuestionAnsweringCSVInput(QuestionAnsweringInputBase):
    @requires("text")
    def load_data(
        self,
        csv_file: PATH_TYPE,
        question_column_name: str = "question",
        context_column_name: str = "context",
        answer_column_name: str = "answer",
    ) -> Dataset:
        return super().load_data(
            Dataset.from_pandas(load_data_frame(csv_file)),
            question_column_name=question_column_name,
            context_column_name=context_column_name,
            answer_column_name=answer_column_name,
        )


class QuestionAnsweringJSONInput(QuestionAnsweringInputBase):
    @requires("text")
    def load_data(
        self,
        json_file: PATH_TYPE,
        field: str,
        question_column_name: str = "question",
        context_column_name: str = "context",
        answer_column_name: str = "answer",
    ) -> Dataset:
        dataset_dict = load_dataset("json", data_files={"data": str(json_file)}, field=field)
        return super().load_data(
            dataset_dict["data"],
            question_column_name=question_column_name,
            context_column_name=context_column_name,
            answer_column_name=answer_column_name,
        )


class QuestionAnsweringDictionaryInput(QuestionAnsweringInputBase):
    def load_data(
        self,
        data: Dict[str, Any],
        question_column_name: str = "question",
        context_column_name: str = "context",
        answer_column_name: str = "answer",
    ) -> Dataset:
        return super().load_data(
            Dataset.from_dict(data),
            question_column_name=question_column_name,
            context_column_name=context_column_name,
            answer_column_name=answer_column_name,
        )


class QuestionAnsweringSQuADInput(QuestionAnsweringDictionaryInput):
    def load_data(
        self,
        json_file: PATH_TYPE,
        question_column_name: str = "question",
        context_column_name: str = "context",
        answer_column_name: str = "answer",
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
                    ids.append(qa["id"])
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
            question_column_name=question_column_name,
            context_column_name=context_column_name,
            answer_column_name=answer_column_name,
        )
