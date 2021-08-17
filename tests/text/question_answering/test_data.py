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
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from flash.core.data.data_source import DefaultDataKeys
from flash.text import QuestionAnsweringData
from tests.helpers.utils import _TEXT_TESTING

TEST_BACKBONE = "distilbert-base-uncased"

TEST_DATA = {
    "id": ["12345", "12346", "12347", "12348"],
    "context": [
        "this is an answer one. this is a context one", "this is an answer two. this is a context two",
        "this is an answer three. this is a context three", "this is an answer four. this is a context four"
    ],
    "question": [
        "this is a question one", "this is a question two", "this is a question three", "this is a question four"
    ],
    "answer": [{
        "text": ["this is an answer one"],
        "answer_start": [0]
    }, {
        "text": ["this is an answer two"],
        "answer_start": [0]
    }, {
        "text": ["this is an answer three"],
        "answer_start": [0]
    }, {
        "text": ["this is an answer four"],
        "answer_start": [0]
    }]
}


def get_csv_data():
    df = pd.DataFrame(TEST_DATA)
    return df.to_csv(index=False)


def csv_data(tmpdir):
    path = Path(tmpdir) / "data.csv"
    path.write_text(get_csv_data())
    return path


def get_json_data():
    data = []
    examples = list(zip(TEST_DATA["id"], TEST_DATA["context"], TEST_DATA["question"], TEST_DATA["answer"]))
    for example in examples:
        data.append({"id": example[0], "context": example[1], "question": example[2], "answer": example[3]})
    return data


def json_data(tmpdir):
    data = get_json_data()
    json_data = ""
    for example in data:
        json_data += json.dumps(example) + "\n"
    path = Path(tmpdir) / "data.json"
    path.write_text(json_data)
    return path


def json_data_with_field(tmpdir):
    data = json.dumps({"data": get_json_data()})
    path = Path(tmpdir) / "data.json"
    path.write_text(data)
    return path


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
def test_from_csv(tmpdir):
    csv_path = csv_data(tmpdir)
    dm = QuestionAnsweringData.from_csv(
        question_column_name="question",
        context_column_name="context",
        answer_column_name="answer",
        backbone=TEST_BACKBONE,
        train_file=csv_path
    )
    batch = next(iter(dm.train_dataloader()))
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "start_positions" in batch
    assert "end_positions" in batch


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
def test_from_files(tmpdir):
    csv_path = csv_data(tmpdir)
    dm = QuestionAnsweringData.from_csv(
        question_column_name="question",
        context_column_name="context",
        answer_column_name="answer",
        backbone=TEST_BACKBONE,
        train_file=csv_path,
        val_file=csv_path,
        test_file=csv_path
    )
    batch = next(iter(dm.val_dataloader()))
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert DefaultDataKeys.METADATA in batch
    assert "context" in batch[DefaultDataKeys.METADATA][0]
    assert "answer" in batch[DefaultDataKeys.METADATA][0]
    assert "example_id" in batch[DefaultDataKeys.METADATA][0]
    assert "offset_mapping" in batch[DefaultDataKeys.METADATA][0]

    batch = next(iter(dm.test_dataloader()))
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert DefaultDataKeys.METADATA in batch
    assert "context" in batch[DefaultDataKeys.METADATA][0]
    assert "answer" in batch[DefaultDataKeys.METADATA][0]
    assert "example_id" in batch[DefaultDataKeys.METADATA][0]
    assert "offset_mapping" in batch[DefaultDataKeys.METADATA][0]


@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
def test_postprocess_tokenizer(tmpdir):
    """Tests that the tokenizer property in ``QuestionAnsweringPostprocess`` resolves correctly when a different backbone is
    used.
    """
    backbone = "allenai/longformer-base-4096"
    json_path = json_data(tmpdir)
    dm = QuestionAnsweringData.from_json(
        question_column_name="question",
        context_column_name="context",
        answer_column_name="answer",
        backbone=backbone,
        train_file=json_path,
        batch_size=2
    )
    pipeline = dm.data_pipeline
    pipeline.initialize()
    assert pipeline._postprocess_pipeline.backbone == backbone
    assert pipeline._postprocess_pipeline.tokenizer is not None


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
def test_from_json(tmpdir):
    json_path = json_data(tmpdir)
    dm = QuestionAnsweringData.from_json(
        question_column_name="question",
        context_column_name="context",
        answer_column_name="answer",
        backbone=TEST_BACKBONE,
        train_file=json_path,
        batch_size=2
    )
    batch = next(iter(dm.train_dataloader()))
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "start_positions" in batch
    assert "end_positions" in batch


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
def test_from_json_with_field(tmpdir):
    json_path = json_data_with_field(tmpdir)
    dm = QuestionAnsweringData.from_json(
        question_column_name="question",
        context_column_name="context",
        answer_column_name="answer",
        backbone=TEST_BACKBONE,
        train_file=json_path,
        field="data"
    )
    batch = next(iter(dm.train_dataloader()))
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "start_positions" in batch
    assert "end_positions" in batch
