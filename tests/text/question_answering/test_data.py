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

TEST_CSV_DATA = {
    "id": ["12345", "12346", "12347", "12348"],
    "context": [
        "this is an answer one. this is a context one",
        "this is an answer two. this is a context two",
        "this is an answer three. this is a context three",
        "this is an answer four. this is a context four",
    ],
    "question": [
        "this is a question one",
        "this is a question two",
        "this is a question three",
        "this is a question four",
    ],
    "answer_text": [
        "this is an answer one",
        "this is an answer two",
        "this is an answer three",
        "this is an answer four",
    ],
    "answer_start": [0, 0, 0, 0],
}

TEST_JSON_DATA = [
    {
        "id": "12345",
        "context": "this is an answer one. this is a context one",
        "question": "this is a question one",
        "answer": {"text": ["this is an answer one"], "answer_start": [0]},
    },
    {
        "id": "12346",
        "context": "this is an answer two. this is a context two",
        "question": "this is a question two",
        "answer": {"text": ["this is an answer two"], "answer_start": [0]},
    },
    {
        "id": "12347",
        "context": "this is an answer three. this is a context three",
        "question": "this is a question three",
        "answer": {"text": ["this is an answer three"], "answer_start": [0]},
    },
    {
        "id": "12348",
        "context": "this is an answer four. this is a context four",
        "question": "this is a question four",
        "answer": {"text": ["this is an answer four"], "answer_start": [0]},
    },
]


def get_csv_data():
    df = pd.DataFrame(TEST_CSV_DATA)
    return df.to_csv(index=False)


def csv_data(tmpdir):
    path = Path(tmpdir) / "data.csv"
    path.write_text(get_csv_data())
    return path


def json_data(tmpdir, data):
    json_data = ""
    for example in data:
        json_data += json.dumps(example) + "\n"
    path = Path(tmpdir) / "data.json"
    path.write_text(json_data)
    return path


def json_data_with_field(tmpdir, data):
    data = json.dumps({"data": data})
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
        train_file=csv_path,
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
        test_file=csv_path,
    )
    batch = next(iter(dm.val_dataloader()))
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "start_positions" in batch
    assert "end_positions" in batch
    assert DefaultDataKeys.METADATA in batch
    assert "context" in batch[DefaultDataKeys.METADATA][0]
    assert "answer" in batch[DefaultDataKeys.METADATA][0]
    assert "example_id" in batch[DefaultDataKeys.METADATA][0]
    assert "offset_mapping" in batch[DefaultDataKeys.METADATA][0]

    batch = next(iter(dm.test_dataloader()))
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "start_positions" in batch
    assert "end_positions" in batch
    assert DefaultDataKeys.METADATA in batch
    assert "context" in batch[DefaultDataKeys.METADATA][0]
    assert "answer" in batch[DefaultDataKeys.METADATA][0]
    assert "example_id" in batch[DefaultDataKeys.METADATA][0]
    assert "offset_mapping" in batch[DefaultDataKeys.METADATA][0]


@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
def test_postprocess_tokenizer(tmpdir):
    """Tests that the tokenizer property in ``QuestionAnsweringPostprocess`` resolves correctly when a different
    backbone is used."""
    backbone = "allenai/longformer-base-4096"
    json_path = json_data(tmpdir, TEST_JSON_DATA)
    dm = QuestionAnsweringData.from_json(
        question_column_name="question",
        context_column_name="context",
        answer_column_name="answer",
        backbone=backbone,
        train_file=json_path,
        batch_size=2,
    )
    pipeline = dm.data_pipeline
    pipeline.initialize()
    assert pipeline._postprocess_pipeline.backbone == backbone
    assert pipeline._postprocess_pipeline.tokenizer is not None


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
def test_from_json(tmpdir):
    json_path = json_data(tmpdir, TEST_JSON_DATA)
    dm = QuestionAnsweringData.from_json(
        question_column_name="question",
        context_column_name="context",
        answer_column_name="answer",
        backbone=TEST_BACKBONE,
        train_file=json_path,
        batch_size=2,
    )
    batch = next(iter(dm.train_dataloader()))
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "start_positions" in batch
    assert "end_positions" in batch


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
def test_from_json_with_field(tmpdir):
    json_path = json_data_with_field(tmpdir, TEST_JSON_DATA)
    dm = QuestionAnsweringData.from_json(
        question_column_name="question",
        context_column_name="context",
        answer_column_name="answer",
        backbone=TEST_BACKBONE,
        train_file=json_path,
        field="data",
    )
    batch = next(iter(dm.train_dataloader()))
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "start_positions" in batch
    assert "end_positions" in batch


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
def test_wrong_keys_and_types(tmpdir):
    TEST_CSV_DATA.pop("answer_text")
    with pytest.raises(KeyError):
        csv_path = csv_data(tmpdir)
        QuestionAnsweringData.from_csv(
            question_column_name="question",
            context_column_name="context",
            answer_column_name="answer",
            backbone=TEST_BACKBONE,
            train_file=csv_path,
        )

    TEST_CSV_DATA.pop("answer_start")
    with pytest.raises(KeyError):
        csv_path = csv_data(tmpdir)
        QuestionAnsweringData.from_csv(
            question_column_name="question",
            context_column_name="context",
            answer_column_name="answer",
            backbone=TEST_BACKBONE,
            train_file=csv_path,
        )

    TEST_JSON_DATA = [
        {
            "id": "12345",
            "context": "this is an answer one. this is a context one",
            "question": "this is a question one",
        },
        {
            "id": "12346",
            "context": "this is an answer two. this is a context two",
            "question": "this is a question two",
        },
    ]

    with pytest.raises(KeyError):
        json_path = json_data(tmpdir, TEST_JSON_DATA)
        QuestionAnsweringData.from_json(
            question_column_name="question",
            context_column_name="context",
            answer_column_name="answer",
            backbone=TEST_BACKBONE,
            train_file=json_path,
            batch_size=2,
        )

    TEST_JSON_DATA[0]["answer"] = "this is an answer one"
    TEST_JSON_DATA[1]["answer"] = "this is an answer two"
    with pytest.raises(TypeError):
        json_path = json_data(tmpdir, TEST_JSON_DATA)
        QuestionAnsweringData.from_json(
            question_column_name="question",
            context_column_name="context",
            answer_column_name="answer",
            backbone=TEST_BACKBONE,
            train_file=json_path,
            batch_size=2,
        )
