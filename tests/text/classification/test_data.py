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
import os
from pathlib import Path

import pandas as pd
import pytest

from flash.core.utilities.imports import _TEXT_AVAILABLE
from flash.text import TextClassificationData
from flash.text.classification.data import (
    TextCSVDataSource,
    TextDataFrameDataSource,
    TextDataSource,
    TextFileDataSource,
    TextJSONDataSource,
    TextListDataSource,
    TextSentencesDataSource,
)
from tests.helpers.utils import _TEXT_TESTING

if _TEXT_AVAILABLE:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

TEST_BACKBONE = "prajjwal1/bert-tiny"  # super small model for testing

TEST_CSV_DATA = """sentence,label
this is a sentence one,0
this is a sentence two,1
this is a sentence three,0
"""

TEST_JSON_DATA = """
{"sentence": "this is a sentence one","lab":0}
{"sentence": "this is a sentence two","lab":1}
{"sentence": "this is a sentence three","lab":0}
"""

TEST_JSON_DATA_FIELD = """{"data": [
{"sentence": "this is a sentence one","lab":0},
{"sentence": "this is a sentence two","lab":1},
{"sentence": "this is a sentence three","lab":0}]}
"""


TEST_DATA_FRAME_DATA = pd.DataFrame(
    {"sentence": ["this is a sentence one", "this is a sentence two", "this is a sentence three"], "lab": [0, 1, 0]},
)


TEST_LIST_DATA = ["this is a sentence one", "this is a sentence two", "this is a sentence three"]
TEST_LIST_TARGETS = [0, 1, 0]


def csv_data(tmpdir):
    path = Path(tmpdir) / "data.csv"
    path.write_text(TEST_CSV_DATA)
    return path


def json_data(tmpdir):
    path = Path(tmpdir) / "data.json"
    path.write_text(TEST_JSON_DATA)
    return path


def json_data_with_field(tmpdir):
    path = Path(tmpdir) / "data.json"
    path.write_text(TEST_JSON_DATA_FIELD)
    return path


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
def test_from_csv(tmpdir):
    csv_path = csv_data(tmpdir)
    dm = TextClassificationData.from_csv("sentence", "label", backbone=TEST_BACKBONE, train_file=csv_path, batch_size=1)
    batch = next(iter(dm.train_dataloader()))
    assert batch["labels"].item() in [0, 1]
    assert "input_ids" in batch


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
def test_test_valid(tmpdir):
    csv_path = csv_data(tmpdir)
    dm = TextClassificationData.from_csv(
        "sentence",
        "label",
        backbone=TEST_BACKBONE,
        train_file=csv_path,
        val_file=csv_path,
        test_file=csv_path,
        batch_size=1,
    )
    batch = next(iter(dm.val_dataloader()))
    assert batch["labels"].item() in [0, 1]
    assert "input_ids" in batch

    batch = next(iter(dm.test_dataloader()))
    assert batch["labels"].item() in [0, 1]
    assert "input_ids" in batch


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
def test_from_json(tmpdir):
    json_path = json_data(tmpdir)
    dm = TextClassificationData.from_json("sentence", "lab", backbone=TEST_BACKBONE, train_file=json_path, batch_size=1)
    batch = next(iter(dm.train_dataloader()))
    assert batch["labels"].item() in [0, 1]
    assert "input_ids" in batch


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
def test_from_json_with_field(tmpdir):
    json_path = json_data_with_field(tmpdir)
    dm = TextClassificationData.from_json(
        "sentence", "lab", backbone=TEST_BACKBONE, train_file=json_path, batch_size=1, field="data"
    )
    batch = next(iter(dm.train_dataloader()))
    assert batch["labels"].item() in [0, 1]
    assert "input_ids" in batch


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
def test_from_data_frame():
    dm = TextClassificationData.from_data_frame(
        "sentence", "lab", backbone=TEST_BACKBONE, train_data_frame=TEST_DATA_FRAME_DATA, batch_size=1
    )
    batch = next(iter(dm.train_dataloader()))
    assert batch["labels"].item() in [0, 1]
    assert "input_ids" in batch


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
def test_from_list():
    dm = TextClassificationData.from_list(
        backbone=TEST_BACKBONE, train_data=TEST_LIST_DATA, train_targets=TEST_LIST_TARGETS, batch_size=1
    )
    batch = next(iter(dm.train_dataloader()))
    assert batch["labels"].item() in [0, 1]
    assert "input_ids" in batch


@pytest.mark.skipif(_TEXT_AVAILABLE, reason="text libraries are installed.")
def test_text_module_not_found_error():
    with pytest.raises(ModuleNotFoundError, match="[text]"):
        TextClassificationData.from_json("sentence", "lab", backbone=TEST_BACKBONE, train_file="", batch_size=1)


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
@pytest.mark.parametrize(
    "cls, kwargs",
    [
        (TextDataSource, {}),
        (TextFileDataSource, {"filetype": "csv"}),
        (TextCSVDataSource, {}),
        (TextJSONDataSource, {}),
        (TextDataFrameDataSource, {}),
        (TextListDataSource, {}),
        (TextSentencesDataSource, {}),
    ],
)
def test_tokenizer_state(cls, kwargs):
    """Tests that the tokenizer is not in __getstate__"""
    instance = cls(backbone="sshleifer/tiny-mbart", **kwargs)
    state = instance.__getstate__()
    tokenizers = []
    for name, attribute in instance.__dict__.items():
        if isinstance(attribute, PreTrainedTokenizerBase):
            assert name not in state
            setattr(instance, name, None)
            tokenizers.append(name)
    instance.__setstate__(state)
    for name in tokenizers:
        assert getattr(instance, name, None) is not None
