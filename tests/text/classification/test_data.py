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

from flash.core.data.data_source import DefaultDataKeys
from flash.core.utilities.imports import _TEXT_AVAILABLE
from flash.text import TextClassificationData
from tests.helpers.utils import _TEXT_TESTING

if _TEXT_AVAILABLE:
    from datasets import Dataset

    from flash.text.tokenizers import TEXT_CLASSIFIER_TOKENIZERS

TEST_BACKBONE = "prajjwal1/bert-tiny"  # super small model for testing

TEST_CSV_DATA = """sentence,label
this is a sentence one,0
this is a sentence two,1
this is a sentence three,0
"""

TEST_CSV_DATA_MULTILABEL = """sentence,lab1,lab2
this is a sentence one,0,1
this is a sentence two,1,0
this is a sentence three,0,1
"""

TEST_JSON_DATA = """
{"sentence": "this is a sentence one","lab":0}
{"sentence": "this is a sentence two","lab":1}
{"sentence": "this is a sentence three","lab":0}
"""

TEST_JSON_DATA_MULTILABEL = """
{"sentence": "this is a sentence one","lab1":0, "lab2": 1}
{"sentence": "this is a sentence two","lab1":1, "lab2": 0}
{"sentence": "this is a sentence three","lab1":0, "lab2": 1}
"""

TEST_JSON_DATA_FIELD = """{"data": [
{"sentence": "this is a sentence one","lab":0},
{"sentence": "this is a sentence two","lab":1},
{"sentence": "this is a sentence three","lab":0}]}
"""

TEST_JSON_DATA_FIELD_MULTILABEL = """{"data": [
{"sentence": "this is a sentence one","lab1":0, "lab2": 1},
{"sentence": "this is a sentence two","lab1":1, "lab2": 0},
{"sentence": "this is a sentence three","lab1":0, "lab2": 1}]}
"""

TEST_DATA_FRAME_DATA = pd.DataFrame(
    {
        "sentence": ["this is a sentence one", "this is a sentence two", "this is a sentence three"],
        "lab1": [0, 1, 0],
        "lab2": [1, 0, 1],
    },
)

TEST_LIST_DATA = ["this is a sentence one", "this is a sentence two", "this is a sentence three"]
TEST_LIST_TARGETS = [0, 1, 0]
TEST_LIST_TARGETS_MULTILABEL = [[0, 1], [1, 0], [0, 1]]


def csv_data(tmpdir, multilabel: bool):
    path = Path(tmpdir) / "data.csv"
    if multilabel:
        path.write_text(TEST_CSV_DATA_MULTILABEL)
    else:
        path.write_text(TEST_CSV_DATA)
    return path


def json_data(tmpdir, multilabel: bool):
    path = Path(tmpdir) / "data.json"
    if multilabel:
        path.write_text(TEST_JSON_DATA_MULTILABEL)
    else:
        path.write_text(TEST_JSON_DATA)
    return path


def json_data_with_field(tmpdir, multilabel: bool):
    path = Path(tmpdir) / "data.json"
    if multilabel:
        path.write_text(TEST_JSON_DATA_FIELD_MULTILABEL)
    else:
        path.write_text(TEST_JSON_DATA_FIELD)
    return path


def parquet_data(tmpdir):
    path = Path(tmpdir) / "data.parquet"
    TEST_DATA_FRAME_DATA.to_parquet(path)
    return path


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
@pytest.mark.parametrize("pretrained", [True, False])
def test_from_csv(tmpdir, pretrained):
    csv_path = csv_data(tmpdir, multilabel=False)
    dm = TextClassificationData.from_csv(
        "sentence",
        "label",
        backbone=TEST_BACKBONE,
        pretrained=pretrained,
        train_file=csv_path,
        val_file=csv_path,
        test_file=csv_path,
        predict_file=csv_path,
        batch_size=1,
    )

    batch = next(iter(dm.train_dataloader()))
    assert batch[DefaultDataKeys.TARGET].item() in [0, 1]
    assert "input_ids" in batch

    batch = next(iter(dm.val_dataloader()))
    assert batch[DefaultDataKeys.TARGET].item() in [0, 1]
    assert "input_ids" in batch

    batch = next(iter(dm.test_dataloader()))
    assert batch[DefaultDataKeys.TARGET].item() in [0, 1]
    assert "input_ids" in batch

    batch = next(iter(dm.predict_dataloader()))
    assert "input_ids" in batch


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
@pytest.mark.parametrize("pretrained", [True, False])
def test_from_csv_multilabel(tmpdir, pretrained):
    csv_path = csv_data(tmpdir, multilabel=True)
    dm = TextClassificationData.from_csv(
        "sentence",
        ["lab1", "lab2"],
        pretrained=pretrained,
        backbone=TEST_BACKBONE,
        train_file=csv_path,
        val_file=csv_path,
        test_file=csv_path,
        predict_file=csv_path,
        batch_size=1,
    )

    batch = next(iter(dm.train_dataloader()))
    assert all([label in [0, 1] for label in batch[DefaultDataKeys.TARGET][0]])
    assert "input_ids" in batch

    batch = next(iter(dm.val_dataloader()))
    assert all([label in [0, 1] for label in batch[DefaultDataKeys.TARGET][0]])
    assert "input_ids" in batch

    batch = next(iter(dm.test_dataloader()))
    assert all([label in [0, 1] for label in batch[DefaultDataKeys.TARGET][0]])
    assert "input_ids" in batch

    batch = next(iter(dm.predict_dataloader()))
    assert "input_ids" in batch


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
@pytest.mark.parametrize("pretrained", [True, False])
def test_from_json(tmpdir, pretrained):
    json_path = json_data(tmpdir, multilabel=False)
    dm = TextClassificationData.from_json(
        "sentence",
        "lab",
        pretrained=pretrained,
        backbone=TEST_BACKBONE,
        train_file=json_path,
        val_file=json_path,
        test_file=json_path,
        predict_file=json_path,
        batch_size=1,
    )

    batch = next(iter(dm.train_dataloader()))
    assert batch[DefaultDataKeys.TARGET].item() in [0, 1]
    assert "input_ids" in batch

    batch = next(iter(dm.val_dataloader()))
    assert batch[DefaultDataKeys.TARGET].item() in [0, 1]
    assert "input_ids" in batch

    batch = next(iter(dm.test_dataloader()))
    assert batch[DefaultDataKeys.TARGET].item() in [0, 1]
    assert "input_ids" in batch

    batch = next(iter(dm.predict_dataloader()))
    assert "input_ids" in batch


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
@pytest.mark.parametrize("pretrained", [True, False])
def test_from_json_multilabel(tmpdir, pretrained):
    json_path = json_data(tmpdir, multilabel=True)
    dm = TextClassificationData.from_json(
        "sentence",
        ["lab1", "lab2"],
        pretrained=pretrained,
        backbone=TEST_BACKBONE,
        train_file=json_path,
        val_file=json_path,
        test_file=json_path,
        predict_file=json_path,
        batch_size=1,
    )

    batch = next(iter(dm.train_dataloader()))
    assert all([label in [0, 1] for label in batch[DefaultDataKeys.TARGET][0]])
    assert "input_ids" in batch

    batch = next(iter(dm.val_dataloader()))
    assert all([label in [0, 1] for label in batch[DefaultDataKeys.TARGET][0]])
    assert "input_ids" in batch

    batch = next(iter(dm.test_dataloader()))
    assert all([label in [0, 1] for label in batch[DefaultDataKeys.TARGET][0]])
    assert "input_ids" in batch

    batch = next(iter(dm.predict_dataloader()))
    assert "input_ids" in batch


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
@pytest.mark.parametrize("pretrained", [True, False])
def test_from_json_with_field(tmpdir, pretrained):
    json_path = json_data_with_field(tmpdir, multilabel=False)
    dm = TextClassificationData.from_json(
        "sentence",
        "lab",
        pretrained=pretrained,
        backbone=TEST_BACKBONE,
        train_file=json_path,
        val_file=json_path,
        test_file=json_path,
        predict_file=json_path,
        batch_size=1,
        field="data",
    )

    batch = next(iter(dm.train_dataloader()))
    assert batch[DefaultDataKeys.TARGET].item() in [0, 1]
    assert "input_ids" in batch

    batch = next(iter(dm.val_dataloader()))
    assert batch[DefaultDataKeys.TARGET].item() in [0, 1]
    assert "input_ids" in batch

    batch = next(iter(dm.test_dataloader()))
    assert batch[DefaultDataKeys.TARGET].item() in [0, 1]
    assert "input_ids" in batch

    batch = next(iter(dm.predict_dataloader()))
    assert "input_ids" in batch


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
@pytest.mark.parametrize("pretrained", [True, False])
def test_from_json_with_field_multilabel(tmpdir, pretrained):
    json_path = json_data_with_field(tmpdir, multilabel=True)
    dm = TextClassificationData.from_json(
        "sentence",
        ["lab1", "lab2"],
        pretrained=pretrained,
        backbone=TEST_BACKBONE,
        train_file=json_path,
        val_file=json_path,
        test_file=json_path,
        predict_file=json_path,
        batch_size=1,
        field="data",
    )

    batch = next(iter(dm.train_dataloader()))
    assert all([label in [0, 1] for label in batch[DefaultDataKeys.TARGET][0]])
    assert "input_ids" in batch

    batch = next(iter(dm.val_dataloader()))
    assert all([label in [0, 1] for label in batch[DefaultDataKeys.TARGET][0]])
    assert "input_ids" in batch

    batch = next(iter(dm.test_dataloader()))
    assert all([label in [0, 1] for label in batch[DefaultDataKeys.TARGET][0]])
    assert "input_ids" in batch

    batch = next(iter(dm.predict_dataloader()))
    assert "input_ids" in batch


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
@pytest.mark.parametrize("pretrained", [True, False])
def test_from_parquet(tmpdir, pretrained):
    parquet_path = parquet_data(tmpdir)
    dm = TextClassificationData.from_parquet(
        "sentence",
        "lab1",
        pretrained=pretrained,
        backbone=TEST_BACKBONE,
        train_file=parquet_path,
        val_file=parquet_path,
        test_file=parquet_path,
        predict_file=parquet_path,
        batch_size=1,
    )

    batch = next(iter(dm.train_dataloader()))
    assert batch[DefaultDataKeys.TARGET].item() in [0, 1]
    assert "input_ids" in batch

    batch = next(iter(dm.val_dataloader()))
    assert batch[DefaultDataKeys.TARGET].item() in [0, 1]
    assert "input_ids" in batch

    batch = next(iter(dm.test_dataloader()))
    assert batch[DefaultDataKeys.TARGET].item() in [0, 1]
    assert "input_ids" in batch

    batch = next(iter(dm.predict_dataloader()))
    assert "input_ids" in batch


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
@pytest.mark.parametrize("pretrained", [True, False])
def test_from_parquet_multilabel(tmpdir, pretrained):
    parquet_path = parquet_data(tmpdir)
    dm = TextClassificationData.from_parquet(
        "sentence",
        ["lab1", "lab2"],
        pretrained=pretrained,
        backbone=TEST_BACKBONE,
        train_file=parquet_path,
        val_file=parquet_path,
        test_file=parquet_path,
        predict_file=parquet_path,
        batch_size=1,
    )

    batch = next(iter(dm.train_dataloader()))
    assert all([label in [0, 1] for label in batch[DefaultDataKeys.TARGET][0]])
    assert "input_ids" in batch

    batch = next(iter(dm.val_dataloader()))
    assert all([label in [0, 1] for label in batch[DefaultDataKeys.TARGET][0]])
    assert "input_ids" in batch

    batch = next(iter(dm.test_dataloader()))
    assert all([label in [0, 1] for label in batch[DefaultDataKeys.TARGET][0]])
    assert "input_ids" in batch

    batch = next(iter(dm.predict_dataloader()))
    assert "input_ids" in batch


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
@pytest.mark.parametrize("pretrained", [True, False])
def test_from_data_frame(pretrained):
    dm = TextClassificationData.from_data_frame(
        "sentence",
        "lab1",
        pretrained=pretrained,
        backbone=TEST_BACKBONE,
        train_data_frame=TEST_DATA_FRAME_DATA,
        val_data_frame=TEST_DATA_FRAME_DATA,
        test_data_frame=TEST_DATA_FRAME_DATA,
        predict_data_frame=TEST_DATA_FRAME_DATA,
        batch_size=1,
    )

    batch = next(iter(dm.train_dataloader()))
    assert batch[DefaultDataKeys.TARGET].item() in [0, 1]
    assert "input_ids" in batch

    batch = next(iter(dm.val_dataloader()))
    assert batch[DefaultDataKeys.TARGET].item() in [0, 1]
    assert "input_ids" in batch

    batch = next(iter(dm.test_dataloader()))
    assert batch[DefaultDataKeys.TARGET].item() in [0, 1]
    assert "input_ids" in batch

    batch = next(iter(dm.predict_dataloader()))
    assert "input_ids" in batch


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
@pytest.mark.parametrize("pretrained", [True, False])
def test_from_data_frame_multilabel(pretrained):
    dm = TextClassificationData.from_data_frame(
        "sentence",
        ["lab1", "lab2"],
        pretrained=pretrained,
        backbone=TEST_BACKBONE,
        train_data_frame=TEST_DATA_FRAME_DATA,
        val_data_frame=TEST_DATA_FRAME_DATA,
        test_data_frame=TEST_DATA_FRAME_DATA,
        predict_data_frame=TEST_DATA_FRAME_DATA,
        batch_size=1,
    )

    batch = next(iter(dm.train_dataloader()))
    assert all([label in [0, 1] for label in batch[DefaultDataKeys.TARGET][0]])
    assert "input_ids" in batch

    batch = next(iter(dm.val_dataloader()))
    assert all([label in [0, 1] for label in batch[DefaultDataKeys.TARGET][0]])
    assert "input_ids" in batch

    batch = next(iter(dm.test_dataloader()))
    assert all([label in [0, 1] for label in batch[DefaultDataKeys.TARGET][0]])
    assert "input_ids" in batch

    batch = next(iter(dm.predict_dataloader()))
    assert "input_ids" in batch


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
@pytest.mark.parametrize("pretrained", [True, False])
def test_from_hf_datasets(pretrained):
    TEST_HF_DATASET_DATA = Dataset.from_pandas(TEST_DATA_FRAME_DATA)
    dm = TextClassificationData.from_hf_datasets(
        "sentence",
        "lab1",
        pretrained=pretrained,
        backbone=TEST_BACKBONE,
        train_hf_dataset=TEST_HF_DATASET_DATA,
        val_hf_dataset=TEST_HF_DATASET_DATA,
        test_hf_dataset=TEST_HF_DATASET_DATA,
        predict_hf_dataset=TEST_HF_DATASET_DATA,
        batch_size=1,
    )

    batch = next(iter(dm.train_dataloader()))
    assert batch[DefaultDataKeys.TARGET].item() in [0, 1]
    assert "input_ids" in batch

    batch = next(iter(dm.val_dataloader()))
    assert batch[DefaultDataKeys.TARGET].item() in [0, 1]
    assert "input_ids" in batch

    batch = next(iter(dm.test_dataloader()))
    assert batch[DefaultDataKeys.TARGET].item() in [0, 1]
    assert "input_ids" in batch

    batch = next(iter(dm.predict_dataloader()))
    assert "input_ids" in batch


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
@pytest.mark.parametrize("pretrained", [True, False])
def test_from_hf_datasets_multilabel(pretrained):
    TEST_HF_DATASET_DATA = Dataset.from_pandas(TEST_DATA_FRAME_DATA)
    dm = TextClassificationData.from_hf_datasets(
        "sentence",
        ["lab1", "lab2"],
        pretrained=pretrained,
        backbone=TEST_BACKBONE,
        train_hf_dataset=TEST_HF_DATASET_DATA,
        val_hf_dataset=TEST_HF_DATASET_DATA,
        test_hf_dataset=TEST_HF_DATASET_DATA,
        predict_hf_dataset=TEST_HF_DATASET_DATA,
        batch_size=1,
    )

    batch = next(iter(dm.train_dataloader()))
    assert all([label in [0, 1] for label in batch[DefaultDataKeys.TARGET][0]])
    assert "input_ids" in batch

    batch = next(iter(dm.val_dataloader()))
    assert all([label in [0, 1] for label in batch[DefaultDataKeys.TARGET][0]])
    assert "input_ids" in batch

    batch = next(iter(dm.test_dataloader()))
    assert all([label in [0, 1] for label in batch[DefaultDataKeys.TARGET][0]])
    assert "input_ids" in batch

    batch = next(iter(dm.predict_dataloader()))
    assert "input_ids" in batch


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
@pytest.mark.parametrize("pretrained", [True, False])
def test_from_lists(pretrained):
    dm = TextClassificationData.from_lists(
        pretrained=pretrained,
        backbone=TEST_BACKBONE,
        train_data=TEST_LIST_DATA,
        train_targets=TEST_LIST_TARGETS,
        val_data=TEST_LIST_DATA,
        val_targets=TEST_LIST_TARGETS,
        test_data=TEST_LIST_DATA,
        test_targets=TEST_LIST_TARGETS,
        predict_data=TEST_LIST_DATA,
        batch_size=1,
    )

    batch = next(iter(dm.train_dataloader()))
    assert batch[DefaultDataKeys.TARGET].item() in [0, 1]
    assert "input_ids" in batch

    batch = next(iter(dm.val_dataloader()))
    assert batch[DefaultDataKeys.TARGET].item() in [0, 1]
    assert "input_ids" in batch

    batch = next(iter(dm.test_dataloader()))
    assert batch[DefaultDataKeys.TARGET].item() in [0, 1]
    assert "input_ids" in batch

    batch = next(iter(dm.predict_dataloader()))
    assert "input_ids" in batch


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
@pytest.mark.parametrize("pretrained", [True, False])
def test_from_lists_multilabel(pretrained):
    dm = TextClassificationData.from_lists(
        pretrained=pretrained,
        backbone=TEST_BACKBONE,
        train_data=TEST_LIST_DATA,
        train_targets=TEST_LIST_TARGETS_MULTILABEL,
        val_data=TEST_LIST_DATA,
        val_targets=TEST_LIST_TARGETS_MULTILABEL,
        test_data=TEST_LIST_DATA,
        test_targets=TEST_LIST_TARGETS_MULTILABEL,
        predict_data=TEST_LIST_DATA,
        batch_size=1,
    )

    batch = next(iter(dm.train_dataloader()))
    assert all([label in [0, 1] for label in batch[DefaultDataKeys.TARGET][0]])
    assert "input_ids" in batch

    batch = next(iter(dm.val_dataloader()))
    assert all([label in [0, 1] for label in batch[DefaultDataKeys.TARGET][0]])
    assert "input_ids" in batch

    batch = next(iter(dm.test_dataloader()))
    assert all([label in [0, 1] for label in batch[DefaultDataKeys.TARGET][0]])
    assert "input_ids" in batch

    batch = next(iter(dm.predict_dataloader()))
    assert "input_ids" in batch


@pytest.mark.skipif(_TEXT_AVAILABLE, reason="text libraries are installed.")
def test_text_module_not_found_error():
    with pytest.raises(ModuleNotFoundError, match="[text]"):
        TextClassificationData.from_json("sentence", "lab", backbone=TEST_BACKBONE, train_file="", batch_size=1)


@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
def test_tokenizer_training():
    tok1, vocab_size1 = TEXT_CLASSIFIER_TOKENIZERS.get(TEST_BACKBONE)(pretrained=True)
    tok2, vocab_size2 = TEXT_CLASSIFIER_TOKENIZERS.get(TEST_BACKBONE)(pretrained=False, vocab_size=10)
    assert tok1.vocab_size != tok2.vocab_size
    assert tok2.vocab_size == 10
    assert tok1.vocab_size == vocab_size1
    assert tok2.vocab_size == vocab_size2
