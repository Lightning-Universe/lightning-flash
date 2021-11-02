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
from flash.text import TranslationData
from tests.helpers.utils import _TEXT_TESTING

if _TEXT_AVAILABLE:
    from datasets import Dataset


TEST_BACKBONE = "sshleifer/tiny-mbart"  # super small model for testing

TEST_CSV_DATA = """input,target
this is a sentence one,this is a translated sentence one
this is a sentence two,this is a translated sentence two
this is a sentence three,this is a translated sentence three
"""

TEST_JSON_DATA = """
{"input": "this is a sentence one","target":"this is a translated sentence one"}
{"input": "this is a sentence two","target":"this is a translated sentence two"}
{"input": "this is a sentence three","target":"this is a translated sentence three"}
"""

TEST_JSON_DATA_FIELD = """{"data": [
{"input": "this is a sentence one","target":"this is a translated sentence one"},
{"input": "this is a sentence two","target":"this is a translated sentence two"},
{"input": "this is a sentence three","target":"this is a translated sentence three"}]}
"""

TEST_DATA_FRAME_DATA = pd.DataFrame(
    {
        "input": ["this is a sentence one", "this is a sentence two", "this is a sentence three"],
        "target": ["this is a translated sentence one", "this is a translated sentence two", "this is a translated sentence three"],
    }
)

TEST_LIST_DATA = ["this is a sentence one", "this is a sentence two", "this is a sentence three"]
TEST_LIST_TARGETS = ["this is a translated sentence one", "this is a translated sentence two", "this is a translated sentence three"]


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


def parquet_data(tmpdir):
    path = Path(tmpdir) / "data.parquet"
    TEST_DATA_FRAME_DATA.to_parquet(path)
    return path


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
@pytest.mark.parametrize("pretrained", [True, False])
def test_from_csv(tmpdir, pretrained):
    csv_path = csv_data(tmpdir)
    dm = TranslationData.from_csv(
        "input",
        "target",
        backbone=TEST_BACKBONE,
        pretrained=pretrained,
        train_file=csv_path,
        val_file=csv_path,
        test_file=csv_path,
        predict_file=csv_path,
        batch_size=1,
    )

    batch = next(iter(dm.train_dataloader()))
    assert "input_ids" in batch
    assert DefaultDataKeys.TARGET in batch

    batch = next(iter(dm.val_dataloader()))
    assert "input_ids" in batch
    assert DefaultDataKeys.TARGET in batch

    batch = next(iter(dm.test_dataloader()))
    assert "input_ids" in batch
    assert DefaultDataKeys.TARGET in batch

    batch = next(iter(dm.predict_dataloader()))
    assert "input_ids" in batch


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
@pytest.mark.parametrize("pretrained", [True, False])
def test_from_json(tmpdir, pretrained):
    json_path = json_data(tmpdir)
    dm = TranslationData.from_json(
        "input",
        "target",
        pretrained=pretrained,
        backbone=TEST_BACKBONE,
        train_file=json_path,
        val_file=json_path,
        test_file=json_path,
        predict_file=json_path,
        batch_size=1,
    )

    batch = next(iter(dm.train_dataloader()))
    assert "input_ids" in batch
    assert DefaultDataKeys.TARGET in batch

    batch = next(iter(dm.val_dataloader()))
    assert "input_ids" in batch
    assert DefaultDataKeys.TARGET in batch

    batch = next(iter(dm.test_dataloader()))
    assert "input_ids" in batch
    assert DefaultDataKeys.TARGET in batch

    batch = next(iter(dm.predict_dataloader()))
    assert "input_ids" in batch



@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
@pytest.mark.parametrize("pretrained", [True, False])
def test_from_json_with_field(tmpdir, pretrained):
    json_path = json_data_with_field(tmpdir)
    dm = TranslationData.from_json(
        "input",
        "target",
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
    assert "input_ids" in batch
    assert DefaultDataKeys.TARGET in batch

    batch = next(iter(dm.val_dataloader()))
    assert "input_ids" in batch
    assert DefaultDataKeys.TARGET in batch

    batch = next(iter(dm.test_dataloader()))
    assert "input_ids" in batch
    assert DefaultDataKeys.TARGET in batch

    batch = next(iter(dm.predict_dataloader()))
    assert "input_ids" in batch


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
@pytest.mark.parametrize("pretrained", [True, False])
def test_from_parquet(tmpdir, pretrained):
    parquet_path = parquet_data(tmpdir)
    dm = TranslationData.from_parquet(
        "input",
        "target",
        pretrained=pretrained,
        backbone=TEST_BACKBONE,
        train_file=parquet_path,
        val_file=parquet_path,
        test_file=parquet_path,
        predict_file=parquet_path,
        batch_size=1,
    )

    batch = next(iter(dm.train_dataloader()))
    assert "input_ids" in batch
    assert DefaultDataKeys.TARGET in batch

    batch = next(iter(dm.val_dataloader()))
    assert "input_ids" in batch
    assert DefaultDataKeys.TARGET in batch

    batch = next(iter(dm.test_dataloader()))
    assert "input_ids" in batch
    assert DefaultDataKeys.TARGET in batch

    batch = next(iter(dm.predict_dataloader()))
    assert "input_ids" in batch


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
@pytest.mark.parametrize("pretrained", [True, False])
def test_from_data_frame(pretrained):
    dm = TranslationData.from_data_frame(
        "input",
        "target",
        pretrained=pretrained,
        backbone=TEST_BACKBONE,
        train_data_frame=TEST_DATA_FRAME_DATA,
        val_data_frame=TEST_DATA_FRAME_DATA,
        test_data_frame=TEST_DATA_FRAME_DATA,
        predict_data_frame=TEST_DATA_FRAME_DATA,
        batch_size=1,
    )

    batch = next(iter(dm.train_dataloader()))
    assert "input_ids" in batch
    assert DefaultDataKeys.TARGET in batch

    batch = next(iter(dm.val_dataloader()))
    assert "input_ids" in batch
    assert DefaultDataKeys.TARGET in batch

    batch = next(iter(dm.test_dataloader()))
    assert "input_ids" in batch
    assert DefaultDataKeys.TARGET in batch

    batch = next(iter(dm.predict_dataloader()))
    assert "input_ids" in batch


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
@pytest.mark.parametrize("pretrained", [True, False])
def test_from_hf_datasets(pretrained):
    TEST_HF_DATASET_DATA = Dataset.from_pandas(TEST_DATA_FRAME_DATA)
    dm = TranslationData.from_hf_datasets(
        "input",
        "target",
        pretrained=pretrained,
        backbone=TEST_BACKBONE,
        train_hf_dataset=TEST_HF_DATASET_DATA,
        val_hf_dataset=TEST_HF_DATASET_DATA,
        test_hf_dataset=TEST_HF_DATASET_DATA,
        predict_hf_dataset=TEST_HF_DATASET_DATA,
        batch_size=1,
    )

    batch = next(iter(dm.train_dataloader()))
    assert "input_ids" in batch
    assert DefaultDataKeys.TARGET in batch

    batch = next(iter(dm.val_dataloader()))
    assert "input_ids" in batch
    assert DefaultDataKeys.TARGET in batch

    batch = next(iter(dm.test_dataloader()))
    assert "input_ids" in batch
    assert DefaultDataKeys.TARGET in batch

    batch = next(iter(dm.predict_dataloader()))
    assert "input_ids" in batch


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
@pytest.mark.parametrize("pretrained", [True, False])
def test_from_lists(pretrained):
    dm = TranslationData.from_lists(
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
    assert "input_ids" in batch
    assert DefaultDataKeys.TARGET in batch

    batch = next(iter(dm.val_dataloader()))
    assert "input_ids" in batch
    assert DefaultDataKeys.TARGET in batch

    batch = next(iter(dm.test_dataloader()))
    assert "input_ids" in batch
    assert DefaultDataKeys.TARGET in batch

    batch = next(iter(dm.predict_dataloader()))
    assert "input_ids" in batch


@pytest.mark.skipif(_TEXT_AVAILABLE, reason="text libraries are installed.")
def test_text_module_not_found_error():
    with pytest.raises(ModuleNotFoundError, match="[text]"):
        TranslationData.from_json("input", "target", backbone=TEST_BACKBONE, train_file="", batch_size=1)
