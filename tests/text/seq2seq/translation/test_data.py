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
from typing import Any, Dict

import pytest
from torch import Tensor

from flash import DataKeys
from flash.core.utilities.imports import _TEXT_TESTING
from flash.core.utilities.stages import RunningStage
from flash.text import TranslationData

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


def assert_batch_values_ok(batch: Dict[str, Any], stage: RunningStage):
    assert all(key in batch.keys() for key in ["input_ids", "attention_mask"])
    assert isinstance(batch["input_ids"], Tensor)
    assert isinstance(batch["attention_mask"], Tensor)

    if stage in [RunningStage.TRAINING, RunningStage.VALIDATING, RunningStage.TESTING]:
        assert isinstance(batch[DataKeys.TARGET], Tensor)


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
def test_from_csv(tmpdir):
    csv_path = csv_data(tmpdir)
    dm = TranslationData.from_csv(
        "input",
        "target",
        train_file=csv_path,
        batch_size=1,
    )
    batch = next(iter(dm.train_dataloader()))
    assert_batch_values_ok(batch, RunningStage.TRAINING)


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
def test_from_files(tmpdir):
    csv_path = csv_data(tmpdir)
    dm = TranslationData.from_csv(
        "input",
        "target",
        train_file=csv_path,
        val_file=csv_path,
        test_file=csv_path,
        batch_size=1,
    )
    batch = next(iter(dm.val_dataloader()))
    assert_batch_values_ok(batch, RunningStage.VALIDATING)

    batch = next(iter(dm.test_dataloader()))
    assert_batch_values_ok(batch, RunningStage.TESTING)


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
def test_from_json(tmpdir):
    json_path = json_data(tmpdir)
    dm = TranslationData.from_json(
        "input",
        "target",
        train_file=json_path,
        batch_size=1,
    )
    batch = next(iter(dm.train_dataloader()))
    assert_batch_values_ok(batch, RunningStage.TRAINING)


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TEXT_TESTING, reason="text libraries aren't installed.")
def test_from_json_with_field(tmpdir):
    json_path = json_data_with_field(tmpdir)
    dm = TranslationData.from_json(
        "input",
        "target",
        train_file=json_path,
        batch_size=1,
        field="data",
    )
    batch = next(iter(dm.train_dataloader()))
    assert_batch_values_ok(batch, RunningStage.TRAINING)
