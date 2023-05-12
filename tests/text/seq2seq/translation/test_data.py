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

import pytest

from flash import DataKeys
from flash.core.utilities.imports import _TOPIC_TEXT_AVAILABLE
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


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TOPIC_TEXT_AVAILABLE, reason="text libraries aren't installed.")
def test_from_csv(tmpdir):
    csv_path = csv_data(tmpdir)
    dm = TranslationData.from_csv(
        "input",
        "target",
        train_file=csv_path,
        batch_size=1,
    )
    batch = next(iter(dm.train_dataloader()))
    assert isinstance(batch[DataKeys.INPUT][0], str)
    assert isinstance(batch[DataKeys.TARGET][0], str)


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TOPIC_TEXT_AVAILABLE, reason="text libraries aren't installed.")
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
    assert isinstance(batch[DataKeys.INPUT][0], str)
    assert isinstance(batch[DataKeys.TARGET][0], str)

    batch = next(iter(dm.test_dataloader()))
    assert isinstance(batch[DataKeys.INPUT][0], str)
    assert isinstance(batch[DataKeys.TARGET][0], str)


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TOPIC_TEXT_AVAILABLE, reason="text libraries aren't installed.")
def test_from_json(tmpdir):
    json_path = json_data(tmpdir)
    dm = TranslationData.from_json(
        "input",
        "target",
        train_file=json_path,
        batch_size=1,
    )
    batch = next(iter(dm.train_dataloader()))
    assert isinstance(batch[DataKeys.INPUT][0], str)
    assert isinstance(batch[DataKeys.TARGET][0], str)


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _TOPIC_TEXT_AVAILABLE, reason="text libraries aren't installed.")
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
    assert isinstance(batch[DataKeys.INPUT][0], str)
    assert isinstance(batch[DataKeys.TARGET][0], str)
