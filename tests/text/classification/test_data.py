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

from flash.text import TextClassificationData

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


def csv_data(tmpdir):
    path = Path(tmpdir) / "data.csv"
    path.write_text(TEST_CSV_DATA)
    return path


def json_data(tmpdir):
    path = Path(tmpdir) / "data.json"
    path.write_text(TEST_JSON_DATA)
    return path


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
def test_from_csv(tmpdir):
    csv_path = csv_data(tmpdir)
    dm = TextClassificationData.from_files(
        backbone=TEST_BACKBONE, train_file=csv_path, input="sentence", target="label", batch_size=1
    )
    batch = next(iter(dm.train_dataloader()))
    assert batch["labels"].item() in [0, 1]
    assert "input_ids" in batch


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
def test_test_valid(tmpdir):
    csv_path = csv_data(tmpdir)
    dm = TextClassificationData.from_files(
        backbone=TEST_BACKBONE,
        train_file=csv_path,
        valid_file=csv_path,
        test_file=csv_path,
        input="sentence",
        target="label",
        batch_size=1
    )
    batch = next(iter(dm.val_dataloader()))
    assert batch["labels"].item() in [0, 1]
    assert "input_ids" in batch

    batch = next(iter(dm.test_dataloader()))
    assert batch["labels"].item() in [0, 1]
    assert "input_ids" in batch


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
def test_from_json(tmpdir):
    json_path = json_data(tmpdir)
    dm = TextClassificationData.from_files(
        backbone=TEST_BACKBONE, train_file=json_path, input="sentence", target="lab", filetype="json", batch_size=1
    )
    batch = next(iter(dm.train_dataloader()))
    assert batch["labels"].item() in [0, 1]
    assert "input_ids" in batch
