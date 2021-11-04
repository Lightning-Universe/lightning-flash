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

import pytest

import flash
from flash.audio import SpeechRecognitionData
from flash.core.data.io.input import InputDataKeys
from flash.core.utilities.imports import _AUDIO_AVAILABLE
from tests.helpers.utils import _AUDIO_TESTING

path = str(Path(flash.ASSETS_ROOT) / "example.wav")
sample = {"file": path, "text": "example input."}

TEST_CSV_DATA = f"""file,text
{path},example input.
{path},example input.
{path},example input.
{path},example input.
{path},example input.
"""


def csv_data(tmpdir):
    path = Path(tmpdir) / "data.csv"
    path.write_text(TEST_CSV_DATA)
    return path


def json_data(tmpdir, n_samples=5):
    path = Path(tmpdir) / "data.json"
    with path.open("w") as f:
        f.write("\n".join([json.dumps(sample) for x in range(n_samples)]))
    return path


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _AUDIO_TESTING, reason="speech libraries aren't installed.")
def test_from_csv(tmpdir):
    csv_path = csv_data(tmpdir)
    dm = SpeechRecognitionData.from_csv("file", "text", train_file=csv_path, batch_size=1, num_workers=0)
    batch = next(iter(dm.train_dataloader()))
    assert InputDataKeys.INPUT in batch
    assert InputDataKeys.TARGET in batch


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _AUDIO_TESTING, reason="speech libraries aren't installed.")
def test_stage_test_and_valid(tmpdir):
    csv_path = csv_data(tmpdir)
    dm = SpeechRecognitionData.from_csv(
        "file", "text", train_file=csv_path, val_file=csv_path, test_file=csv_path, batch_size=1, num_workers=0
    )
    batch = next(iter(dm.val_dataloader()))
    assert InputDataKeys.INPUT in batch
    assert InputDataKeys.TARGET in batch

    batch = next(iter(dm.test_dataloader()))
    assert InputDataKeys.INPUT in batch
    assert InputDataKeys.TARGET in batch


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _AUDIO_TESTING, reason="speech libraries aren't installed.")
def test_from_json(tmpdir):
    json_path = json_data(tmpdir)
    dm = SpeechRecognitionData.from_json("file", "text", train_file=json_path, batch_size=1, num_workers=0)
    batch = next(iter(dm.train_dataloader()))
    assert InputDataKeys.INPUT in batch
    assert InputDataKeys.TARGET in batch


@pytest.mark.skipif(_AUDIO_AVAILABLE, reason="audio libraries are installed.")
def test_audio_module_not_found_error():
    with pytest.raises(ModuleNotFoundError, match="[audio]"):
        SpeechRecognitionData.from_json("file", "text", train_file="", batch_size=1, num_workers=0)
