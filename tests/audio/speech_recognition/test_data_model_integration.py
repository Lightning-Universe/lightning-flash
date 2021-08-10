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
from pytorch_lightning import Trainer

import flash
from flash.audio import SpeechRecognition, SpeechRecognitionData
from tests.helpers.utils import _AUDIO_TESTING

TEST_BACKBONE = "patrickvonplaten/wav2vec2_tiny_random_robust"  # super small model for testing

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
@pytest.mark.skipif(not _AUDIO_TESTING, reason="audio libraries aren't installed.")
def test_classification_csv(tmpdir):
    csv_path = csv_data(tmpdir)

    data = SpeechRecognitionData.from_csv(
        "file",
        "text",
        train_file=csv_path,
        num_workers=0,
        batch_size=2,
    )
    model = SpeechRecognition(backbone=TEST_BACKBONE)
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.fit(model, datamodule=data)


@pytest.mark.skipif(os.name == "nt", reason="Huggingface timing out on Windows")
@pytest.mark.skipif(not _AUDIO_TESTING, reason="audio libraries aren't installed.")
def test_classification_json(tmpdir):
    json_path = json_data(tmpdir)

    data = SpeechRecognitionData.from_json(
        "file",
        "text",
        train_file=json_path,
        num_workers=0,
        batch_size=2,
    )
    model = SpeechRecognition(backbone=TEST_BACKBONE)
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.fit(model, datamodule=data)
