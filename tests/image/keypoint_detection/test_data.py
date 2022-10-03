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
import pytest

from flash.core.data.io.input import DataKeys
from flash.core.utilities.imports import _IMAGE_EXTRAS_TESTING
from flash.image.keypoint_detection import KeypointDetectionData
from tests.image.detection.test_data import _create_synth_files_dataset, _create_synth_folders_dataset


@pytest.mark.skipif(not _IMAGE_EXTRAS_TESTING, reason="image libraries aren't installed.")
def test_image_detector_data_from_files(tmpdir):
    predict_files = _create_synth_files_dataset(tmpdir)
    datamodule = KeypointDetectionData.from_files(
        predict_files=predict_files, batch_size=1, transform_kwargs=dict(image_size=(128, 128))
    )
    data = next(iter(datamodule.predict_dataloader()))
    sample = data[0]
    assert sample[DataKeys.INPUT].shape == (128, 128, 3)


@pytest.mark.skipif(not _IMAGE_EXTRAS_TESTING, reason="image libraries aren't installed.")
def test_image_detector_data_from_folders(tmpdir):
    predict_folder = _create_synth_folders_dataset(tmpdir)
    datamodule = KeypointDetectionData.from_folders(
        predict_folder=predict_folder, batch_size=1, transform_kwargs=dict(image_size=(128, 128))
    )
    data = next(iter(datamodule.predict_dataloader()))
    sample = data[0]
    assert sample[DataKeys.INPUT].shape == (128, 128, 3)
