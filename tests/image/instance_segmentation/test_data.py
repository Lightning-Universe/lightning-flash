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
import numpy as np
import pytest
import torch

from flash.core.data.io.input import DataKeys
from flash.core.utilities.imports import _IMAGE_EXTRAS_TESTING
from flash.image.instance_segmentation import InstanceSegmentationData
from flash.image.instance_segmentation.data import InstanceSegmentationOutputTransform
from tests.image.detection.test_data import _create_synth_files_dataset, _create_synth_folders_dataset


@pytest.mark.skipif(not _IMAGE_EXTRAS_TESTING, reason="image libraries aren't installed.")
def test_image_detector_data_from_files(tmpdir):
    predict_files = _create_synth_files_dataset(tmpdir)
    datamodule = InstanceSegmentationData.from_files(
        predict_files=predict_files, batch_size=2, transform_kwargs=dict(image_size=(128, 128))
    )
    data = next(iter(datamodule.predict_dataloader()))
    sample = data[0]
    assert sample[DataKeys.INPUT].shape == (128, 128, 3)


@pytest.mark.skipif(not _IMAGE_EXTRAS_TESTING, reason="image libraries aren't installed.")
def test_image_detector_data_from_folders(tmpdir):
    predict_folder = _create_synth_folders_dataset(tmpdir)
    datamodule = InstanceSegmentationData.from_folders(
        predict_folder=predict_folder, batch_size=2, transform_kwargs=dict(image_size=(128, 128))
    )
    data = next(iter(datamodule.predict_dataloader()))
    sample = data[0]
    assert sample[DataKeys.INPUT].shape == (128, 128, 3)


@pytest.mark.skipif(not _IMAGE_EXTRAS_TESTING, reason="image libraries aren't installed.")
def test_instance_segmentation_output_transform():
    sample = {
        DataKeys.INPUT: torch.rand(3, 224, 224),
        DataKeys.PREDS: {
            "bboxes": [
                {"xmin": 10, "ymin": 10, "width": 15, "height": 15},
                {"xmin": 30, "ymin": 30, "width": 40, "height": 40},
            ],
            "labels": [0, 1],
            "masks": [
                np.random.randint(2, size=(1, 128, 128), dtype=np.uint8),
                np.random.randint(2, size=(1, 128, 128), dtype=np.uint8),
            ],
            "scores": [0.5, 0.5],
        },
        DataKeys.METADATA: {"size": (224, 224)},
    }

    output_transform_cls = InstanceSegmentationOutputTransform()
    data = output_transform_cls.per_sample_transform(sample)

    assert data["masks"][0].size() == (1, 224, 224)
    assert data["masks"][1].size() == (1, 224, 224)
