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
from typing import Union

import pytest
import torch

from flash.core.utilities.imports import _VIDEO_AVAILABLE
from flash.video.classification.data import VideoClassificationData

if _VIDEO_AVAILABLE:
    from pytorchvideo.data.utils import thwc_to_cthw


def create_dummy_video_frames(num_frames: int, height: int, width: int):
    y, x = torch.meshgrid(torch.linspace(-2, 2, height), torch.linspace(-2, 2, width))
    data = []
    for i in range(num_frames):
        xc = float(i) / num_frames
        yc = 1 - float(i) / (2 * num_frames)
        d = torch.exp(-((x - xc) ** 2 + (y - yc) ** 2) / 2) * 255
        data.append(d.unsqueeze(2).repeat(1, 1, 3).byte())
    return torch.stack(data, 0)


def temp_encoded_tensors(num_frames: int, height=10, width=10):
    if not _VIDEO_AVAILABLE:
        return torch.randint(size=(3, num_frames, height, width), low=0, high=255)
    data = create_dummy_video_frames(num_frames, height, width)
    return thwc_to_cthw(data).to(torch.float32)


def _check_len_and_values(got: list, expected: list):
    assert len(got) == len(expected), f"Expected number of labels: {len(expected)}, but got: {len(got)}"
    assert got == expected


def _check_frames(data, expected_frames_count: Union[list, int]):
    if not isinstance(expected_frames_count, list):
        expected_frames_count = [expected_frames_count]

    # to be replaced
    assert data.size() == len(
        expected_frames_count
    ), f"Expected: {len(expected_frames_count)} but got {data.size()} samples in the dataset."
    for idx, sample_dict in enumerate(data):
        sample = sample_dict["video"]
        assert (
            sample.shape[1] == expected_frames_count[idx]
        ), f"Expected video sample {idx} to have {expected_frames_count[idx]} frames but got {sample.shape[1]} frames"


@pytest.mark.skipif(not _VIDEO_AVAILABLE, reason="PyTorchVideo isn't installed.")
@pytest.mark.parametrize(
    "input_data, input_targets, expected_frames_count",
    [
        ([temp_encoded_tensors(5), temp_encoded_tensors(5)], ["label1", "label2"], [5, 5]),
        ([temp_encoded_tensors(5), temp_encoded_tensors(10)], ["label1", "label2"], [5, 10]),
        (torch.randint(size=(3, 4, 10, 10), low=0, high=255), ["label1"], [4]),
        (torch.stack((temp_encoded_tensors(5), temp_encoded_tensors(5))), ["label1", "label2"], [5, 5]),
        (torch.stack((temp_encoded_tensors(5),)), ["label1"], [5]),
        (temp_encoded_tensors(5), ["label1"], [5]),
    ],
)
def test_load_data_from_tensors(input_data, input_targets, expected_frames_count):
    datamodule = VideoClassificationData.from_tensors(train_data=input_data, train_targets=input_targets, batch_size=1)
    _check_len_and_values(got=datamodule.labels, expected=input_targets)
    _check_frames(data=datamodule.train_dataset.data, expected_frames_count=expected_frames_count)


@pytest.mark.skipif(not _VIDEO_AVAILABLE, reason="PyTorchVideo isn't installed.")
@pytest.mark.parametrize(
    "input_data, input_targets, error_type, match",
    [
        (torch.tensor(1), ["label1"], ValueError, "dimension should be"),
        (torch.randint(size=(2, 3), low=0, high=255), ["label"], ValueError, "dimension should be"),
        (torch.randint(size=(2, 3), low=0, high=255), [], ValueError, "dimension should be"),
        (5, [], TypeError, "Expected either a list/tuple"),
    ],
)
def test_load_incorrect_data_from_tensors(input_data, input_targets, error_type, match):
    with pytest.raises(error_type, match=match):
        VideoClassificationData.from_tensors(train_data=input_data, train_targets=input_targets, batch_size=1)
