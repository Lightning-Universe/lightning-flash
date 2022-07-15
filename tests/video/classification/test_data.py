import contextlib
from typing import Union

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


@contextlib.contextmanager
def temp_encoded_tensors(num_frames: int, height=10, width=10):
    data = create_dummy_video_frames(num_frames, height, width)
    yield thwc_to_cthw(data).to(torch.float32)


@contextlib.contextmanager
def mock_video_tensors(num_frames: int = 5):
    with temp_encoded_tensors(num_frames=num_frames) as tens:
        yield tens


def _check_len_and_values(got: list, expected: list):
    assert len(got) == len(expected), f"Expected number of labels: {len(got)}, but got: {len(expected)}"
    assert got == expected


def _check_frames(data, expected_frames_count: Union[list, int], expected_shapes: list):
    if not isinstance(expected_frames_count, list):
        expected_frames_count = [expected_frames_count]

    # to be replaced
    assert 2 == len(expected_frames_count), f"Expected: ?? but got len(expected_frame_count) samples in the dataset."
    for idx, sample_dict in enumerate(data):
        sample = sample_dict["video"]
        assert (
            sample.shape[1] == expected_frames_count[idx]
        ), "Expected video sample {idx} to have {expected_frames_count[idx]} frames but got {sample.shape[1]} frames"
        assert (
            sample.shape == expected_shapes[idx]
        ), f"Expected video shape {expected_shapes[idx]}, but got {sample.shape}"


# Same number of frames per video/sample
def test_load_data_from_tensors_uniform_frames():
    data = []
    labels = []
    expected_shapes = []
    with mock_video_tensors(num_frames=5) as tens:
        data.extend([tens, tens])
        expected_shapes.extend([tens.shape, tens.shape])
        labels.extend(["label1", "label2"])
        datamodule = VideoClassificationData.from_tensors(
            input_field="data", target_field="target", train_data={"data": data, "target": labels}, batch_size=1
        )

    _check_len_and_values(got=datamodule.labels, expected=labels)
    _check_frames(data=datamodule.train_dataset.data, expected_frames_count=[5, 5], expected_shapes=expected_shapes)


# Different number of frames per video/sample
def test_load_data_from_tensors_different_frames():
    num_frames = [5, 3]
    labels = ["label1", "label2"]

    data = []
    expected_shapes = []
    for num_frame in num_frames:
        with mock_video_tensors(num_frame) as tens:
            data.append(tens)
            expected_shapes.append(tens.shape)

    datamodule = VideoClassificationData.from_tensors(
        input_field="data", target_field="target", train_data={"data": data, "target": labels}, batch_size=1
    )

    _check_len_and_values(got=datamodule.labels, expected=labels)
    _check_frames(data=datamodule.train_dataset.data, expected_frames_count=[5, 3], expected_shapes=expected_shapes)
