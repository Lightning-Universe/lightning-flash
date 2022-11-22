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
import contextlib
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import SequentialSampler

import flash
from flash.core.data.io.input import DataKeys
from flash.core.utilities.imports import _FIFTYONE_AVAILABLE, _VIDEO_AVAILABLE, _VIDEO_TESTING
from flash.video import VideoClassificationData, VideoClassifier
from tests.helpers.task_tester import TaskTester
from tests.video.classification.test_data import create_dummy_video_frames, temp_encoded_tensors

if _FIFTYONE_AVAILABLE:
    import fiftyone as fo

if _VIDEO_AVAILABLE:
    import torchvision.io as io
    from pytorchvideo.data.utils import thwc_to_cthw


class TestVideoClassifier(TaskTester):

    task = VideoClassifier
    task_args = (2,)
    task_kwargs = {"pretrained": False, "backbone": "slow_r50"}
    cli_command = "video_classification"
    is_testing = _VIDEO_TESTING
    is_available = _VIDEO_AVAILABLE

    scriptable = False

    @property
    def example_forward_input(self):
        return torch.rand(1, 3, 10, 244, 244)

    def check_forward_output(self, output: Any):
        assert isinstance(output, Tensor)
        assert output.shape == torch.Size([1, 2])

    @property
    def example_train_sample(self):
        return {DataKeys.INPUT: torch.rand(3, 10, 244, 244), DataKeys.TARGET: 1}

    @property
    def example_val_sample(self):
        return self.example_train_sample

    @property
    def example_test_sample(self):
        return self.example_train_sample


# https://github.com/facebookresearch/pytorchvideo/blob/4feccb607d7a16933d485495f91d067f177dd8db/tests/utils.py#L33
@contextlib.contextmanager
def temp_encoded_video(num_frames: int, fps: int, height=10, width=10, prefix=None, directory=None):
    """Creates a temporary lossless, mp4 video with synthetic content.

    Uses a context which deletes the video after exit.
    """
    # Lossless options.
    video_codec = "libx264rgb"
    options = {"crf": "0"}
    data = create_dummy_video_frames(num_frames, height, width)
    with tempfile.NamedTemporaryFile(prefix=prefix, suffix=".mp4", dir=directory) as f:
        f.close()
        io.write_video(f.name, data, fps=fps, video_codec=video_codec, options=options)
        yield f.name, thwc_to_cthw(data).to(torch.float32)
    os.unlink(f.name)


@contextlib.contextmanager
def mock_video_data_frame():
    """Creates a temporary mock encoded video dataset with 4 videos labeled from 0 to 4.

    Returns a labeled video file which points to this mock encoded video dataset, the ordered label and videos tuples
    and the video duration in seconds.
    """
    num_frames = 10
    fps = 5
    with temp_encoded_video(num_frames=num_frames, fps=fps) as (
        video_file_name_1,
        data_1,
    ):
        with temp_encoded_video(num_frames=num_frames, fps=fps) as (
            video_file_name_2,
            data_2,
        ):
            data_frame = DataFrame.from_dict(
                {
                    "file": [video_file_name_1, video_file_name_2, video_file_name_1, video_file_name_2],
                    "target": ["cat", "dog", "cat", "dog"],
                }
            )

            video_duration = num_frames / fps
            yield data_frame, video_duration


@contextlib.contextmanager
def mock_video_csv_file(tmpdir):
    with mock_video_data_frame() as (data_frame, video_duration):
        csv_file = os.path.join(tmpdir, "data.csv")
        data_frame.to_csv(csv_file)
        yield csv_file, video_duration


@contextlib.contextmanager
def mock_encoded_video_dataset_folder(tmpdir):
    """Creates a temporary mock encoded video directory tree with 2 videos labeled 1, 2.

    Returns a directory that to this mock encoded video dataset and the video duration in seconds.
    """
    num_frames = 10
    fps = 5

    tmp_dir = Path(tmpdir)
    os.makedirs(str(tmp_dir / "c1"))
    os.makedirs(str(tmp_dir / "c2"))

    with temp_encoded_video(num_frames=num_frames, fps=fps, directory=str(tmp_dir / "c1")):
        with temp_encoded_video(num_frames=num_frames, fps=fps, directory=str(tmp_dir / "c2")):
            video_duration = num_frames / fps
            yield str(tmp_dir), video_duration


@pytest.mark.skipif(not _VIDEO_TESTING, reason="PyTorchVideo isn't installed.")
def test_video_classifier_finetune_from_folder(tmpdir):
    with mock_encoded_video_dataset_folder(tmpdir) as (mock_folder, total_duration):

        half_duration = total_duration / 2 - 1e-9

        datamodule = VideoClassificationData.from_folders(
            train_folder=mock_folder,
            clip_sampler="uniform",
            clip_duration=half_duration,
            video_sampler=SequentialSampler,
            decode_audio=False,
            batch_size=1,
        )

        for sample in datamodule.train_dataset.data:
            expected_t_shape = 5
            assert sample["video"].shape[1] == expected_t_shape

        model = VideoClassifier(num_classes=datamodule.num_classes, pretrained=False, backbone="slow_r50")
        trainer = flash.Trainer(default_root_dir=tmpdir, fast_dev_run=True, gpus=torch.cuda.device_count())
        trainer.finetune(model, datamodule=datamodule)


@pytest.mark.skipif(not _VIDEO_TESTING, reason="PyTorchVideo isn't installed.")
def test_video_classifier_finetune_from_files(tmpdir):
    with mock_video_data_frame() as (mock_data_frame, total_duration):

        half_duration = total_duration / 2 - 1e-9

        datamodule = VideoClassificationData.from_files(
            train_files=mock_data_frame["file"],
            train_targets=mock_data_frame["target"],
            clip_sampler="uniform",
            clip_duration=half_duration,
            video_sampler=SequentialSampler,
            decode_audio=False,
            batch_size=1,
        )

        for sample in datamodule.train_dataset.data:
            expected_t_shape = 5
            assert sample["video"].shape[1] == expected_t_shape

        model = VideoClassifier(num_classes=datamodule.num_classes, pretrained=False, backbone="slow_r50")
        trainer = flash.Trainer(default_root_dir=tmpdir, fast_dev_run=True, gpus=torch.cuda.device_count())
        trainer.finetune(model, datamodule=datamodule)


@pytest.mark.skipif(not _VIDEO_TESTING, reason="PyTorchVideo isn't installed.")
def test_video_classifier_finetune_from_data_frame(tmpdir):
    with mock_video_data_frame() as (mock_data_frame, total_duration):

        half_duration = total_duration / 2 - 1e-9

        datamodule = VideoClassificationData.from_data_frame(
            "file",
            "target",
            train_data_frame=mock_data_frame,
            clip_sampler="uniform",
            clip_duration=half_duration,
            video_sampler=SequentialSampler,
            decode_audio=False,
            batch_size=1,
        )

        for sample in datamodule.train_dataset.data:
            expected_t_shape = 5
            assert sample["video"].shape[1] == expected_t_shape

        model = VideoClassifier(num_classes=datamodule.num_classes, pretrained=False, backbone="slow_r50")
        trainer = flash.Trainer(default_root_dir=tmpdir, fast_dev_run=True, gpus=torch.cuda.device_count())
        trainer.finetune(model, datamodule=datamodule)


@pytest.mark.skipif(not _VIDEO_TESTING, reason="PyTorchVideo isn't installed.")
def test_video_classifier_finetune_from_tensors(tmpdir):
    mock_tensors = temp_encoded_tensors(num_frames=5)
    datamodule = VideoClassificationData.from_tensors(
        train_data=[mock_tensors, mock_tensors],
        train_targets=["Patient", "Doctor"],
        video_sampler=SequentialSampler,
        batch_size=1,
    )

    for sample in datamodule.train_dataset.data:
        expected_t_shape = 5
        assert sample["video"].shape[1] == expected_t_shape

    assert len(datamodule.labels) == 2, f"Expected number of labels to be 2 but found {len(datamodule.labels)}"

    model = VideoClassifier(
        num_classes=datamodule.num_classes, pretrained=False, backbone="slow_r50", labels=datamodule.labels
    )
    trainer = flash.Trainer(default_root_dir=tmpdir, fast_dev_run=True, gpus=torch.cuda.device_count())
    trainer.finetune(model, datamodule=datamodule)


@pytest.mark.skipif(not _VIDEO_TESTING, reason="PyTorchVideo isn't installed.")
def test_video_classifier_predict_from_tensors(tmpdir):
    mock_tensors = temp_encoded_tensors(num_frames=5)
    datamodule = VideoClassificationData.from_tensors(
        train_data=[mock_tensors, mock_tensors],
        train_targets=["Patient", "Doctor"],
        predict_data=[mock_tensors, mock_tensors],
        video_sampler=SequentialSampler,
        batch_size=1,
    )

    for sample in datamodule.train_dataset.data:
        expected_t_shape = 5
        assert sample["video"].shape[1] == expected_t_shape

    assert len(datamodule.labels) == 2, f"Expected number of labels to be 2 but found {len(datamodule.labels)}"

    model = VideoClassifier(
        num_classes=datamodule.num_classes, pretrained=False, backbone="slow_r50", labels=datamodule.labels
    )
    trainer = flash.Trainer(default_root_dir=tmpdir, fast_dev_run=True, gpus=torch.cuda.device_count())
    trainer.finetune(model, datamodule=datamodule)
    predictions = trainer.predict(model, datamodule=datamodule, output="labels")

    assert predictions is not None
    assert predictions[0][0] in datamodule.labels


@pytest.mark.skipif(not _VIDEO_TESTING, reason="PyTorchVideo isn't installed.")
def test_video_classifier_finetune_from_csv(tmpdir):
    with mock_video_csv_file(tmpdir) as (mock_csv, total_duration):

        half_duration = total_duration / 2 - 1e-9

        datamodule = VideoClassificationData.from_csv(
            "file",
            "target",
            train_file=mock_csv,
            clip_sampler="uniform",
            clip_duration=half_duration,
            video_sampler=SequentialSampler,
            decode_audio=False,
            batch_size=1,
        )

        for sample in datamodule.train_dataset.data:
            expected_t_shape = 5
            assert sample["video"].shape[1] == expected_t_shape

        model = VideoClassifier(num_classes=datamodule.num_classes, pretrained=False, backbone="slow_r50")
        trainer = flash.Trainer(default_root_dir=tmpdir, fast_dev_run=True, gpus=torch.cuda.device_count())
        trainer.finetune(model, datamodule=datamodule)


@pytest.mark.skipif(not _VIDEO_AVAILABLE, reason="PyTorchVideo isn't installed.")
@pytest.mark.skipif(not _FIFTYONE_AVAILABLE, reason="fiftyone isn't installed.")
def test_video_classifier_finetune_fiftyone(tmpdir):
    with mock_encoded_video_dataset_folder(tmpdir) as (
        dir_name,
        total_duration,
    ):

        half_duration = total_duration / 2 - 1e-9

        train_dataset = fo.Dataset.from_dir(
            dir_name,
            dataset_type=fo.types.VideoClassificationDirectoryTree,
        )
        datamodule = VideoClassificationData.from_fiftyone(
            train_dataset=train_dataset,
            clip_sampler="uniform",
            clip_duration=half_duration,
            video_sampler=SequentialSampler,
            decode_audio=False,
            batch_size=1,
        )

        for sample in datamodule.train_dataset.data:
            expected_t_shape = 5
            assert sample["video"].shape[1] == expected_t_shape

        model = VideoClassifier(num_classes=datamodule.num_classes, pretrained=False, backbone="slow_r50")
        trainer = flash.Trainer(fast_dev_run=True, gpus=torch.cuda.device_count())
        trainer.finetune(model, datamodule=datamodule)
