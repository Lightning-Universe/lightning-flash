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
import re
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import torch
from pandas import DataFrame
from torch.utils.data import SequentialSampler

import flash
from flash.__main__ import main
from flash.core.utilities.imports import _FIFTYONE_AVAILABLE, _VIDEO_AVAILABLE, _VIDEO_TESTING
from flash.video import VideoClassificationData, VideoClassifier

if _FIFTYONE_AVAILABLE:
    import fiftyone as fo

if _VIDEO_AVAILABLE:
    import torchvision.io as io
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
    """
    Creates a temporary mock encoded video dataset with 4 videos labeled from 0 to 4.
    Returns a labeled video file which points to this mock encoded video dataset, the
    ordered label and videos tuples and the video duration in seconds.
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


@pytest.mark.skipif(not _VIDEO_TESTING, reason="PyTorchVideo isn't installed.")
def test_jit(tmpdir):
    sample_input = torch.rand(1, 3, 32, 256, 256)
    path = os.path.join(tmpdir, "test.pt")

    model = VideoClassifier(2, pretrained=False, backbone="slow_r50")
    model.eval()

    # pytorchvideo only works with `torch.jit.trace`
    model = torch.jit.trace(model, sample_input)

    torch.jit.save(model, path)
    model = torch.jit.load(path)

    out = model(sample_input)
    assert isinstance(out, torch.Tensor)
    assert out.shape == torch.Size([1, 2])


@pytest.mark.skipif(_VIDEO_AVAILABLE, reason="video libraries are installed.")
def test_load_from_checkpoint_dependency_error():
    with pytest.raises(ModuleNotFoundError, match=re.escape("'lightning-flash[video]'")):
        VideoClassifier.load_from_checkpoint("not_a_real_checkpoint.pt")


@pytest.mark.skipif(not _VIDEO_TESTING, reason="PyTorchVideo isn't installed.")
def test_cli():
    cli_args = ["flash", "video_classification", "--trainer.fast_dev_run", "True", "num_workers", "0"]
    with mock.patch("sys.argv", cli_args):
        try:
            main()
        except SystemExit:
            pass
