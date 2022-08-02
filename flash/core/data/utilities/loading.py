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
import copy
from functools import partial

import fsspec
import numpy as np
import pandas as pd
import torch

from flash.core.data.utilities.paths import has_file_allowed_extension
from flash.core.utilities.imports import _AUDIO_AVAILABLE, _TORCHVISION_AVAILABLE, Image

if _AUDIO_AVAILABLE:
    from torchaudio.transforms import Spectrogram

if _TORCHVISION_AVAILABLE:
    from torchvision.datasets.folder import IMG_EXTENSIONS
else:
    IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

NP_EXTENSIONS = (".npy",)

AUDIO_EXTENSIONS = (
    ".aiff",
    ".au",
    ".avr",
    ".caf",
    ".flac",
    ".mat",
    ".mat4",
    ".mat5",
    ".mpc2k",
    ".ogg",
    ".paf",
    ".pvf",
    ".rf64",
    ".ircam",
    ".voc",
    ".w64",
    ".wav",
    ".nist",
    ".wavex",
)

CSV_EXTENSIONS = (".csv", ".txt")

TSV_EXTENSIONS = (".tsv",)


def _load_image_from_image(file):
    img = Image.open(file)
    img.load()

    img = img.convert("RGB")
    return img


def _load_image_from_numpy(file):
    return Image.fromarray(np.load(file).astype("uint8")).convert("RGB")


def _load_spectrogram_from_image(file):
    img = _load_image_from_image(file)
    return np.array(img).astype("float32")


def _load_spectrogram_from_numpy(file):
    return np.load(file).astype("float32")


def _load_spectrogram_from_audio(file, sampling_rate: int = 16000, n_fft: int = 400):
    # Import locally to prevent import errors if system dependencies are not available.
    import librosa
    from soundfile import SoundFile

    sound_file = SoundFile(file)
    waveform, _ = librosa.load(sound_file, sr=sampling_rate)
    return Spectrogram(n_fft, normalized=True)(torch.from_numpy(waveform).unsqueeze(0)).permute(1, 2, 0).numpy()


def _load_audio_from_audio(file, sampling_rate: int = 16000):
    # Import locally to prevent import errors if system dependencies are not available.
    import librosa

    waveform, _ = librosa.load(file, sr=sampling_rate)
    return waveform


def _load_data_frame_from_csv(file, encoding: str):
    return pd.read_csv(file, encoding=encoding)


def _load_data_frame_from_tsv(file, encoding: str):
    return pd.read_csv(file, sep="\t", encoding=encoding)


_image_loaders = {
    IMG_EXTENSIONS: _load_image_from_image,
    NP_EXTENSIONS: _load_image_from_numpy,
}


_spectrogram_loaders = {
    IMG_EXTENSIONS: _load_spectrogram_from_image,
    NP_EXTENSIONS: _load_spectrogram_from_numpy,
    AUDIO_EXTENSIONS: _load_spectrogram_from_audio,
}


_audio_loaders = {
    AUDIO_EXTENSIONS: _load_audio_from_audio,
}


_data_frame_loaders = {
    CSV_EXTENSIONS: _load_data_frame_from_csv,
    TSV_EXTENSIONS: _load_data_frame_from_tsv,
}


def _get_loader(file_path: str, loaders):
    for extensions, loader in loaders.items():
        if has_file_allowed_extension(file_path, extensions):
            return loader
    raise ValueError(
        f"File: {file_path} has an unsupported extension. Supported extensions: " f"{list(sum(loaders.keys(), ()))}."
    )


def load(file_path: str, loaders):
    loader = _get_loader(file_path, loaders)
    with fsspec.open(file_path) as file:
        return loader(file)


def load_image(file_path: str):
    """Load an image from a file.

    Args:
        file_path: The image file to load.
    """
    return load(file_path, _image_loaders)


def load_spectrogram(file_path: str, sampling_rate: int = 16000, n_fft: int = 400):
    """Load a spectrogram from an image or audio file.

    Args:
        file_path: The file to load.
        sampling_rate: The sampling rate to resample to if loading from an audio file.
        n_fft: The size of the FFT to use when creating a spectrogram from an audio file.
    """
    loaders = copy.copy(_spectrogram_loaders)
    loaders[AUDIO_EXTENSIONS] = partial(loaders[AUDIO_EXTENSIONS], sampling_rate=sampling_rate, n_fft=n_fft)
    return load(file_path, loaders)


def load_audio(file_path: str, sampling_rate: int = 16000):
    """Load a waveform from an audio file.

    Args:
        file_path: The file to load.
        sampling_rate: The sampling rate to resample to.
    """
    loaders = {
        extensions: partial(loader, sampling_rate=sampling_rate) for extensions, loader in _audio_loaders.items()
    }
    return load(file_path, loaders)


def load_data_frame(file_path: str, encoding: str = "utf-8"):
    """Load a data frame from a CSV (or similar) file.

    Args:
        file_path: The file to load.
        encoding: The encoding to use when reading the file.
    """
    loaders = {extensions: partial(loader, encoding=encoding) for extensions, loader in _data_frame_loaders.items()}
    return load(file_path, loaders)
