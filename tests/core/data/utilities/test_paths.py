# TODO add tests for filter_valid_files

import os
from pathlib import Path
from string import ascii_lowercase
from typing import List, Union

from numpy import random

from flash.audio.data import AUDIO_EXTENSIONS
from flash.core.data.utilities.paths import filter_valid_files
from flash.image.data import IMG_EXTENSIONS, NP_EXTENSIONS

FILEPATH = Path(__file__)
PARENTDIR = FILEPATH.parent

PATH_TYPE = Union[str, bytes, os.PathLike]


def _make_mock_dir(mock_files: List) -> List[PATH_TYPE]:
    temp_path = os.path.join(PARENTDIR, "tmp")
    mock_dir = []
    for idx, f_ext in enumerate(mock_files):
        mock_dir.append(os.path.join(temp_path, mock_files[idx]))
    return mock_dir


def _make_mock_names(mock_extensions) -> List[str]:
    mock_files = mock_extensions[:]
    for idx, f_ext in enumerate(mock_files):
        idxs = random.randint(0, len(ascii_lowercase), size=3)
        fake_name = "".join([ascii_lowercase[ix] for ix in idxs])
        mock_files[idx] = "".join([fake_name, f_ext])
    return mock_files


def _make_fake_extensions() -> List[str]:
    fake_extensions = []
    for i in range(5):
        idxs = random.randint(0, len(ascii_lowercase), size=3)
        fake_extensions.append("".join(["."] + [ascii_lowercase[idx] for idx in idxs]))
    return fake_extensions


def _make_valid_extensions() -> tuple:
    return AUDIO_EXTENSIONS + IMG_EXTENSIONS + NP_EXTENSIONS


def test_filter_valid_files() -> None:
    random.seed(42)
    valid_extensions = _make_valid_extensions()
    valid_extensions = list(valid_extensions)
    fake_extensions = _make_fake_extensions()
    mock_files = valid_extensions + fake_extensions
    mock_files = _make_mock_names(mock_files)
    mockdir = _make_mock_dir(mock_files)
    filtered = filter_valid_files(files=mockdir, valid_extensions=valid_extensions)
    assert all(i not in fake_extensions for i in filtered)


