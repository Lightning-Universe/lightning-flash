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
import os
import warnings
from string import ascii_lowercase
from typing import List

import pytest
from numpy import random

from flash.audio.data import AUDIO_EXTENSIONS
from flash.core.data.utilities.paths import filter_valid_files, PATH_TYPE
from flash.image.data import IMG_EXTENSIONS, NP_EXTENSIONS


def _make_mock_dir(root, mock_files: List) -> List[PATH_TYPE]:
    mockdir = []
    for idx, f_ext in enumerate(mock_files):
        mockdir.append(os.path.join(root, mock_files[idx]))
    return mockdir


def _make_fake_files(mock_extensions, seed: int) -> List[str]:
    random.seed(seed)
    fake_files = mock_extensions[:]
    for idx, f_ext in enumerate(fake_files):
        idxs = random.randint(0, len(ascii_lowercase), size=3)
        fake_name = "".join([ascii_lowercase[ix] for ix in idxs])
        fake_files[idx] = "".join([fake_name, f_ext])
    return fake_files


def _make_fake_extensions(seed: int) -> List[str]:
    random.seed(seed)
    fake_extensions = []
    for i in range(5):
        idxs = random.randint(0, len(ascii_lowercase), size=3)
        fake_extensions.append("".join(["."] + [ascii_lowercase[idx] for idx in idxs]))
    return fake_extensions


def _make_valid_extensions() -> tuple:
    return AUDIO_EXTENSIONS + IMG_EXTENSIONS + NP_EXTENSIONS


def test_filter_valid_files(tmpdir) -> None:
    valid_extensions = _make_valid_extensions()
    valid_extensions = list(valid_extensions)
    fake_extensions = _make_fake_extensions(seed=42)
    mock_extensions = valid_extensions + fake_extensions
    mock_files = _make_fake_files(mock_extensions, seed=42)
    mockdir = _make_mock_dir(tmpdir, mock_files)
    message = "Found invalid file extensions"
    with pytest.warns(UserWarning, match=message):
        filtered = filter_valid_files(mockdir, valid_extensions=valid_extensions)
    assert all(i not in fake_extensions for i in filtered)


def test_filter_valid_files_no_invalid(tmpdir):
    valid_extensions = _make_valid_extensions()
    valid_extensions = list(valid_extensions)
    mock_files = _make_fake_files(valid_extensions, seed=42)
    mockdir = _make_mock_dir(tmpdir, mock_files)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        filtered = filter_valid_files(mockdir, valid_extensions=valid_extensions)
    assert len(filtered) == len(mockdir)


def test_filter_valid_files_with_additional_list(tmpdir) -> None:
    valid_extensions = _make_valid_extensions()
    valid_extensions = list(valid_extensions)
    fake_extensions = _make_fake_extensions(seed=42)
    mock_extensions = valid_extensions + fake_extensions
    mock_files = _make_fake_files(mock_extensions, seed=42)
    mockdir = _make_mock_dir(tmpdir, mock_files)
    message = "Found invalid file extensions"
    with pytest.warns(UserWarning, match=message):
        filtered = filter_valid_files(
            mockdir,
            mockdir,
            valid_extensions=valid_extensions,
        )
    assert all(i not in fake_extensions for i in filtered)


def test_filter_valid_files_no_invalid_with_additional_list(tmpdir):
    valid_extensions = _make_valid_extensions()
    valid_extensions = list(valid_extensions)
    mock_files = _make_fake_files(valid_extensions, seed=42)
    mockdir = _make_mock_dir(tmpdir, mock_files)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        filtered_files, filtered_additional = filter_valid_files(mockdir, mockdir, valid_extensions=valid_extensions)
    assert len(filtered_files) == len(mockdir)
    assert len(filtered_additional) == len(mockdir)
