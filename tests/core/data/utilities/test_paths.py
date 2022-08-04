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

from flash.core.data.utilities.loading import AUDIO_EXTENSIONS, IMG_EXTENSIONS, NP_EXTENSIONS
from flash.core.data.utilities.paths import (
    filter_valid_files,
    isdir,
    list_subdirs,
    list_valid_files,
    make_dataset,
    PATH_TYPE,
)

_VALID_EXTENSIONS = AUDIO_EXTENSIONS + IMG_EXTENSIONS + NP_EXTENSIONS


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


def test_filter_valid_files(tmpdir) -> None:
    valid_extensions = list(_VALID_EXTENSIONS)
    fake_extensions = _make_fake_extensions(seed=42)
    mock_extensions = valid_extensions + fake_extensions
    mock_files = _make_fake_files(mock_extensions, seed=42)
    mockdir = _make_mock_dir(tmpdir, mock_files)
    message = "Found invalid file extensions"
    with pytest.warns(UserWarning, match=message):
        filtered = filter_valid_files(mockdir, valid_extensions=valid_extensions)
    assert all(i not in fake_extensions for i in filtered)


def test_filter_valid_files_no_invalid(tmpdir):
    valid_extensions = list(_VALID_EXTENSIONS)
    mock_files = _make_fake_files(valid_extensions, seed=42)
    mockdir = _make_mock_dir(tmpdir, mock_files)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        filtered = filter_valid_files(mockdir, valid_extensions=valid_extensions)
    assert len(filtered) == len(mockdir)


def test_filter_valid_files_with_additional_list(tmpdir) -> None:
    valid_extensions = list(_VALID_EXTENSIONS)
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
    valid_extensions = list(_VALID_EXTENSIONS)
    mock_files = _make_fake_files(valid_extensions, seed=42)
    mockdir = _make_mock_dir(tmpdir, mock_files)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        filtered_files, filtered_additional = filter_valid_files(mockdir, mockdir, valid_extensions=valid_extensions)
    assert len(filtered_files) == len(mockdir)
    assert len(filtered_additional) == len(mockdir)


# per coverage report, write tests for lines: 152, 155, 163
@pytest.mark.skip(reason="not implemented")
def test_filter_valid_files_remaining_tests_placeholder(tmpdir):
    pass


# test for if len(subdirs) < 0 returns tuple(files, targets)
# test if len(subdirs) > 0 returns list[path_type]
# test if extensions is not None returns bool
# test if ValueError is raised both_none or both_something
@pytest.mark.skip(reason="not implemented")
def test_make_dataset(tmpdir):
    _ = make_dataset()
    pass


def test_isdir_true(tmpdir):
    assert isdir(tmpdir)


def test_isdir_false(tmpdir):
    assert not isdir("non_existent_directory")


# test returns a list of subdirs from a rootdir
@pytest.mark.skip(reason="not implemented")
def test_listsubdir(tmpdir):
    _ = list_subdirs()
    pass


@pytest.mark.parametrize("_valid_extensions", [True, False])
def test_list_valid_files_paths_single_file(tmpdir, _valid_extensions):
    valid_extensions = list(_VALID_EXTENSIONS)
    files = _make_fake_files(valid_extensions, seed=42)
    _valid_exts = tuple(valid_extensions) if _valid_extensions else None
    filtered = list_valid_files(files[0], valid_extensions=_valid_exts)
    if _valid_extensions:
        assert filtered[0].endswith(_valid_exts)
    if not _valid_extensions:
        assert files[0] == filtered[0]


@pytest.mark.parametrize("_valid_extensions", [True, False])
def test_list_valid_files_paths_list(tmpdir, _valid_extensions):
    valid_extensions = list(_VALID_EXTENSIONS)
    fake_extensions = _make_fake_extensions(seed=42)
    fake_files = _make_fake_files(fake_extensions, seed=42)
    valid_files = _make_fake_files(valid_extensions, seed=42)
    _valid_exts = tuple(valid_extensions) if _valid_extensions else None
    filtered = list_valid_files(fake_files + valid_files, valid_extensions=_valid_exts)
    if _valid_extensions:
        assert all(i not in fake_files for i in filtered)
    if not _valid_extensions:
        assert fake_files + valid_files == filtered


@pytest.mark.parametrize("_valid_extensions", [True, False])
def test_list_valid_files_paths_dir(tmpdir, _valid_extensions):
    valid_extensions = list(_VALID_EXTENSIONS)
    fake_extensions = _make_fake_extensions(seed=42)
    fake_files = _make_fake_files(fake_extensions, seed=42)
    valid_files = _make_fake_files(valid_extensions, seed=42)
    mockdir = _make_mock_dir(tmpdir, fake_files + valid_files)
    _valid_exts = tuple(valid_extensions) if _valid_extensions else None
    filtered = list_valid_files(mockdir, valid_extensions=_valid_exts)
    if _valid_extensions:
        assert all(i not in fake_files for i in filtered)
    if not _valid_extensions:
        assert mockdir == filtered
