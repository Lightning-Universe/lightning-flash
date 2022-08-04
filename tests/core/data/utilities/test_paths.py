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
import pathlib
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

VALID_EXTENSIONS = AUDIO_EXTENSIONS + IMG_EXTENSIONS + NP_EXTENSIONS
SEED = 42


def _make_mock_dir(root, mock_files: List) -> List[PATH_TYPE]:
    mockdir = []
    for idx, f_ext in enumerate(mock_files):
        mockdir.append(os.path.join(root, mock_files[idx]))
    return mockdir


def _make_fake_files(mock_extensions) -> List[str]:
    random.seed(SEED)
    fake_files = mock_extensions[:]
    for idx, f_ext in enumerate(fake_files):
        idxs = random.randint(0, len(ascii_lowercase), size=3)
        fake_name = "".join([ascii_lowercase[ix] for ix in idxs])
        fake_files[idx] = "".join([fake_name, f_ext])
    return fake_files


def _make_fake_extensions() -> List[str]:
    random.seed(SEED)
    fake_extensions = []
    for i in range(5):
        idxs = random.randint(0, len(ascii_lowercase), size=3)
        fake_extensions.append("".join(["."] + [ascii_lowercase[idx] for idx in idxs]))
    return fake_extensions


def test_filter_valid_files(tmpdir) -> None:
    valid_extensions = list(VALID_EXTENSIONS)
    fake_extensions = _make_fake_extensions()
    mock_extensions = valid_extensions + fake_extensions
    mock_files = _make_fake_files(mock_extensions)
    mockdir = _make_mock_dir(tmpdir, mock_files)
    message = "Found invalid file extensions"
    with pytest.warns(UserWarning, match=message):
        filtered = filter_valid_files(mockdir, valid_extensions=valid_extensions)
    assert all(i not in fake_extensions for i in filtered)


def test_filter_valid_files_no_invalid(tmpdir):
    valid_extensions = list(VALID_EXTENSIONS)
    mock_files = _make_fake_files(valid_extensions)
    mockdir = _make_mock_dir(tmpdir, mock_files)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        filtered = filter_valid_files(mockdir, valid_extensions=valid_extensions)
    assert len(filtered) == len(mockdir)


def test_filter_valid_files_with_additional_list(tmpdir) -> None:
    valid_extensions = list(VALID_EXTENSIONS)
    fake_extensions = _make_fake_extensions()
    mock_extensions = valid_extensions + fake_extensions
    mock_files = _make_fake_files(mock_extensions)
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
    valid_extensions = list(VALID_EXTENSIONS)
    mock_files = _make_fake_files(valid_extensions)
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


# adapted from torchvision
# https://github.com/pytorch/vision/blob/main/test/test_datasets_utils.py
@pytest.mark.parametrize(
    ("kwargs", "expected_error_msg"),
    [
        (dict(extensions=None, is_valid_file=None), "Both extensions"),
        (dict(extensions=(".png", ".jpeg"), is_valid_file=True), "Both extensions"),
    ],
)
def test_make_dataset_no_valid_files(tmpdir, kwargs, expected_error_msg):
    tmpdir = pathlib.Path(tmpdir)

    (tmpdir / "a").mkdir()
    (tmpdir / "a" / "a.png").touch()

    (tmpdir / "b").mkdir()
    (tmpdir / "b" / "b.jpeg").touch()

    (tmpdir / "c").mkdir()
    (tmpdir / "c" / "c.unknown").touch()

    with pytest.raises(ValueError, match=expected_error_msg):
        make_dataset(str(tmpdir), **kwargs)


# test for if len(subdirs) < 0 returns tuple(files, targets)
# test if len(subdirs) > 0 returns list[path_type]
# test if extensions is not None returns bool
@pytest.mark.skip(reason="not implemented")
def test_make_dataset_valid_files(tmpdir):
    pass


def test_isdir_true(tmpdir):
    assert isdir(tmpdir)


def test_isdir_false(tmpdir):
    assert not isdir("non_existent_directory")


@pytest.mark.skip(reason="not implemented")
def test_listsubdir(tmpdir):
    _ = list_subdirs()
    pass


@pytest.mark.parametrize("valid_extensions", [VALID_EXTENSIONS, None])
def test_list_valid_files_paths_single_file(tmpdir, valid_extensions):
    files = _make_fake_files(list(VALID_EXTENSIONS))
    filtered = list_valid_files(files[0], valid_extensions=valid_extensions)
    if valid_extensions:
        assert filtered[0].endswith(VALID_EXTENSIONS)
    if not valid_extensions:
        assert files[0] == filtered[0]


@pytest.mark.parametrize("valid_extensions", [VALID_EXTENSIONS, None])
def test_list_valid_files_paths_list(tmpdir, valid_extensions):
    fake_extensions = _make_fake_extensions()
    fake_files = _make_fake_files(fake_extensions)
    valid_files = _make_fake_files(list(VALID_EXTENSIONS))
    filtered = list_valid_files(fake_files + valid_files, valid_extensions=valid_extensions)
    if valid_extensions:
        assert all(i not in fake_files for i in filtered)
    if not valid_extensions:
        assert fake_files + valid_files == filtered


@pytest.mark.parametrize("valid_extensions", [VALID_EXTENSIONS, None])
def test_list_valid_files_paths_dir(tmpdir, valid_extensions):
    fake_extensions = _make_fake_extensions()
    fake_files = _make_fake_files(fake_extensions)
    valid_files = _make_fake_files(list(VALID_EXTENSIONS))
    mockdir = _make_mock_dir(tmpdir, fake_files + valid_files)
    filtered = list_valid_files(mockdir, valid_extensions=valid_extensions)
    if valid_extensions:
        assert all(i not in fake_files for i in filtered)
    if not valid_extensions:
        assert mockdir == filtered
