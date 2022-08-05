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

import pytest
from numpy import random
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from flash.core.data.utilities.loading import AUDIO_EXTENSIONS, IMG_EXTENSIONS, NP_EXTENSIONS
from flash.core.data.utilities.paths import filter_valid_files, isdir, list_subdirs, list_valid_files, make_dataset

VALID_EXTENSIONS = AUDIO_EXTENSIONS + IMG_EXTENSIONS + NP_EXTENSIONS
SEED = 42


def _make_dir_list(rootpath, files):
    directory = []
    for fname in files:
        directory.append(os.path.join(rootpath, fname))
    return directory


def _make_files(extensions):
    random.seed(SEED)
    files = extensions[:]
    for idx, f_ext in enumerate(files):
        idxs = random.randint(0, len(ascii_lowercase), size=3)
        name = "".join([ascii_lowercase[ix] for ix in idxs])
        files[idx] = "".join([name, f_ext])
    return files


def _make_extensions():
    random.seed(SEED)
    extensions = []
    for i in range(5):
        idxs = random.randint(0, len(ascii_lowercase), size=3)
        extensions.append("".join(["."] + [ascii_lowercase[idx] for idx in idxs]))
    return extensions


def test_filter_valid_files(tmpdir):
    valid_extensions = list(VALID_EXTENSIONS)
    fake_extensions = _make_extensions()
    extensions = valid_extensions + fake_extensions
    files = _make_files(extensions)
    mockdir = _make_dir_list(tmpdir, files)
    message = "Found invalid file extensions"
    with pytest.warns(UserWarning, match=message):
        filtered = filter_valid_files(mockdir, valid_extensions=valid_extensions)
    assert all(i not in fake_extensions for i in filtered)


def test_filter_valid_files_path_no_valid_extensions(tmpdir):
    def unpack_additional(*additional):
        return additional

    valid_extensions = list(VALID_EXTENSIONS)
    files = _make_files(valid_extensions)
    additional_lists = _make_dir_list(tmpdir, files)
    tmpdir = pathlib.Path(tmpdir)
    for f in files:
        (tmpdir / f).touch()
    expected = ([tmpdir],) + unpack_additional(additional_lists)
    got = filter_valid_files(tmpdir, additional_lists, valid_extensions=None)
    assert expected == got


def test_filter_valid_files_no_invalid(tmpdir):
    valid_extensions = list(VALID_EXTENSIONS)
    files = _make_files(valid_extensions)
    mockdir = _make_dir_list(tmpdir, files)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        filtered = filter_valid_files(mockdir, valid_extensions=valid_extensions)
    assert len(filtered) == len(mockdir)


def test_filter_valid_files_with_additional_list(tmpdir):
    valid_extensions = list(VALID_EXTENSIONS)
    fake_extensions = _make_extensions()
    extensions = valid_extensions + fake_extensions
    files = _make_files(extensions)
    mockdir = _make_dir_list(tmpdir, files)
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
    files = _make_files(valid_extensions)
    mockdir = _make_dir_list(tmpdir, files)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        filtered_files, filtered_additional = filter_valid_files(mockdir, mockdir, valid_extensions=valid_extensions)
    assert len(filtered_files) == len(mockdir)
    assert len(filtered_additional) == len(mockdir)


def test_filter_valid_files_incorrect_additional_list_size(tmpdir):
    valid_extensions = list(VALID_EXTENSIONS)
    fake_extensions = _make_extensions()
    extensions = valid_extensions + fake_extensions
    files = _make_files(extensions)
    mockdir = _make_dir_list(tmpdir, files)
    message = "The number of files"
    with pytest.raises(MisconfigurationException, match=message):
        filter_valid_files(
            mockdir,
            mockdir[:-1],
            valid_extensions=valid_extensions,
        )


# adapted from torchvision
# https://github.com/pytorch/vision/blob/main/test/test_datasets_utils.py
@pytest.mark.parametrize(
    ("kwargs", "expected_error_msg"),
    [
        (dict(extensions=None, is_valid_file=None), "Both extensions"),
        (dict(extensions=(".png", ".jpeg"), is_valid_file=True), "Both extensions"),
    ],
)
def test_make_dataset_error_messages(tmpdir, kwargs, expected_error_msg):
    tmpdir = pathlib.Path(tmpdir)

    (tmpdir / "a").mkdir()
    (tmpdir / "a" / "a.png").touch()

    (tmpdir / "b").mkdir()
    (tmpdir / "b" / "b.jpeg").touch()

    (tmpdir / "c").mkdir()
    (tmpdir / "c" / "c.unknown").touch()

    with pytest.raises(ValueError, match=expected_error_msg):
        make_dataset(str(tmpdir), **kwargs)


def test_make_dataset_with_extensions_with_subdir(tmpdir):
    tmpdir = pathlib.Path(tmpdir)

    (tmpdir / "a").mkdir()
    (tmpdir / "a" / "a.png").touch()

    (tmpdir / "b").mkdir()
    (tmpdir / "b" / "b.jpeg").touch()

    (tmpdir / "c").mkdir()
    (tmpdir / "c" / "c.unknown").touch()

    # make a file at tmpdir root for line 80, 81 test
    (tmpdir / "tmp.png").touch()

    expected_files = [str(tmpdir / "a" / "a.png"), str(tmpdir / "b" / "b.jpeg")]
    expected_targets = ["a", "b"]
    expected = (expected_files, expected_targets)  # no c because of unk file ext
    got = make_dataset(tmpdir, extensions=VALID_EXTENSIONS)
    assert expected == got


@pytest.mark.parametrize("valid_extensions", [VALID_EXTENSIONS, None])
def test_make_dataset_with_extensions_no_subdir(tmpdir, valid_extensions):
    tmpdir = pathlib.Path(tmpdir)
    (tmpdir / "a.png").touch()
    (tmpdir / "b.jpeg").touch()
    (tmpdir / "c.unknown").touch()
    if valid_extensions:
        expected = ([str(tmpdir / "b.jpeg"), str(tmpdir / "a.png")], None)
        got = make_dataset(tmpdir, extensions=VALID_EXTENSIONS)
    if not valid_extensions:
        expected = ([str(tmpdir / "c.unknown"), str(tmpdir / "b.jpeg"), str(tmpdir / "a.png")], None)
        got = make_dataset(tmpdir, is_valid_file=True)
    assert expected == got


def test_isdir_true(tmpdir):
    assert isdir(tmpdir)


def test_isdir_false(tmpdir):
    assert not isdir("non_existent_directory")


def test_listsubdir(tmpdir):
    tmpdir = pathlib.Path(tmpdir)
    (tmpdir / "a").mkdir()
    (tmpdir / "b").mkdir()
    (tmpdir / "c").mkdir()
    expected = ["a", "b", "c"]
    got = list_subdirs(tmpdir)
    assert expected == got


@pytest.mark.parametrize("valid_extensions", [VALID_EXTENSIONS, None])
def test_list_valid_files_paths_single_file(tmpdir, valid_extensions):
    files = _make_files(list(VALID_EXTENSIONS))
    filtered = list_valid_files(files[0], valid_extensions=valid_extensions)
    if valid_extensions:
        assert filtered[0].endswith(VALID_EXTENSIONS)
    if not valid_extensions:
        assert files[0] == filtered[0]


@pytest.mark.parametrize("valid_extensions", [VALID_EXTENSIONS, None])
def test_list_valid_files_paths_list(tmpdir, valid_extensions):
    fake_extensions = _make_extensions()
    fake_files = _make_files(fake_extensions)
    valid_files = _make_files(list(VALID_EXTENSIONS))
    filtered = list_valid_files(fake_files + valid_files, valid_extensions=valid_extensions)
    if valid_extensions:
        assert all(i not in fake_files for i in filtered)
    if not valid_extensions:
        assert fake_files + valid_files == filtered


@pytest.mark.parametrize("valid_extensions", [VALID_EXTENSIONS, None])
def test_list_valid_files_paths_dir(tmpdir, valid_extensions):
    fake_extensions = _make_extensions()
    fake_files = _make_files(fake_extensions)
    valid_files = _make_files(list(VALID_EXTENSIONS))
    files = fake_files + valid_files
    tmpdir = pathlib.Path(tmpdir)
    for f in files:
        (tmpdir / f).touch()
    filtered = list_valid_files(tmpdir, valid_extensions=valid_extensions)
    if valid_extensions:
        assert all(i not in fake_files for i in filtered)
    if not valid_extensions:
        assert [os.path.join(tmpdir, file) for file in os.listdir(tmpdir)] == filtered
