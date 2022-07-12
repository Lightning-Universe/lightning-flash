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


def _make_tmp_dir(
    temp_path: PATH_TYPE,
    file_extensions: Union[PATH_TYPE, List[PATH_TYPE]],
) -> None:
    os.mkdir(temp_path)
    for f_ext in file_extensions:
        with open(os.path.join(temp_path, f_ext), "w"):
            pass


def test_filter_valid_files() -> None:
    random.seed(42)
    valid_ext = AUDIO_EXTENSIONS + IMG_EXTENSIONS + NP_EXTENSIONS
    random_ext = []
    for i in range(5):
        idxs = random.randint(0, len(ascii_lowercase), size=3)
        random_ext.append("".join(["."] + [ascii_lowercase[idx] for idx in idxs]))
    valid_ext = list(valid_ext)
    file_exts = valid_ext + random_ext
    for idx, f_ext in enumerate(file_exts):
        idxs = random.randint(0, len(ascii_lowercase), size=3)
        fake_name = "".join([ascii_lowercase[idx] for idx in idxs])
        file_exts[idx] = "".join([fake_name, f_ext])
    tmppath = os.path.join(PARENTDIR, "tmp")
    if not os.path.isdir(tmppath):
        _make_tmp_dir(tmppath, file_exts)
    filtered = filter_valid_files(files=os.listdir(tmppath), valid_extensions=valid_ext)
    assert all(i not in random_ext for i in filtered)


if __name__ == "__main__":
    test_filter_valid_files()
