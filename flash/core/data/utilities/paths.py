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
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

import pandas as pd

from flash.core.utilities.types import PATH_TYPE


# adapted from torchvision:
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py#L10
def has_file_allowed_extension(filename: PATH_TYPE, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return str(filename).lower().endswith(extensions)


# Copied from torchvision:
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py#L48
def make_dataset(
    directory: PATH_TYPE,
    class_to_idx: Dict[str, int],
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).

    Args:
        directory (str): root dataset directory
        class_to_idx (Dict[str, int]): dictionary mapping class name to class index
        extensions (optional): A list of allowed extensions.
            Either extensions or is_valid_file should be passed. Defaults to None.
        is_valid_file (optional): A function that takes path of a file
            and checks if the file is a valid file
            (used to check of corrupt files) both extensions and
            is_valid_file should not be passed. Defaults to None.

    Raises:
        ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.

    Returns:
        List[Tuple[str, int]]: samples of a form (path_to_sample, class)
    """
    instances = []
    directory = os.path.expanduser(str(directory))
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))

    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
    return instances


def isdir(path: Any) -> bool:
    try:
        return os.path.isdir(path)
    except TypeError:
        # data is not path-like (e.g. it may be a list of paths)
        return False


def find_classes(dir: PATH_TYPE) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset. Ensures that no class is a subdirectory of another.

    Args:
        dir: Root directory path.

    Returns:
        (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
    """
    classes = [d.name for d in os.scandir(str(dir)) if d.is_dir()]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def list_valid_files(
    paths: Union[PATH_TYPE, List[PATH_TYPE]], valid_extensions: Optional[Tuple[str, ...]] = None
) -> List[PATH_TYPE]:
    """List the files with a valid extension present in: a single file, a list of files, or a directory.

    Args:
        paths: A single file, a list of files, or a directory.
        valid_extensions: The tuple of valid file extensions.

    Returns:
        The list of files present in ``paths`` that have a valid extension.
    """
    if isdir(paths):
        paths = [os.path.join(paths, file) for file in os.listdir(paths)]

    if not isinstance(paths, list):
        paths = [paths]

    if valid_extensions is None:
        return paths
    return list(
        filter(
            lambda file: has_file_allowed_extension(file, valid_extensions),
            paths,
        )
    )


def read_csv(file: PATH_TYPE) -> pd.DataFrame:
    """A wrapper for ``pd.read_csv`` which tries to handle errors gracefully.

    Args:
        file: The CSV file to read.

    Returns:
        A ``DataFrame`` containing the contents of the file.
    """
    try:
        return pd.read_csv(file, encoding="utf-8")
    except UnicodeDecodeError:
        warnings.warn("A UnicodeDecodeError was raised when reading the CSV. This error will be ignored.")
        return pd.read_csv(file, encoding="utf-8", encoding_errors="ignore")
