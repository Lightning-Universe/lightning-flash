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
from typing import Any, Callable, cast, List, Optional, Tuple, Union

from pytorch_lightning.utilities import rank_zero_warn

from flash.core.data.utilities.sort import sorted_alphanumeric

PATH_TYPE = Union[str, bytes, os.PathLike]


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


# Adapted from torchvision:
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py#L48
def make_dataset(
    directory: PATH_TYPE,
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> Tuple[List[PATH_TYPE], Optional[List[PATH_TYPE]]]:
    """Generates a list of samples of a form (path_to_sample, class).

    Args:
        directory (str): root dataset directory
        extensions (optional): A list of allowed extensions.
            Either extensions or is_valid_file should be passed. Defaults to None.
        is_valid_file (optional): A function that takes path of a file
            and checks if the file is a valid file
            (used to check of corrupt files) both extensions and
            is_valid_file should not be passed. Defaults to None.

    Raises:
        ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.

    Returns:
        (files, targets) Tuple containing the list of files and corresponding list of targets.
    """
    files, targets = [], []
    directory = os.path.expanduser(str(directory))
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))

    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    subdirs = list_subdirs(directory)
    if len(subdirs) > 0:
        for target_class in subdirs:
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        files.append(path)
                        targets.append(target_class)
        return files, targets
    return list_valid_files(directory), None


def isdir(path: Any) -> bool:
    try:
        return os.path.isdir(path)
    except TypeError:
        # data is not path-like (e.g. it may be a list of paths)
        return False


def list_subdirs(folder: PATH_TYPE) -> List[str]:
    """List the subdirectories of a given directory.

    Args:
        folder: The directory to scan.

    Returns:
        The list of subdirectories.
    """
    return list(sorted_alphanumeric(d.name for d in os.scandir(str(folder)) if d.is_dir()))


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
    return [path for path in paths if has_file_allowed_extension(path, valid_extensions)]


def filter_valid_files(
    files: Union[PATH_TYPE, List[PATH_TYPE]],
    *additional_lists: List[Any],
    valid_extensions: Optional[Tuple[str, ...]] = None,
) -> Union[List[Any], Tuple[List[Any], ...]]:
    """Filter the given list of files and any additional lists to include only the entries that contain a file with
    a valid extension.

    Args:
        files: The list of files to filter by.
        additional_lists: Any additional lists to be filtered together with files.
        valid_extensions: The tuple of valid file extensions.

    Returns:
        The filtered lists.
    """
    if not isinstance(files, List):
        files = [files]

    if valid_extensions is None:
        return (files,) + additional_lists

    if not isinstance(valid_extensions, tuple):
        valid_extensions = tuple(valid_extensions)

    additional_lists = tuple([a] if not isinstance(a, List) else a for a in additional_lists)

    if not all(len(a) == len(files) for a in additional_lists):
        raise ValueError(
            f"The number of files ({len(files)}) and the number of items in any additional lists must be the same."
        )

    filtered = list(
        filter(lambda sample: has_file_allowed_extension(sample[0], valid_extensions), zip(files, *additional_lists))
    )

    filtered_files = [f[0] for f in filtered]

    invalid = [f for f in files if f not in filtered_files]

    if invalid:
        invalid_extensions = list({"." + f.split(".")[-1] for f in invalid})
        rank_zero_warn(
            f"Found invalid file extensions: {', '.join(invalid_extensions)}. "
            "Files with these extensions will be ignored. "
            f"The supported file extensions are: {', '.join(valid_extensions)}."
        )

    if additional_lists:
        return tuple(zip(*filtered))

    return filtered_files
