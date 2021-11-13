import os
from typing import Any, List, Optional, Tuple, Union


# Copied from torchvision:
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py#L10
def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def isdir(data: Union[str, Tuple[List[str], List[Any]]]) -> bool:
    try:
        return os.path.isdir(data)
    except TypeError:
        # data is not path-like (e.g. it may be a list of paths)
        return False


def list_valid_files(data: Union[str, List[str]], valid_extensions: Optional[Tuple[str, ...]] = None):
    if isdir(data):
        data = [os.path.join(data, file) for file in os.listdir(data)]

    if not isinstance(data, list):
        data = [data]

    return list(
        filter(
            lambda file: has_file_allowed_extension(file, valid_extensions),
            data,
        )
    )
