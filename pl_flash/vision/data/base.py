from typing import Union
import pathlib
import os

import torch

from pl_flash.data import BaseData

__all__ = ["VisionData"]


class VisionData(BaseData):
    """Extension to :class:`BaseData` to also load iamge specific formats using pillow

    Raises:
        ImportError: PIL or numpy are not available
        RuntimeError: Unsupported file extension

    """
    @staticmethod
    def load_file(path: Union[str, pathlib.Path], **kwargs) -> torch.Tensor:
        """loads a file. Supported formats are:
            * PyTorch (.pt)
            * NumPy (.npy, .npz, .txt)
            * Pickle (.pkl)
            * Image files (.png, .jpg, .jpeg, ...)

        Args:
            path: the path containing the file to load

        Returns:
            torch.Tensor: the tensor containing the original file content
        """
        path = str(path)
        try:
            return BaseData.load_file(path, **kwargs)

        except RuntimeError:
            return torch.utils.data._utils.collate.default_convert(VisionData.load_pillow(path, **kwargs))

    @staticmethod
    def load_pillow(path, **kwargs) -> torch.Tensor:
        """loads image files with pillow

        Args:
            path: the path pointing to the image file to load

        Raises:
            ImportError: PIL or numpy not installed
            RuntimeError: file format not supported by PIL

        Returns:
            torch.Tensor: the tensor containing the image contents
        """
        try:
            from PIL import Image
            from PIL.Image import EXTENSION as IMAGE_EXTENSIONS, init as extensions_init # used to check supported image extensions
        except ImportError as e:
            raise ImportError("PIL is not available. Please install it with `pip install pillow`") from e

        try:
            import numpy as np
        except ImportError as e:
            raise ImportError("numpy is not available. Please install it with `pip install numpy`") from e

        ext = os.path.splitext(path)[1]

        extensions_init()
        if ext not in IMAGE_EXTENSIONS:
            raise RuntimeError

        loaded = Image.open(path, **kwargs)

        return torch.from_numpy(np.array(loaded))
