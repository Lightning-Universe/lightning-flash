from os import stat
import pathlib
import os
from typing import Tuple, Union, Callable
import torch

class _FileSuffixDataset(torch.utils.data.Dataset):
    def __init__(self, root_path: Union[str, pathlib.Path], loader: Callable, suffix: str = '_mask') -> None:
        super().__init__()

        self.data = self.parse_dir(root_path, suffix)

        self.loader = loader

    @staticmethod
    def parse_dir(path: Union[str, pathlib.Path], suffix: str) -> tuple:
        path = str(path)

        data = []

        for item in [os.path.join(path, x) for x in os.listdir(path)]:
            if not os.path.isdir(item):
                continue

            file, ext = os.path.splitext(item)

            mask_file = file + suffix + os.path.extsep + ext

            if os.path.isfile(mask_file):
                data.append((item, mask_file))

        return tuple(data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data_file, mask_file = self.data[index]

        data = self.loader(data_file)
        mask = self.loader(mask_file)

        data = self.format_data(data).float()
        mask = self.format_mask(mask).long()

        return data, mask


    @staticmethod
    def format_data(data: torch.Tensor) -> torch.Tensor:
        if data.ndim == 2:
            return data[None]

        if data.ndim == 3:
            if data.size(0) in [1, 3, 4]:
                return data
            elif data.size(-1) in [1, 3, 4]:
                return data.permute(2, 0, 1)

        return data

    @staticmethod
    def format_mask(mask: torch.Tensor) -> torch.Tensor:
        if mask.ndim == 2:
            return mask

        if mask.ndim == 3:
            if mask.size(0) == 1:
                return mask[0]

            elif mask.size(-1) == 1:
                return mask[..., 0]

            else:
                return mask.argmax(0)

        return mask

    def __len__(self) -> int:
        return len(self.data)

class _FilePathDirDataset(_FileSuffixDataset):
    def __init__(self, image_path: Union[str, pathlib.Path], mask_path, loader: Callable) -> None:
        super().__init__((image_path, mask_path), loader, '')

    def parse_dir(path: Tuple[Union[str, pathlib.Path], Union[str, pathlib.Path]]) -> tuple:
        image_path, mask_path = [str(_path) for _path in path]

        data = []

        for file in os.listdir(image_path):
            image_file = os.path.join(image_path, file)
            mask_file = os.path.join(mask_path, file)

            if os.path.isfile(image_file) and os.path.isfile(mask_file):
                data.append((image_file, mask_file))

        return tuple(data)
