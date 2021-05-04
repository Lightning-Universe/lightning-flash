import pathlib
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union

import torch
from PIL import Image
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
from torchvision.transforms.functional import to_pil_image

from flash.data.data_source import FilesDataSource, FoldersDataSource, NumpyDataSource, TensorDataSource


class ImageFoldersDataSource(FoldersDataSource):

    def __init__(self):
        super().__init__(extensions=IMG_EXTENSIONS)

    def load_sample(self, sample: Mapping[str, Any], dataset: Optional[Any] = None) -> Any:
        result = {}  # TODO: this is required to avoid a memory leak, can we automate this?
        result.update(sample)
        result['input'] = default_loader(sample['input'])
        return result


class ImageFilesDataSource(FilesDataSource):

    def __init__(self):
        super().__init__(extensions=IMG_EXTENSIONS)

    def load_sample(self, sample: Mapping[str, Any], dataset: Optional[Any] = None) -> Any:
        result = {}  # TODO: this is required to avoid a memory leak, can we automate this?
        result.update(sample)
        result['input'] = default_loader(sample['input'])
        return result


class ImageTensorDataSource(TensorDataSource):

    def load_sample(self, sample: Dict[str, Any], dataset: Optional[Any] = None) -> Any:
        sample['input'] = to_pil_image(sample['input'])
        return sample


class ImageNumpyDataSource(NumpyDataSource):

    def load_sample(self, sample: Dict[str, Any], dataset: Optional[Any] = None) -> Any:
        sample['input'] = to_pil_image(torch.from_numpy(sample['input']))
        return sample
