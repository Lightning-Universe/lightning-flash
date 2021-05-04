import pathlib
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

from PIL import Image
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS

from flash.data.data_source import FilesDataSource, FoldersDataSource


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
