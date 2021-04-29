import pathlib
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

from PIL import Image
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS

from flash.data.data_source import FilesDataSource, FoldersDataSource, SequenceDataSource


class ImageFoldersDataSource(FoldersDataSource):

    def __init__(
        self,
        train_folder: Optional[Union[str, pathlib.Path, list]] = None,
        val_folder: Optional[Union[str, pathlib.Path, list]] = None,
        test_folder: Optional[Union[str, pathlib.Path, list]] = None,
        predict_folder: Optional[Union[str, pathlib.Path, list]] = None,
    ):
        super().__init__(
            train_folder=train_folder,
            val_folder=val_folder,
            test_folder=test_folder,
            predict_folder=predict_folder,
            extensions=IMG_EXTENSIONS,
        )

    def load_sample(self, sample: Mapping[str, Any], dataset: Optional[Any] = None) -> Any:
        result = {}  # TODO: this is required to avoid a memory leak, can we automate this?
        result.update(sample)
        result['input'] = default_loader(sample['input'])
        return result


class ImageFilesDataSource(FilesDataSource):

    def __init__(
        self,
        train_files: Optional[Union[Sequence[Union[str, pathlib.Path]], Iterable[Union[str, pathlib.Path]]]] = None,
        train_targets: Optional[Union[Sequence[Any], Iterable[Any]]] = None,
        val_files: Optional[Union[Sequence[Union[str, pathlib.Path]], Iterable[Union[str, pathlib.Path]]]] = None,
        val_targets: Optional[Union[Sequence[Any], Iterable[Any]]] = None,
        test_files: Optional[Union[Sequence[Union[str, pathlib.Path]], Iterable[Union[str, pathlib.Path]]]] = None,
        test_targets: Optional[Union[Sequence[Any], Iterable[Any]]] = None,
        predict_files: Optional[Union[Sequence[Union[str, pathlib.Path]], Iterable[Union[str, pathlib.Path]]]] = None,
    ):
        super().__init__(  # TODO: This feels like it can be simplified
            train_files=train_files,
            train_targets=train_targets,
            val_files=val_files,
            val_targets=val_targets,
            test_files=test_files,
            test_targets=test_targets,
            predict_files=predict_files,
            extensions=IMG_EXTENSIONS
        )

    def load_sample(self, sample: Mapping[str, Any], dataset: Optional[Any] = None) -> Any:
        result = {}  # TODO: this is required to avoid a memory leak, can we automate this?
        result.update(sample)
        result['input'] = default_loader(sample['input'])
        return result
