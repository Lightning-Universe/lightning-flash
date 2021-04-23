from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import kornia as K
import numpy as np
import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import Dataset

from flash.data.auto_dataset import AutoDataset
from flash.data.callback import BaseDataFetcher
from flash.data.data_module import DataModule
from flash.data.process import Preprocess


class SemantincSegmentationPreprocess(Preprocess):

    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
        image_size: Tuple[int, int] = (196, 196),
    ) -> 'SemantincSegmentationPreprocess':

        # TODO: implement me
        '''train_transform, val_transform, test_transform, predict_transform = self._resolve_transforms(
            train_transform, val_transform, test_transform, predict_transform
        )'''
        train_transform = dict(to_tensor_transform=self.to_tensor)
        val_transform = dict(to_tensor_transform=self.to_tensor)
        test_transform = dict(to_tensor_transform=self.to_tensor)
        predict_transform = dict(to_tensor_transform=self.to_tensor)

        super().__init__(train_transform, val_transform, test_transform, predict_transform)

    @staticmethod
    def to_tensor(self, x):
        return K.utils.image_to_tensor(np.array(x))

    def load_data(self, data: Any, dataset: Optional[AutoDataset] = None) -> Iterable:
        pass

    def load_sample(sample) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def collate(self, samples: Sequence) -> Any:
        pass

    def pre_tensor_transform(self, sample: Any) -> Any:
        pass

    def to_tensor_transform(self, sample: Any) -> Any:
        pass

    def post_tensor_transform(self, sample: Any) -> Any:
        pass

    def per_batch_transform(self, sample: Any) -> Any:
        pass

    def per_batch_transform_on_device(self, sample: Any) -> Any:
        pass


class SemanticSegmentationData(DataModule):
    """Data module for semantic segmentation tasks."""

    @staticmethod
    def _check_valid_filepaths(filepaths: List[str]):
        if filepaths is not None and (
            not isinstance(filepaths, list) or not all(isinstance(n, str) for n in filepaths)
        ):
            raise MisconfigurationException(f"`filepaths` must be of type List[str]. Got: {filepaths}.")

    @classmethod
    def from_filepaths(
        cls,
        train_filepaths: Optional[Sequence[str]] = None,
        train_labels: Optional[Sequence[str]] = None,
        val_filepaths: Optional[Sequence[str]] = None,
        val_labels: Optional[Sequence[str]] = None,
        test_filepaths: Optional[Sequence[str]] = None,
        test_labels: Optional[Sequence[str]] = None,
        predict_filepaths: Optional[Sequence[str]] = None,
        train_transform: Union[str, Dict] = 'default',
        val_transform: Union[str, Dict] = 'default',
        test_transform: Union[str, Dict] = 'default',
        predict_transform: Union[str, Dict] = 'default',
        image_size: Tuple[int, int] = (196, 196),
        batch_size: int = 64,
        num_workers: Optional[int] = None,
        #seed: Optional[int] = 42,  # SEED NEVER USED
        data_fetcher: BaseDataFetcher = None,
        preprocess: Optional[Preprocess] = None,
        # val_split: Optional[float] = None,  # MAKES IT CRASH. NEED TO BE FIXED
        #**kwargs,
    ) -> 'SemanticSegmentationData':

        # verify input data format
        SemanticSegmentationData._check_valid_filepaths(train_filepaths)
        SemanticSegmentationData._check_valid_filepaths(train_labels)
        SemanticSegmentationData._check_valid_filepaths(val_filepaths)
        SemanticSegmentationData._check_valid_filepaths(val_labels)
        SemanticSegmentationData._check_valid_filepaths(test_filepaths)
        SemanticSegmentationData._check_valid_filepaths(test_labels)
        SemanticSegmentationData._check_valid_filepaths(predict_filepaths)

        # create the preprocess objects
        preprocess = preprocess or SemantincSegmentationPreprocess(
            train_transform,
            val_transform,
            test_transform,
            predict_transform,
        )

        # instantiate the data module class
        return DataModule.from_load_data_inputs(
            train_load_data_input=list(zip(train_filepaths, train_labels)) if train_filepaths else None,
            val_load_data_input=list(zip(val_filepaths, val_labels)) if val_filepaths else None,
            test_load_data_input=list(zip(test_filepaths, test_labels)) if test_filepaths else None,
            predict_load_data_input=predict_filepaths,
            batch_size=batch_size,
            num_workers=num_workers,
            data_fetcher=data_fetcher,
            preprocess=preprocess,
            #seed=seed,
            #val_split=val_split,
            #**kwargs
        )
