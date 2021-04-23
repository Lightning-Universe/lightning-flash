from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

import kornia as K
import numpy as np
import torch
from torch.utils.data import Dataset

from flash.data.auto_dataset import AutoDataset
from flash.data.data_module import DataModule
from flash.data.process import Preprocess


class SemantincSegmentationPreprocess(Preprocess):

    def __init__(
        self,
        train_transform: Optional[Dict[str, Callable]] = None,
        val_transform: Optional[Dict[str, Callable]] = None,
        test_transform: Optional[Dict[str, Callable]] = None,
        predict_transform: Optional[Dict[str, Callable]] = None,
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

    def __init__(
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        predict_dataset: Optional[Dataset] = None,
        batch_size: int = 1,
        num_workers: Optional[int] = None,
        seed: int = 1234,
        train_split: Optional[float] = None,
        val_split: Optional[float] = None,
        # test_split: Optional[float] = None,  ## THIS WILL GO OUT
        preprocess: Optional[Preprocess] = None,
    ) -> None:
        pass

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
        batch_size: int = 64,
        num_workers: Optional[int] = None,
        seed: Optional[int] = 42,
        preprocess: Optional[Preprocess] = None,
        val_split: Optional[float] = None,
    ) -> 'SemanticSegmentationData':

        preprocess = preprocess or SemantincSegmentationPreprocess(
            train_transform,
            val_transform,
            test_transform,
            predict_transform,
        )

        return cls()
        '''return cls.from_load_data_inputs(
            train_load_data_input=list(zip(train_filepaths, train_labels)) if train_filepaths else None,
            val_load_data_input=list(zip(val_filepaths, val_labels)) if val_filepaths else None,
            test_load_data_input=list(zip(test_filepaths, test_labels)) if test_filepaths else None,
            predict_load_data_input=predict_filepaths,
            batch_size=batch_size,
            num_workers=num_workers,
            preprocess=preprocess,
            seed=seed,
        )'''
