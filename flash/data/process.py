import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Union

import torch
from pytorch_lightning.utilities.apply_func import apply_to_collection
from torch.nn import Module, ModuleDict, ModuleList
from torch.utils.data._utils.collate import default_collate

from flash.data.batch import default_uncollate


class FuncModule(torch.nn.Module):

    def __init__(self, func) -> None:
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def _convert_to_modules(transforms: Dict):

    if transforms is None or isinstance(transforms, Module):
        return transforms

    elif isinstance(transforms, Mapping) and not isinstance(transforms, ModuleDict):
        for k, v in transforms.items():
            transforms[k] = v if isinstance(v, Module) else FuncModule(v)
        return ModuleDict(transforms)

    elif isinstance(transforms, Iterable) and not isinstance(transforms, ModuleList):
        return ModuleList([v if isinstance(v, Module) else FuncModule(v) for v in transforms])

    else:
        return FuncModule(transforms)


class Preprocess(torch.nn.Module):

    def __init__(
        self,
        train_transform: Optional[Union[Callable, Module, Dict[str, Callable]]] = None,
        valid_transform: Optional[Union[Callable, Module, Dict[str, Callable]]] = None,
        test_transform: Optional[Union[Callable, Module, Dict[str, Callable]]] = None,
        predict_transform: Optional[Union[Callable, Module, Dict[str, Callable]]] = None,
    ):
        super().__init__()
        self.train_transform = _convert_to_modules(train_transform)
        self.valid_transform = _convert_to_modules(valid_transform)
        self.test_transform = _convert_to_modules(test_transform)
        self.predict_transform = _convert_to_modules(predict_transform)

    @classmethod
    def load_data(cls, data: Any, dataset: Optional[Any] = None) -> Any:
        """Loads entire data from Dataset"""
        return data

    @classmethod
    def load_sample(cls, sample: Any, dataset: Optional[Any] = None) -> Any:
        """Loads single sample from dataset"""
        return sample

    def per_sample_transform(self, sample: Any) -> Any:
        """Transforms to apply to the data before the collation (per-sample basis)"""
        return sample

    def per_batch_transform(self, batch: Any) -> Any:
        """Transforms to apply to a whole batch (if possible use this for efficiency)
        .. note::
            This option is mutually exclusive with :meth:`per_sample_transform_on_device`,
            since if both are specified, uncollation has to be applied.
        """
        return batch

    def collate(self, samples: Sequence) -> Any:
        return default_collate(samples)

    def per_sample_transform_on_device(self, sample: Any) -> Any:
        """Transforms to apply to the data before the collation (per-sample basis).
        .. note::
            This option is mutually exclusive with :meth:`per_batch_transform`,
            since if both are specified, uncollation has to be applied.
        .. note::
            This function won't be called within the dataloader workers, since to make that happen
            each of the workers would have to create it's own CUDA-context which would pollute GPU memory (if on GPU).
        """
        return sample

    def per_batch_transform_on_device(self, batch: Any) -> Any:
        """
        Transforms to apply to a whole batch (if possible use this for efficiency).
        .. note::
            This function won't be called within the dataloader workers, since to make that happen
            each of the workers would have to create it's own CUDA-context which would pollute GPU memory (if on GPU).
        """
        return batch


class Postprocess(torch.nn.Module):

    def __init__(self, save_path: Optional[str] = None):
        super().__init__()
        self._saved_samples = 0
        self._save_path = save_path

    def per_batch_transform(self, batch: Any) -> Any:
        """Transforms to apply to a whole batch before uncollation to single samples.
        Can involve both CPU and Device transforms as this is not applied in separate workers.
        """
        return batch

    def per_sample_transform(self, sample: Any) -> Any:
        """Transforms to apply to a single sample after splitting up the batch.
        Can involve both CPU and Device transforms as this is not applied in separate workers.
        """
        return sample

    def uncollate(self, batch: Any) -> Any:
        """Uncollates a batch into single samples.
        Tries to preserve the type whereever possible.
        """
        return default_uncollate(batch)

    def save_data(self, data: Any, path: str) -> None:
        """Saves all data together to a single path.
        """
        torch.save(data, path)

    def save_sample(self, sample: Any, path: str) -> None:
        """Saves each sample individually to a given path.
        """
        torch.save(sample, path)

    # TODO: Are those needed ?
    def format_sample_save_path(self, path: str) -> str:
        path = os.path.join(path, f'sample_{self._saved_samples}.ptl')
        self._saved_samples += 1
        return path

    def _save_data(self, data: Any) -> None:
        self.save_data(data, self._save_path)

    def _save_sample(self, sample: Any) -> None:
        self.save_sample(sample, self.format_sample_save_path(self._save_path))
