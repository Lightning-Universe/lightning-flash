import os
from typing import Any, Optional

import torch

from flash.data.batch import default_uncollate


class Preprocess:

    def load_data(self, data: Any, dataset: Optional[Any]) -> Any:
        """Loads entire data from Dataset"""
        return data

    def load_sample(self, sample: Any) -> Any:
        """Loads single sample from dataset"""
        return sample

    def pre_collate(self, sample: Any) -> Any:
        """Transforms to apply to the data before the collation (per-sample basis)"""
        return sample

    def post_collate(self, batch: Any) -> Any:
        """Transforms to apply to a whole batch (if possible use this for efficiency)
        .. note::
            This option is mutually exclusive with :meth:`device_pre_collate`,
            since if both are specified, uncollation has to be applied.
        """
        return batch

    def device_pre_collate(self, sample: Any) -> Any:
        """Transforms to apply to the data before the collation (per-sample basis).
        .. note::
            This option is mutually exclusive with :meth:`post_collate`,
            since if both are specified, uncollation has to be applied.
        .. note::
            This function won't be called within the dataloader workers, since to make that happen
            each of the workers would have to create it's own CUDA-context which would pollute GPU memory (if on GPU).
        """
        return sample

    def device_post_collate(self, batch: Any) -> Any:
        """
        Transforms to apply to a whole batch (if possible use this for efficiency).
        .. note::
            This function won't be called within the dataloader workers, since to make that happen
            each of the workers would have to create it's own CUDA-context which would pollute GPU memory (if on GPU).
        """
        return batch


class Postprocess:

    def __init__(self, save_path: Optional[str] = None):
        self._saved_samples = 0
        self._save_path = save_path

    def pre_uncollate(self, batch: Any) -> Any:
        """Transforms to apply to a whole batch before uncollation to single samples.
        Can involve both CPU and Device transforms as this is not applied in separate workers.
        """
        return batch

    def post_uncollate(self, sample: Any) -> Any:
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
