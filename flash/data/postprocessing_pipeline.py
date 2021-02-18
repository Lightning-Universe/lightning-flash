import os
from functools import wraps
from typing import Any, Callable, Mapping, Optional, Sequence

import torch

from flash.core.model import Task


class PostProcessingPipeline:

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

    def format_sample_save_path(self, path: str) -> None:
        path = os.path.join(path, f'sample_{self._saved_samples}.ptl')
        self._saved_samples += 1
        return path

    def _save_data(self, data: Any) -> None:
        self.save_data(data, self._save_path)

    def _save_sample(self, sample: Any) -> None:
        self.save_sample(sample, self.format_sample_save_path(self._save_path))

    def is_overriden(self, method_name: str) -> bool:
        """Cropped Version of https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/utilities/model_helpers.py
        """

        super_obj = PostProcessingPipeline

        if not hasattr(self, method_name) or not hasattr(super_obj, method_name):
            return False

        return getattr(self, method_name).__code__ is not getattr(super_obj, method_name)

    @staticmethod
    def model_predict_wrapper(func: Callable, uncollater: UnCollater) -> Callable:

        @wraps(func)
        def new_func(*args, **kwargs):
            predicted = func(*args, **kwargs)
            return uncollater(predicted)

        return new_func

    def attach_to_model(self, model: Task) -> Task:

        if self._save_path is None:
            save_per_sample = None
            save_fn = None

        else:
            save_per_sample = self.is_overriden('save_sample')

            if save_per_sample:
                save_fn = self._save_sample
            else:
                save_fn = self._save_data
        model.predict = self.model_predict_wrapper(
            model.predict,
            UnCollater(
                self.uncollate,
                self.pre_uncollate,
                self.post_uncollate,
                save_fn=save_fn,
                save_per_sample=save_per_sample
            )
        )
        return model


class UnCollater:

    def __init__(
        self,
        uncollate_fn: Callable,
        pre_uncollate: Callable,
        post_uncollate: Callable,
        save_fn: Optional[Callable] = None,
        save_per_sample: bool = False
    ):
        self.uncollate_fn = uncollate_fn
        self.pre_uncollate = pre_uncollate
        self.post_uncollate = post_uncollate

        self.save_fn = save_fn
        self.save_per_sample = save_per_sample

    def __call__(self, batch: Sequence[Any]):
        uncollated = self.uncollate_fn(self.pre_uncollate(batch))

        final_preds = type(uncollated)([self.post_uncollate(sample) for sample in uncollated])

        if self.save_fn is not None:
            if self.save_per_sample:
                for pred in final_preds:
                    self.save_fn(pred)
            else:
                self.save_fn(final_preds)

    def __repr__(self) -> str:
        repr_str = f'UnCollater:\n\t(pre_uncollate): {repr(self.pre_uncollate)}\n\t(uncollate_fn): {repr(self.uncollate_fn)}\n\t(post_uncollate): {repr(self.post_uncollate)}'
        return repr_str


def default_uncollate(batch: Any):

    batch_type = type(batch)

    if isinstance(batch, torch.Tensor):
        return list(torch.unbind(batch, 0))

    elif isinstance(batch, Mapping):
        return [batch_type(dict(zip(batch, default_uncollate(t)))) for t in zip(*batch.values())]

    elif isinstance(batch, tuple) and hasattr(batch, '_fields'):  # namedtuple
        return [batch_type(*default_uncollate(sample)) for sample in zip(*batch)]

    elif isinstance(batch, Sequence) and not isinstance(batch, str):
        return [default_uncollate(sample) for sample in batch]

    return batch
