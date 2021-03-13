from typing import Any, Callable, Mapping, Optional, Sequence, Union

import torch
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from flash.data.utils import _contains_any_tensor, convert_to_modules


class _Chainer(torch.nn.Module):

    def __init__(
        self,
        per_sample_pre_tensor_transform: Callable,
        per_sample_to_tensor_transform: Callable,
        per_sample_post_tensor_transform: Callable,
        assert_contains_tensor: bool = False
    ):
        super().__init__()

        self.per_sample_pre_tensor_transform = convert_to_modules(per_sample_pre_tensor_transform)
        self.per_sample_to_tensor_transform = convert_to_modules(per_sample_to_tensor_transform)
        self.per_sample_post_tensor_transform = convert_to_modules(per_sample_post_tensor_transform)
        self.assert_contains_tensor = assert_contains_tensor

    def forward(self, sample: Any):
        sample = self.per_sample_pre_tensor_transform(sample)
        sample = self.per_sample_to_tensor_transform(sample)
        if self.assert_contains_tensor:
            if not _contains_any_tensor(sample):
                raise MisconfigurationException(
                    "When ``per_sample_to_tensor_transform`` is overriden, ``DataPipeline`` expects the outputs to be ``tensors``"
                )
        sample = self.per_sample_post_tensor_transform(sample)
        return sample

    def __repr__(self) -> str:
        repr_str = f'{self.__class__.__name__}:'
        repr_str += f'\n\t\t(per_sample_pre_tensor_transform): {repr(self.per_sample_pre_tensor_transform)}'
        repr_str += f'\n\t\t(per_sample_to_tensor_transform): {repr(self.per_sample_to_tensor_transform)}'
        repr_str += f'\n\t\t(per_sample_post_tensor_transform): {repr(self.per_sample_post_tensor_transform)}'
        repr_str += f'\n\t\t(assert_contains_tensor): {repr(self.assert_contains_tensor)}'
        return repr_str


class _PreProcessor(torch.nn.Module):

    def __init__(
        self,
        collate_fn: Callable,
        per_sample_transform: Union[Callable, _Chainer],
        per_batch_transform: Callable,
        stage: Optional[RunningStage] = None,
        apply_per_sample_transform: bool = True,
    ):
        super().__init__()
        self.collate_fn = convert_to_modules(collate_fn)
        self.per_sample_transform = convert_to_modules(per_sample_transform)
        self.per_batch_transform = convert_to_modules(per_batch_transform)
        self.apply_per_sample_transform = apply_per_sample_transform
        self.stage = stage

    def forward(self, samples: Sequence[Any]):
        if self.apply_per_sample_transform:
            samples = [self.per_sample_transform(sample) for sample in samples]
            samples = type(samples)(samples)
            samples = self.collate_fn(samples)
        samples = self.per_batch_transform(samples)
        return samples

    def __repr__(self) -> str:
        repr_str = '_PreProcessor:'
        repr_str += f'\n\t(per_sample_transform): {repr(self.per_sample_transform)}'
        repr_str += f'\n\t(collate_fn): {repr(self.collate_fn)}'
        repr_str += f'\n\t(per_batch_transform): {repr(self.per_batch_transform)}'
        repr_str += f'\n\t(apply_per_sample_transform): {repr(self.apply_per_sample_transform)}'
        repr_str += f'\n\t(stage): {repr(self.stage)}'
        return repr_str


class _PostProcessor(torch.nn.Module):

    def __init__(
        self,
        uncollate_fn: Callable,
        per_batch_transform: Callable,
        per_sample_transform: Callable,
        save_fn: Optional[Callable] = None,
        save_per_sample: bool = False
    ):
        super().__init__()
        self.uncollate_fn = convert_to_modules(uncollate_fn)
        self.per_batch_transform = convert_to_modules(per_batch_transform)
        self.per_sample_transform = convert_to_modules(per_sample_transform)
        self.save_fn = convert_to_modules(save_fn)
        self.save_per_sample = convert_to_modules(save_per_sample)

    def forward(self, batch: Sequence[Any]):
        uncollated = self.uncollate_fn(self.per_batch_transform(batch))

        final_preds = type(uncollated)([self.per_sample_transform(sample) for sample in uncollated])

        if self.save_fn is not None:
            if self.save_per_sample:
                for pred in final_preds:
                    self.save_fn(pred)
            else:
                self.save_fn(final_preds)
        else:
            return final_preds

    def __str__(self) -> str:
        repr_str = '_PostProcessor:'
        repr_str += f'\n\t(per_batch_transform): {repr(self.per_batch_transform)}'
        repr_str += f'\n\t(uncollate_fn): {repr(self.uncollate_fn)}'
        repr_str += f'\n\t(per_sample_transform): {repr(self.per_sample_transform)}'

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
