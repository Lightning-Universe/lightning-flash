from typing import Any, Callable, Mapping, Optional, Sequence

import torch


class _PreProcessor:

    def __init__(self, collate_fn: Callable, per_sample_transform: Callable, per_batch_transform: Callable):
        self.collate_fn = collate_fn
        self.per_sample_transform = per_sample_transform
        self.per_batch_transform = per_batch_transform

    def __call__(self, samples: Sequence[Any]):
        samples = [self.per_sample_transform(sample) for sample in samples]
        samples = type(samples)(samples)
        samples = self.per_batch_transform(self.collate_fn(samples))
        return samples

    def __repr__(self) -> str:
        repr_str = '_PreProcessor:'
        repr_str += f'\n\t(per_sample_transform): {repr(self.per_sample_transform)}'
        repr_str += f'\n\t(collate_fn): {repr(self.collate_fn)}'
        repr_str += f'\n\t(per_batch_transform): {repr(self.per_batch_transform)}'
        return repr_str


class _PostProcessor:

    def __init__(
        self,
        uncollate_fn: Callable,
        per_batch_transform: Callable,
        per_sample_transform: Callable,
        save_fn: Optional[Callable] = None,
        save_per_sample: bool = False
    ):
        self.uncollate_fn = uncollate_fn
        self.per_batch_transform = per_batch_transform
        self.per_sample_transform = per_sample_transform
        self.save_fn = save_fn
        self.save_per_sample = save_per_sample

    def __call__(self, batch: Sequence[Any]):
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
