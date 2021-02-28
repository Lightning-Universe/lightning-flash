from typing import Any, Callable, Mapping, Optional, Sequence

import torch


class _PreProcessor:

    def __init__(self, collate_fn: Callable, pre_collate: Callable, post_collate: Callable):
        self.collate_fn = collate_fn
        self.pre_collate = pre_collate
        self.post_collate = post_collate

    def __call__(self, samples: Sequence[Any]):
        samples = [self.pre_collate(sample) for sample in samples]
        samples = type(samples)(samples)
        samples = self.post_collate(self.collate_fn(samples))
        return samples

    def __repr__(self) -> str:
        repr_str = '_PreProcessor:'
        repr_str += f'\n\t(pre_collate): {repr(self.pre_collate)}'
        repr_str += f'\n\t(collate_fn): {repr(self.collate_fn)}'
        repr_str += f'\n\t(post_collate): {repr(self.post_collate)}'
        return repr_str


class _PostProcessor:

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
        else:
            return final_preds

    def __repr__(self) -> str:
        repr_str = '_PostProcessor:'
        repr_str += f'\n\t(pre_uncollate): {repr(self.pre_uncollate)}'
        repr_str += f'\n\t(uncollate_fn): {repr(self.uncollate_fn)}'
        repr_str += f'\n\t(post_uncollate): {repr(self.post_uncollate)}'

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
