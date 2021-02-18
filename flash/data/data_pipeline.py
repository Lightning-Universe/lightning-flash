from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union
import torch
from functools import wraps
from torch.utils.data.dataloader import default_collate, DataLoader
from pytorch_lightning.core import LightningModule

from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.trainer.connectors.data_connector import _PatchDataLoader


class DataPipeline:

    def load_data(self, data: Any) -> Any:
        """Loads entire data from Dataset"""

    def load_sample(self, sample: Any) -> Any:
        """Loads single sample from dataset"""

    def pre_collate(self, sample: Any) -> Any:
        """Transforms to apply to the data before the collation (per-sample basis)"""
        return sample

    def post_collate(self, batch: Any) -> Any:
        """Transforms to apply to a whole batch (if possible use this for efficiency)
        
        .. note::
            This option is mutually exclusive with :meth:`device_pre_collate`, since if both are specified, uncollation has to be applied.
        """
        return batch

    def device_pre_collate(self, sample: Any) -> Any:
        """Transforms to apply to the data before the collation (per-sample basis).
        
        .. note::
            This option is mutually exclusive with :meth:`post_collate`, since if both are specified, uncollation has to be applied.

        .. note::
            This function won't be called within the dataloader workers, since to make that happen each of the workers would have to create it's own CUDA-context which would pollute GPU memory (if on GPU).
        """
        return sample

    def device_post_collate(self, batch: Any) -> Any:
        """
        Transforms to apply to a whole batch (if possible use this for efficiency).

        .. note::
            This function won't be called within the dataloader workers, since to make that happen each of the workers would have to create it's own CUDA-context which would pollute GPU memory (if on GPU).
        """
        return batch

    def is_overriden(self, method_name: str) -> bool:
        """Cropped Version of https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/utilities/model_helpers.py
        """

        super_obj = DataPipeline

        if not hasattr(self, method_name) or not hasattr(super_obj, method_name):
            return False

        return getattr(self, method_name).__code__ is not getattr(super_obj, method_name)

    @staticmethod
    def do_nothing_collate(samples: Sequence[Any]) -> Sequence[Any]:
        return samples

    def split_around_collate(self, collate_fn: Optional[Callable] = None) -> Tuple[Collater, Collater]:

        if collate_fn is None:
            collate_fn = default_collate

        post_collate_overriden = self.is_overriden('post_collate')
        device_pre_collate_overriden = self.is_overriden('device_pre_collate')

        if post_collate_overriden and device_pre_collate_overriden:
            raise MisconfigurationException(
                f'{self.__class__.__name__}: post_collate and gpu_pre_collate are mutual exclusive.'
            )

        elif post_collate_overriden:
            worker_collate = collate_fn
            device_collate = self.do_nothing_collate

        elif device_pre_collate_overriden:
            worker_collate = self.do_nothing_collate
            device_collate = collate_fn

        else:
            worker_collate = collate_fn
            device_collate = self.do_nothing_collate

        worker_callable = Collater(worker_collate, self.pre_collate, self.post_collate)
        device_callable = Collater(device_collate, self.device_pre_collate, self.device_post_collate)

        return worker_callable, device_callable

    @staticmethod
    def model_transfer_to_device_wrapper(func: Callable, collater: Collater) -> Callable:

        @wraps(func)
        def new_func(*args, **kwargs):
            moved_to_device = func(*args, **kwargs)
            return collater(moved_to_device)

        return new_func

    def attach_to_model(self, model: LightningModule, loader_stage: str = 'all') -> LightningModule:
        if loader_stage == 'all':
            loader_stage = ['train', 'test', 'val', 'predict']

        elif isinstance(loader_stage, str):
            loader_stage = [loader_stage]

        for stage in loader_stage:
            loader_name = f'{stage}_loader'

            if hasattr(model, loader_name):
                dataloader = getattr(model, loader_name)

                if isinstance(dataloader, _PatchDataLoader):
                    wrap_patch_loader = True
                    dataloader = dataloader()

                else:
                    wrap_patch_loader = False

                if isinstance(dataloader, Sequence):
                    was_seq = True
                else:
                    dataloader = [dataloader]
                    was_seq = False

                for idx, loader in enumerate(dataloader):
                    if isinstance(loader, DataLoader):
                        dl_args = {k: v for k, v in vars(loader).items() if not k.startswith("_")}

                        dl_args['collate_fn'], device_collater = self.split_around_collate(
                            collate_fn=dl_args['collate_fn']
                        )

                        loader = type(loader)(**dl_args)

                    dataloader[idx] = loader

                if not was_seq:
                    dataloader = dataloader[0]

                if wrap_patch_loader:
                    dataloader = _PatchDataLoader(dataloader)

                setattr(model, loader_name, dataloader)

        model.transfer_batch_to_device = (
            self.model_transfer_to_device_wrapper(model.transfer_batch_to_device, device_collater)
        )
        return model

    def generate_auto_dset(self, data: Union[Iterable, Any]):
        if isinstance(data, Iterable) and self.is_overriden('load_sample'):
            load_per_sample = True
            load_fn = self.load_sample
        else:
            load_per_sample = False
            load_fn = self.load_data

        return AutoDataset(data=data, load_fn=load_fn, load_per_sample=load_per_sample)


class Collater:

    def __init__(self, collate_fn: Callable, pre_collate: Callable, post_collate: Callable):
        self.collate_fn = collate_fn
        self.pre_collate = pre_collate
        self.post_collate = post_collate

    def __call__(self, samples: Sequence[Any]):
        return self.post_collate(self.collate_fn(type(samples)([self.pre_collate(sample) for sample in samples])))

    def __repr__(self) -> str:
        repr_str = f'Collater:\n\t(pre_collate): {repr(self.pre_collate)}\n\t(collate_fn): {repr(self.collate_fn)}\n\t(post_collate): {repr(self.post_collate)}'
        return repr_str


class AutoDataset(torch.utils.data.Dataset):

    def __init__(self, data: Union[Iterable, Any], load_fn: Callable, load_per_sample: bool) -> None:
        super().__init__()

        self.data = data
        self.load_fn = load_fn

        self._load_lazy = load_per_sample

        if not self._load_lazy:
            self.data = self.load_fn(data)

    def __getitem__(self, index: int) -> Any:
        sample = self.data[index]

        if self._load_lazy:
            sample = self.load_fn(sample)

    def __len__(self) -> int:
        return len(self.data)
