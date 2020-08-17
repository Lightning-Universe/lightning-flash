from typing import Callable, Mapping, Optional, Sequence, Union
import os

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data._utils.collate import default_convert
from torch.optim import Optimizer, Adam

from pytorch_lightning import Trainer, LightningModule, LightningDataModule

from pl_flash.data.dataset import PredefinedSequenceDataset, PredefinedMappingDataset

# THis Shall become the base class for all tasks
class Flash(LightningModule):
    def __init__(self, model: torch.nn.Module, loss_functions: Union[Callable, torch.nn.Module, dict], metrics: Optional[dict] = None):
        super().__init__()
        self.model = model

        def format_loss_metric(functions: Union[Callable, torch.nn.Module, dict], default_name: str) -> Union[torch.nn.ModuleDict, dict]:
            if isinstance(functions, dict) and not isinstance(functions, torch.nn.ModuleDict) and any([isinstance(func, torch.nn.Module) for func in functions.values()]):
                functions = torch.nn.ModuleDict(functions)

            elif isinstance(functions, torch.nn.Module):
                functions = torch.nn.ModuleDict({'loss': functions})

            elif callable(loss_functions):
                functions = {default_name: loss_functions}
            else:
                raise ValueError('{} must be of type dict, ModuleDict, torch.nn.Module or Callable, but got {}'.format(default_name, type(functions).__name__))

            return functions
        
        self.loss_functions = format_loss_metric(loss_functions, 'loss')

        if metrics is None:
            metrics = {}
        else:
            metrics = format_loss_metric(metrics, 'metric')
    def fit(self, data, val_split: Optional[float]=None):
        train_loader, val_loader = self._create_dloader(data, val_split)

        # TODO: Define self.trainer_kwargs
        trainer = Trainer(**self.trainer_kwargs)

        trainer.fit(self, train_dataloader=train_loader, val_dataloaders=val_loader)

        # TODO: Load best model checkpoint

        return self

    def _create_dloader(self, data, val_split: Optional[float]): 
        data = default_convert(data)

        if isinstance(data, LightningDataModule):
            train_loader, val_loader = data.train_dataloader, data.val_dataloader

        elif isinstance(data, DataLoader):
            # TODO: shall we split here as well? I don't think so.
            train_loader, val_loader = data, None

        else:
            if isinstance(data, Sequence):
                data = PredefinedSequenceDataset(data)

            elif isinstance(data, Mapping):
                data = PredefinedMappingDataset(data)

            if not isinstance(data, Dataset):
                raise ValueError("expected type of DataModule, DataLoader, Dataset, Sequence or Mapping but got {}".format(type(data).__name__))

            if val_split is None:
                train_set, val_set = data, None
            else:
                val_set_length = int(round(len(data) * val_split))
                train_set_length = len(data) - val_set_length
                train_set, val_set = random_split(data, [train_set_length, val_set_length], generator=torch.Generator().manual_seed(42))

            # TODO: Define self.batch_size
            train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count())

            if val_set is None:
                val_loader = None
            else:
                val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count())

        return train_loader, val_loader

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=1e-4)


    



            

