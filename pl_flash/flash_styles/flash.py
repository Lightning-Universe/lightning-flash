from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union
import os
from pytorch_lightning.core.step_result import Result

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data._utils.collate import default_convert
from torch.optim import Optimizer, Adam

from pytorch_lightning import Trainer, LightningModule, LightningDataModule, EvalResult, TrainResult

from pl_flash.data.dataset import PredefinedSequenceDataset, PredefinedMappingDataset

# THis Shall become the base class for all tasks
class Flash(LightningModule):
    """
    A ``Flash`` is a basic definition of a specific task and this is ne most common baseclass providing good defaults to most tasks.

    Args:
        model: the model to use inside this task
        loss_functions: the functions to update the model with. All provided losses will be summed up. The resulting total loss 
            will also be used for checkpointing and early stopping in during evaluation.
        metrics: [description]. Defaults to None.

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
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

    def forward(self, *args, **kwargs) -> Any:
        return self.model(*args, **kwargs)

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=1e-4)

    @staticmethod
    def unpack_batch_sequence(batch: Sequence) -> Tuple[torch.Tensor, torch.Tensor]:
        return (batch[0], batch[1])

    @staticmethod
    def unpack_batch_mapping(batch: Mapping) -> Tuple[torch.Tensor, torch.Tensor]:
        return (batch['x'], batch['y'])

    def compute_metrics(self, result: Result, x: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor) -> Result:
        for k, v in self.metrics.items():
            result.log(k, v(y_hat, y))
        return result

    def compute_losses(self, x: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        total_loss = 0

        loss_dict = {}

        for name, loss in self.loss_functions.items():
            loss_val = loss(y_hat, y)
            total_loss = total_loss + loss_val
            loss_dict[name] = loss_val

        assert isinstance(total_loss, torch.Tensor)

        return total_loss, loss_dict



    def _step(self, batch: Union[Sequence, Mapping], batch_idx: int, prefix: str) -> Union[EvalResult, TrainResult]:
        if isinstance(batch, Sequence):
            x, y = self.unpack_batch_sequence(batch)
        elif isinstance(batch, Mapping):
            x, y = self.unpack_batch_mapping(batch)

        else:
            raise TypeError('Expected Type of sequence or mapping for batch, got {}'.format(type(batch).__name__))

        y_hat = self(x)

        total_loss = 0

        total_loss, loss_val_dict = self.compute_losses(x, y, y_hat)

        if prefix == 'train':
            result = TrainResult(total_loss, early_stop_on=total_loss, checkpoint_on=total_loss)
        elif prefix == 'val':
            result = EvalResult(early_stop_on=total_loss, checkpoint_on=total_loss)
        elif prefix == 'test':
            result = EvalResult()
        else:
            raise ValueError('Expected prefix to be one of \{train, val, test\} but got {}'.format(prefix))

        for k, v in loss_val_dict.items():
            result.log(k, v, on_epoch=True)

        result.log('total_loss', total_loss, on_epoch=True)

        for k, v in self.metrics.items():
            result.log(k, v(y_hat, y), on_epoch=True)

        return result

    def training_step(self, *args, **kwargs) -> Union[int, Result]:
        return self._step(*args, **kwargs, prefix='train')

    def validation_step(self, *args, **kwargs) -> EvalResult:
        return self._step(*args, **kwargs, prefix='val')

    def test_step(self, *args, **kwargs) -> EvalResult:
        return self._step(*args, **kwargs, prefix='test')
            

        






    



            

