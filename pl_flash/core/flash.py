from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union
import os
from pytorch_lightning.core.step_result import Result

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data._utils.collate import default_convert
from torch.optim import Optimizer, Adam

from pytorch_lightning import (
    Trainer,
    LightningModule,
    LightningDataModule,
    EvalResult,
    TrainResult,
)

from pl_flash.data.dataset import PredefinedSequenceDataset, PredefinedMappingDataset

# This Shall become the base class for all tasks
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

    def __init__(
        self,
        model: torch.nn.Module,
        loss: Union[Callable, torch.nn.Module, dict],
        metrics: Optional[dict] = None,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        optimizer: Optimizer = Adam,
    ):
        def format_functions(
            functions: Union[Callable, torch.nn.Module, dict, list]
        ) -> Union[torch.nn.ModuleDict, dict]:
            if isinstance(functions, list):
                functions = {func.__name__: func for func in functions}

            if isinstance(functions, dict):
                if any([isinstance(func, torch.nn.Module) for func in functions.values()]):
                    functions = torch.nn.ModuleDict(functions)
            elif isinstance(functions, torch.nn.Module):
                functions = torch.nn.ModuleDict({functions.__name__: functions})

            elif callable(functions):
                functions = {functions.__name__: functions}
            else:
                raise ValueError(
                    "metrics must be of type dict, list, ModuleDict, torch.nn.Module or Callable, but got {}".format(
                        type(functions).__name__
                    )
                )

            return functions

        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate

        self.losses = format_functions(loss)
        self.metrics = format_functions(metrics) if metrics is not None else {}

    # def fit(self, data, val_split: Optional[float] = None):
    #    train_loader, val_loader = self._create_dloader(data, val_split)

    def fit(self, train_loader, val_loader=None, **kwargs):

        # TODO: Define self.trainer_kwargs
        # trainer = Trainer(**self.trainer_kwargs)

        trainer = Trainer(**kwargs)

        trainer.fit(self, train_dataloader=train_loader, val_dataloaders=[val_loader])

        # TODO: Load best model checkpoint
        return self

    def forward(self, *args, **kwargs) -> Any:
        return self.model(*args, **kwargs)

    def configure_optimizers(self) -> Optimizer:
        return self.optimizer(self.parameters(), lr=self.learning_rate)

    def compute_metrics(self, y_hat: torch.Tensor, y: torch.Tensor, prefix: str = "val") -> Result:
        return {f"{prefix}_{name}": metric(y_hat, y) for name, metric in self.metrics.items()}

    def compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor, prefix: str = "train") -> Result:
        loss_dict = {f"{prefix}_{name}": loss(y_hat, y) for name, loss in self.losses.items()}
        return sum(loss_dict.values()), loss_dict

    def _step(self, batch: Union[Sequence, Mapping], batch_idx: int, prefix: str) -> Union[EvalResult, TrainResult]:
        # if isinstance(batch, Sequence):
        #    x, y = self.unpack_batch_sequence(batch)
        # elif isinstance(batch, Mapping):
        #    x, y = self.unpack_batch_mapping(batch)

        # else:
        #    raise TypeError(
        #        "Expected Type of sequence or mapping for batch, got {}".format(
        #            type(batch).__name__
        #        )
        #    )
        x, y = batch
        y_hat = self(x)

        loss, loss_dict = self.compute_loss(y_hat, y)

        if prefix == "train":
            result = TrainResult(loss, early_stop_on=loss, checkpoint_on=loss)
        elif prefix == "val":
            result = EvalResult(early_stop_on=loss, checkpoint_on=loss)
        elif prefix == "test":
            result = EvalResult()
        else:
            raise ValueError("Expected prefix to be one of {{train, val, test}} but got {}".format(prefix))

        for k, v in loss_dict.items():
            result.log(k, v, on_epoch=True)

        result.log("total_loss", loss, on_epoch=True)

        for k, v in self.compute_metrics(y_hat, y, prefix).items():
            result.log(k, v, on_epoch=True)

        return result

    def training_step(self, *args, **kwargs) -> Union[int, Result]:
        return self._step(*args, **kwargs, prefix="train")

    def validation_step(self, *args, **kwargs) -> EvalResult:
        return self._step(*args, **kwargs, prefix="val")

    def test_step(self, *args, **kwargs) -> EvalResult:
        return self._step(*args, **kwargs, prefix="test")

    # @staticmethod
    # def unpack_batch_sequence(batch: Sequence) -> Tuple[torch.Tensor, torch.Tensor]:
    #     return (batch[0], batch[1])

    # @staticmethod
    # def unpack_batch_mapping(batch: Mapping) -> Tuple[torch.Tensor, torch.Tensor]:
    #     return (batch['x'], batch['y'])


# def _create_dloader(batch_size: int, data, val_split: Optional[float]):
#     data = default_convert(data)
#
#     if isinstance(data, LightningDataModule):
#         train_loader, val_loader = data.train_dataloader, data.val_dataloader
#
#     elif isinstance(data, DataLoader):
#         # TODO: shall we split here as well? I don't think so.
#         train_loader, val_loader = data, None
#
#     else:
#         if isinstance(data, Sequence):
#             data = PredefinedSequenceDataset(data)
#
#         elif isinstance(data, Mapping):
#             data = PredefinedMappingDataset(data)
#
#         if not isinstance(data, Dataset):
#             raise ValueError(
#                 "expected type of DataModule, DataLoader, Dataset, Sequence or Mapping but got {}".format(
#                     type(data).__name__
#                 )
#             )
#
#         if val_split is None:
#             train_set, val_set = data, None
#         else:
#             val_set_length = int(round(len(data) * val_split))
#             train_set_length = len(data) - val_set_length
#             train_set, val_set = random_split(
#                 data,
#                 [train_set_length, val_set_length],
#                 generator=torch.Generator().manual_seed(42),
#             )
#
#         train_loader = DataLoader(
#             train_set, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(),
#         )
#
#         if val_set is None:
#             val_loader = None
#         else:
#             val_loader = DataLoader(
#                 val_set,
#                 batch_size=batch_size,
#                 shuffle=False,
#                 num_workers=os.cpu_count(),
#             )
#
#     return train_loader, val_loader
