from typing import Any, Callable, Mapping, Sequence, Tuple, Type, Union
from pytorch_lightning.core.step_result import Result

import torch
import torch.optim
from torch.optim import Optimizer

from pytorch_lightning import LightningModule, EvalResult, TrainResult

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
        loss: Union[Callable, torch.nn.Module, Mapping, Sequence],
        metrics: Union[Callable, torch.nn.Module, Mapping, Sequence, None] = None,
        learning_rate: float = 1e-3,
        optimizer: Union[Type[Optimizer], str] = "Adam",
    ):
        def format_functions(
            functions: Union[Callable, torch.nn.Module, Mapping, Sequence]
        ) -> Union[torch.nn.ModuleDict, dict]:
            def get_name(func):
                try:
                    return func.__name__
                except AttributeError:
                    return func.__class__.__name__

            def resolve_str(item: Union[Any, str]):
                if isinstance(item, str):
                    try:
                        item = getattr(torch.nn, item)()
                    except AttributeError:
                        try:
                            item = getattr(torch.nn.functional, item)

                        except AttributeError:
                            raise ValueError(
                                f"{item} is not known. We looked in torch.nn and torch.nn.functional for it."
                                + "Please either adapt the value or provide a callable directly"
                            )

                return item

            if isinstance(functions, str):
                functions = resolve_str(functions)

            if isinstance(functions, Sequence):
                functions = {get_name(func): resolve_str(func) for func in functions}

            if isinstance(functions, Mapping) and not isinstance(
                functions, torch.nn.ModuleDict
            ):
                functions = {k: resolve_str(v) for k, v in functions.items()}
                if any(
                    [isinstance(func, torch.nn.Module) for func in functions.values()]
                ):
                    functions = torch.nn.ModuleDict(functions)
            elif isinstance(functions, torch.nn.Module):
                functions = torch.nn.ModuleDict({get_name(functions): functions})

            elif callable(functions):
                functions = {get_name(functions): functions}
            else:
                raise ValueError(
                    "metrics must be of type dict, list, ModuleDict, torch.nn.Module or Callable, but got {}".format(
                        type(functions).__name__
                    )
                )

            return functions

        super().__init__()
        self.save_hyperparameters()
        self.model = model

        if isinstance(optimizer, str):
            optimizer = getattr(torch.optim, optimizer)
        self.optimizer_cls = optimizer
        self.learning_rate = learning_rate

        self.losses = format_functions(loss)
        self.metrics = format_functions(metrics) if metrics is not None else {}

    def forward(self, *args, **kwargs) -> Any:
        return self.model(*args, **kwargs)

    def configure_optimizers(self) -> Optimizer:
        return self.optimizer_cls(self.parameters(), lr=self.learning_rate)

    def compute_loss(
        self,
        y_hat: torch.Tensor,
        y: torch.tensor,
        prefix: str = "train",
        sep: str = "/",
    ):
        loss_dict = self.compute_dict(self.losses, y_hat, y, prefix, sep)
        return sum(loss_dict.values()), loss_dict

    def compute_metrics(
        self, y_hat: torch.Tensor, y: torch.tensor, prefix: str = "val", sep: str = "/"
    ):
        return self.compute_dict(self.metrics, y_hat, y, prefix, sep)

    def _step(
        self, batch: Union[Sequence, Mapping], batch_idx: int, prefix: str
    ) -> Union[EvalResult, TrainResult]:

        x, y = self.unpack_batch(batch)
        y_hat = self(x)

        loss, loss_dict = self.compute_loss(y_hat, y, prefix=prefix)

        if prefix == "train":
            result = TrainResult(loss, early_stop_on=loss, checkpoint_on=loss)
        elif prefix == "val":
            result = EvalResult(early_stop_on=loss, checkpoint_on=loss)
        elif prefix == "test":
            result = EvalResult()
        else:
            raise ValueError(
                "Expected prefix to be one of {{train, val, test}} but got {}".format(
                    prefix
                )
            )

        for k, v in loss_dict.items():
            result.log(k, v, on_epoch=True, on_step=True)

        result.log(f"{prefix}/loss", loss, on_epoch=True, on_step=True)

        for k, v in self.compute_metrics(y_hat, y, prefix).items():
            result.log(k, v, on_epoch=True, on_step=True)

        return result

    def training_step(self, *args, **kwargs) -> Union[int, Result]:
        return self._step(*args, **kwargs, prefix="train")

    def validation_step(self, *args, **kwargs) -> EvalResult:
        return self._step(*args, **kwargs, prefix="val")

    def test_step(self, *args, **kwargs) -> EvalResult:
        return self._step(*args, **kwargs, prefix="test")

    def unpack_batch(self, batch: Union[torch.Tensor, Sequence, Mapping]):
        if isinstance(batch, torch.Tensor):
            x, y = batch, None
        elif isinstance(batch, Sequence):
            x, y = self.unpack_batch_sequence(batch)
        elif isinstance(batch, Mapping):
            x, y = self.unpack_batch_mapping(batch)

        else:
            raise TypeError(
                "Expected Type of sequence or mapping for batch, got {}".format(
                    type(batch).__name__
                )
            )

        return x, y

    @staticmethod
    def unpack_batch_sequence(batch: Sequence) -> Tuple[torch.Tensor, torch.Tensor]:
        return (batch[0], batch[1])

    @staticmethod
    def unpack_batch_mapping(batch: Mapping) -> Tuple[torch.Tensor, torch.Tensor]:
        return (batch["x"], batch["y"])

    @staticmethod
    def compute_dict(
        functions, y_hat: torch.Tensor, y: torch.Tensor, prefix: str = "val", sep="/"
    ) -> dict:
        return {
            f"{prefix}{sep}{name}": func(y_hat, y) for name, func in functions.items()
        }
