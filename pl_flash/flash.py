from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Type, Union

import torch
import torch.optim
from torch.optim import Optimizer

from pytorch_lightning import LightningModule, EvalResult, TrainResult


class Flash(LightningModule):
    """
    A ``Flash`` is a basic definition of a specific task and this is ne most common baseclass providing
    good defaults to most tasks.

    Args:
        model: the model to use inside this task
        loss: the functions to update the model with. All provided losses will be summed up.
            The resulting total loss will also be used for checkpointing and early stopping in during evaluation.
        metrics: The provided metrics. All metrics here will be logged to progress bar and the respective logger.
            Defaults to None.
        learning_rate: The learning rate for the optimizer to use for training. Defaults to 1e-3.
        optimizer: The optimizer to use for training. Can either be the actual class or the class name.
            Defaults to "Adam".

    Raises:
        ValueError: If only names were specified for metrics or functions and they could not be imported automatically
            or if metrics or losses are of invalid type

    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss: Union[Callable, torch.nn.Module, Mapping, Sequence],
        metrics: Union[Callable, torch.nn.Module, Mapping, Sequence, None] = None,
        learning_rate: float = 1e-3,
        optimizer: Union[Type[Optimizer], str] = "Adam",
    ) -> None:
        def format_functions(
            functions: Union[Callable, torch.nn.Module, Mapping, Sequence]
        ) -> Union[torch.nn.ModuleDict, dict]:
            """resolves the actual class or function from given names and makes sure the result is either a dict or a
                Moduledict

            Args:
                functions: the functions to resolve and format

            Returns:
                Union[torch.nn.ModuleDict, dict]: formatted ad resolved function
            """

            def get_name(func) -> str:
                """Gets the correct name of a function/class

                Args:
                    func: the function/class to get the name from

                Returns:
                    str: the extracted name
                """
                try:
                    return func.__name__
                except AttributeError:
                    return func.__class__.__name__

            def resolve_str(item: Union[Any, str]) -> Union[Callable, torch.nn.Module]:
                """Gets the actual function or Module from torch that matches the given name

                Args:
                    item: the name corresponding to the actual function/module or an actual function/module (will be
                        returned as is)

                Raises:
                    ValueError: the provided item is a string but does not correspond to a class/function
                        in ``torch.nn`` or ``torch.nn.functional``

                Returns:
                    Union[Callable, torch.nn.Module]: the resolved function module
                """
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

            if isinstance(functions, Mapping) and not isinstance(functions, torch.nn.ModuleDict):
                functions = {k: resolve_str(v) for k, v in functions.items()}
                if any([isinstance(func, torch.nn.Module) for func in functions.values()]):
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
        """Forwards all inputs to the model

        Returns:
            Any: the models outputs
        """
        return self.model(*args, **kwargs)

    def configure_optimizers(self) -> Optimizer:
        """Defines the optimizer based on :attr:`optimizer_cls` and :attr:`learning_rate`

        Returns:
            Optimizer: the constructed optimizer
        """
        return self.optimizer_cls(self.parameters(), lr=self.learning_rate)

    def compute_loss(
        self,
        y_hat: torch.Tensor,
        y: torch.tensor,
        prefix: str = "train",
        sep: str = "/",
    ) -> Tuple[torch.Tensor, dict]:
        """Computes the loss functions and the resulting overall loss

        Args:
            y_hat: the networks predictions
            y: the groundtruth
            prefix: the prefix indicating whether the current prediction is from training, evaluation or testing.
                Defaults to "train".
            sep: the separator between the actual prefix and the loss name. Defaults to "/".

        Returns:
            Tuple[torch.Tensor, dict]: the dictionary containing the partial loss values and the tensor
                 containing the overall value
        """
        loss_dict = self.compute_dict(self.losses, y_hat, y, prefix, sep)
        return sum(loss_dict.values()), loss_dict

    def compute_metrics(self, y_hat: torch.Tensor, y: torch.tensor, prefix: str = "val", sep: str = "/") -> dict:
        """Computes the metric functions

        Args:
            y_hat: the networks predictions
            y: the groundtruth
            prefix: the prefix indicating whether the current prediction is from training, evaluation
                or testing. Defaults to "val".
            sep: the separator between the actual prefix and the loss name. Defaults to "/".

        Returns:
            dict: the dictionary containing the partial metric values
        """
        return self.compute_dict(self.metrics, y_hat, y, prefix, sep)

    def _step(self, batch: Union[Sequence, Mapping], batch_idx: int, prefix: str) -> Union[EvalResult, TrainResult]:
        """Protoype Step for training, validation and testing

        Args:
            batch: a sequence or mapping containing the actual batch elements (input and groundtruth)
            batch_idx: the index of the currently processed batch
            prefix: the prefix, whether the current step is in train, validation or test phase

        Raises:
            ValueError: if prefix is neither train, nor val or test

        Returns:
            Union[EvalResult, TrainResult]: the step result containing all losses and metrics
        """

        x, y = self.unpack_batch(batch)
        y_hat = self(x)

        if prefix != "test":
            assert y is not None

        if y is not None:
            loss, loss_dict = self.compute_loss(y_hat, y, prefix=prefix)
        else:
            loss, loss_dict = None, {}

        if prefix == "train":
            result = TrainResult(loss, early_stop_on=loss, checkpoint_on=loss)
        elif prefix == "val":
            result = EvalResult(early_stop_on=loss, checkpoint_on=loss)
        elif prefix == "test":
            result = EvalResult()
        else:
            raise ValueError("Expected prefix to be one of {{train, val, test}} but got {}".format(prefix))

        for k, v in loss_dict.items():
            result.log(k, v)

        if loss is not None:
            result.log(f"{prefix}/loss", loss, prog_bar=True)

        if y is not None:
            for k, v in self.compute_metrics(y_hat, y, prefix).items():
                result.log(k, v, prog_bar=True)

        return result

    def training_step(self, *args, **kwargs) -> Union[int, TrainResult]:
        """The training step.
        All inputs are directly forwarded to :attr`_step` with the train prefix

        Returns:
            Union[int, TrainResult]: the training step result
        """
        return self._step(*args, **kwargs, prefix="train")

    def validation_step(self, *args, **kwargs) -> EvalResult:
        """The validation step.
        All inputs are directly forwarded to :attr`_step` with the val prefix

        Returns:
            Union[EvalResult]: the validation step result
        """
        return self._step(*args, **kwargs, prefix="val")

    def test_step(self, *args, **kwargs) -> EvalResult:
        """The test step.
        All inputs are directly forwarded to :attr`_step` with the test prefix

        Returns:
            Union[EvalResult]: the test step result
        """
        return self._step(*args, **kwargs, prefix="test")

    def unpack_batch(
        self, batch: Union[torch.Tensor, Sequence, Mapping]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Unpacks a batch from different types into a tuple of input and groundtruth tensor

        Args:
            batch: the batch to extract the tuple from

        Raises:
            TypeError: if the provided type is not supported
                (currently only tensors, Sequences and mappings are supported)

        Returns:
            Tuple[torch.Tensor,Optional[torch.Tensor]]: the input tensor and an optional target tensor
        """
        if isinstance(batch, torch.Tensor):
            x, y = batch, None
        elif isinstance(batch, Sequence):
            x, y = self.unpack_batch_sequence(batch)
        elif isinstance(batch, Mapping):
            x, y = self.unpack_batch_mapping(batch)

        else:
            raise TypeError("Expected Type of sequence or mapping for batch, got {}".format(type(batch).__name__))

        return x, y

    @staticmethod
    def unpack_batch_sequence(batch: Sequence) -> Tuple[torch.Tensor, torch.Tensor]:
        """unpacks a sequence of batch items to a tuple of tensors for input and groundtruth

        Args:
            batch: the sequence of batch items to unpack

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: the tuple of input and groundtruth tensors
        """

        return (batch[0], batch[1])

    @staticmethod
    def unpack_batch_mapping(batch: Mapping) -> Tuple[torch.Tensor, torch.Tensor]:
        """unpacks a mapping of batch items to a tuple of tensors for input and groundtruth

        Args:
            batch: the mapping of batch items to unpack

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: the tuple of input and groundtruth tensors
        """
        return (batch["x"], batch["y"])

    @staticmethod
    def compute_dict(
        functions: Mapping,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        prefix: str = "val",
        sep="/",
    ) -> dict:
        """computes values from a dict of functions based on prediction and groundtruth (typically losses and/or
            metrics)

        Args:
            functions: the dict of functions to compute
            y_hat: the prediction tensor
            y: the target tensor
            prefix: the prefix to add to names. Defaults to "val".
            sep: the separator between prefix and names. Necessary for loggers. Defaults to "/".

        Returns:
            dict: the dictionary of calculated values. For each item in :attr:`functions`
                it will have one entry with key prefix + sep + name and the value to be the
                output from the corresponding function
        """
        return {f"{prefix}{sep}{name}": func(y_hat, y) for name, func in functions.items()}
