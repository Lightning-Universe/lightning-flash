from typing import Callable, Mapping, Optional, Sequence, Tuple, Type, Union
import warnings

import torch
from torch.optim import Optimizer

from pytorch_lightning.core.step_result import EvalResult, TrainResult

from pl_flash.core import Flash


class SemanticSegmenter(Flash):
    """Semantic Image segmentation task

    Args:
        num_classes: the number of classes in the current task
        model: either a string of :attr`available_models`  or a custom nn.Module. Defaults to 'resnet18'.
        loss: the functions to update the model with. All provided losses will be summed up.
            The resulting total loss will also be used for checkpointing and early stopping in during evaluation.
        metrics: The provided metrics. All metrics here will be logged to progress bar and the respective logger. 
            Defaults to None.
        learning_rate: The learning rate for the optimizer to use for training. Defaults to 1e-3.
        optimizer: The optimizer to use for training. Can either be the actual class or the class name. 
            Defaults to Adam.
        pretrained: Whether the model from torchvision or bolts should be loaded with it's pretrained weights. 
            Has no effect for custom models. Defaults to True.
        in_channels: If your images have a different number of channels, this replaces the first layer by a 
            non-pretrained one with the corresponding number of channels.

    Raises:
        RuntimeError: If a custom model was provided and we need to extract the classifcation head 
            (which is not supported for custom models). Or the last layer to repalce with the correct number of 
            classes cannot be inferred.
        ValueError: If a not supported model was requested from torchvision or bolts
        NotImplementedError: When trying to request a model from bolts (which is not yet implemented)
        ImportError: if torchvision is required but not installed

    """

    _available_models_torchvision = (
        "fcn_resnet50",
        "fcn_resnet101",
        "deeplabv3_resnet50",
        "deeplabv3_resnet101",
    )

    _available_models_bolts = ()

    available_models: tuple = tuple(list(_available_models_torchvision) + list(_available_models_bolts))

    def __init__(
        self,
        num_classes: int,
        model: Union[str, torch.nn.Module] = "fcn_resnet50",
        loss: Union[Callable, torch.nn.Module, Mapping, Sequence] = "cross_entropy",
        metrics: Union[Callable, torch.nn.Module, Mapping, Sequence, None] = None,
        learning_rate: float = 1e-3,
        optimizer: Union[Type[Optimizer], str] = "Adam",
        pretrained: bool = True,
        in_channels: int = 3,
        **kwargs,
    ) -> None:

        if isinstance(model, str):
            assert model in self.available_models

        super().__init__(
            model=model, loss=loss, metrics=metrics, learning_rate=learning_rate, optimizer=optimizer,
        )

        self.num_classes = num_classes
        self.in_channels = in_channels
        self._num_default_classes = 21  # 21 classes from coco segmentation (20 + background)

        if isinstance(self.model, str) and self.model in self._available_models_torchvision:
            self.model, self.example_input_array = self._model_from_torchvision(model, pretrained=pretrained, **kwargs)
            self._origin = "torchvision"

        elif isinstance(self.model, str) and self.model in self._available_models_bolts:
            self.model, self.example_input_array = self._model_from_bolts(model, pretrained=pretrained, **kwargs)
            self._origin = "bolts"

        else:
            self._origin = "custom"

        if self._origin != "custom":

            if self.in_channels != self.example_input_array.size(1):
                self._replace_input_layer()

            if self.num_classes != self._num_default_classes:
                self._replace_output_layer()

    @staticmethod
    def _model_from_torchvision(model: str, pretrained: bool, **kwargs) -> Tuple[torch.nn.Module, torch.Tensor]:
        """Retrieves a (optionally pretrained) segmentation model from torchvision

        Args:
            model: the name of the model to retrieve from torchvision.models.segmentation.
            pretrained: Whether the model should also include pretrained weights

        Raises:
            ImportError: if torchvision was not correctly installed

        Returns:
            Tuple[torch.nn.Module, torch.Tensor]: the retrieved model as well as an example input tensor
        """
        try:
            import torchvision.models.segmentation

        except ImportError:
            raise ImportError(
                "Torchvision is not installed please install it following the guides on"
                + "https://pytorch.org/get-started/locally/"
            )

        try:
            pretrained_model = getattr(torchvision.models.segmentation, model)(pretrained=pretrained, **kwargs)
        except (ValueError, NotImplementedError):
            warnings.warn(f"{model} is not available as pretrained model. Switching pretrained to False instead!")
            pretrained_model = getattr(torchvision.models.segmentation, model)(pretrained=False, **kwargs)

        return pretrained_model, torch.rand(2, 3, 224, 224)

    @staticmethod
    def _model_from_bolts(model: str, pretrained: bool, **kwargs) -> Tuple[torch.nn.Module, torch.Tensor]:
        """Retrieves a (optionally pretrained) segmentation model from bolts

        Args:
            model: the name of the model to retrieve from torchvision.models.segmentation.
            pretrained: Whether the model should also include pretrained weights
        Raises:
            NotImplementedError: this is not yet implemented for bolts

        Returns:
            Tuple[torch.nn.Module, torch.Tensor]: the retrieved model as well as an example input tensor
        """
        raise NotImplementedError

    def _replace_input_layer(self) -> None:
        """replaces input layers of models to enable different number of channels

        Raises:
            RuntimeError: custom model was provided, so we cannot infere those things automatically
            ValueError: Unknown model origin

        """

        if self._origin == "custom":
            raise RuntimeError("Cannot infer classification head from custom model autmatically")

        elif self._origin == "torchvision":
            return self._replace_input_layer_torchvision()

        elif self._origin == "bolts":
            self._replace_input_layer_bolts()
        else:
            raise ValueError("Unknown model origin (neither torchvision nor bolts nor custom)")

    def _replace_input_layer_torchvision(self) -> None:
        """replaces the input layer for torchvision models to support different numbers of input channels

        Raises:
            ImportError: torchvision is not (correctly) installed
            TypeError: model, submodels or layers do not match the expected types
        """
        try:
            import torchvision.models.segmentation

        except ImportError:
            raise ImportError(
                "Torchvision is not installed please install it following the guides on"
                + "https://pytorch.org/get-started/locally/"
            )

        if not isinstance(
            self.model, (torchvision.models.segmentation.FCN, torchvision.models.segmentation.DeepLabV3,),
        ):
            raise TypeError(f"Expected model of types FCN or DeepLabV3. Got {type(self.model).__name__}")

        if not isinstance(self.model.backbone, torchvision.models._utils.IntermediateLayerGetter):
            raise TypeError(
                f"Expected backbone to be of type IntermediateLayerGetter. Got {type(self.model.backbone).__name__}"
            )

        old_conv = self.model.backbone.conv1

        if not isinstance(old_conv, torch.nn.Conv2d):
            raise TypeError(
                f"Expected first conv of backbone to be of type torch.nn.Conv2d. "
                + f"Got {type(self.model.backbone.conv1).__name__}"
            )
        self.model.backbone.conv1 = self._replace_conv2d(old_conv, in_channels=self.in_channels)

        self.example_input_array = torch.rand(2, self.in_channels, *self.example_input_array.shape[2:])

    def _replace_input_layer_bolts(self) -> None:
        """replaces the input layer for bolts models to support different numbers of input channels.

        Raises:
            NotImplementedError: This is not yet implemented for bolts models
        """
        raise NotImplementedError("Replacing input layer for bolts models is not yet implemented")

    def _replace_output_layer(self) -> None:
        """Replaces the output layer to support different numbers of classes

        Raises:
            RuntimeError: the specified model is a custom one. We cannot infer which layer to change automatically
            ValueError: the model origin is unknown

        """
        if self._origin == "custom":
            raise RuntimeError("Cannot infer classification head from custom model autmatically")

        elif self._origin == "torchvision":
            return self._replace_output_layer_torchvision()

        elif self._origin == "bolts":
            self._replace_output_layer_bolts()
        else:
            raise ValueError("Unknown model origin (neither torchvision nor bolts nor custom)")

    @staticmethod
    def _replace_conv2d(
        old_conv: torch.nn.Conv2d, in_channels: Optional[int] = None, out_channels: Optional[int] = None,
    ) -> torch.nn.Conv2d:
        """replaces a convolution with same parameters except intput and output channels (if specified)

        Args:
            old_conv: the convolution to take all non-specified parameters from
            in_channels: the number of input channels. 
                Defaults to None, which takes the parameter from :attr:`old_conv`.
            out_channels: the number of output channels. 
                Defaults to None, which takes the parameter from :attr:`old_conv`.

        Returns:
            torch.nn.Conv2d: the new convolutional layer
        """

        return torch.nn.Conv2d(
            old_conv.in_channels if in_channels is None else in_channels,
            old_conv.out_channels if out_channels is None else out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            bias=old_conv.bias is not None,
        )

    def _replace_head_torchvision(self, old_head: torch.nn.Sequential) -> torch.nn.Sequential:
        try:
            import torchvision.models.segmentation

        except ImportError:
            raise ImportError(
                "Torchvision is not installed please install it following the guides on"
                + "https://pytorch.org/get-started/locally/"
            )
        if isinstance(old_head, torchvision.models.segmentation.deeplabv3.DeepLabHead):
            # DeepLabHead -> ASPP -> ModuleList -> Sequential -> Conv
            return torchvision.models.segmentation.deeplabv3.DeepLabHead(
                in_channels=old_head[0].convs[0][0].in_channels, num_classes=self.num_classes,
            )

        elif isinstance(old_head, torchvision.models.segmentation.fcn.FCNHead):
            return torchvision.models.segmentation.fcn.FCNHead(
                in_channels=old_head[0].in_channels, channels=self.num_classes
            )

        else:
            raise TypeError(f"Expected classifier of type FCNHead or DeepLabHead. Got {type(self.model).__name__}")

    def _replace_output_layer_torchvision(self) -> None:
        """replaces the outputlayer for torchvision models to support a different number of classes

        Raises:
            ImportError: torchvision was not (correctly) installed
            TypeError: the model, a sub model or a layer don't match the expected type
        """
        try:
            import torchvision.models.segmentation

        except ImportError:
            raise ImportError(
                "Torchvision is not installed please install it following the guides on"
                + "https://pytorch.org/get-started/locally/"
            )

        if not isinstance(
            self.model, (torchvision.models.segmentation.FCN, torchvision.models.segmentation.DeepLabV3,),
        ):
            raise TypeError(f"Expected model of types FCN or DeepLabV3. Got {type(self.model).__name__}")

        self.model.classifier = self._replace_head_torchvision(self.model.classifier)

        if hasattr(self.model, "aux_classifier") and self.model.aux_classifier is not None:
            self.model.aux_classifier = self._replace_head_torchvision(self.model.aux_classifier)

    def _replace_output_layer_bolts(self) -> None:
        """Replaces the output layer for bolts models to support different number of classes

        Raises:
            NotImplementedError: not yet implemented for bolts models
        """
        raise NotImplementedError("Replacing output layer for bolts models is not yet implemented")

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
            if isinstance(y_hat, torch.Tensor):
                loss, loss_dict = self.compute_loss(y_hat, y, prefix=prefix)
            elif isinstance(y_hat, dict):
                loss, loss_dict = self.compute_loss(y_hat["out"], y, prefix=prefix)

                if "aux" in y_hat:
                    loss_aux, loss_dict_aux = self.compute_loss(y_hat["aux"], y, prefix=prefix)

                    loss = loss + loss_aux
                    loss_dict.update({k + "_aux": v for k, v in loss_dict_aux.items()})
            else:
                raise TypeError
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
            result.log(k, v, on_epoch=True, on_step=True, prog_bar=True)

        if loss is not None:
            result.log(f"{prefix}/loss", loss, on_epoch=True, on_step=True)

        if y is not None:
            if isinstance(y_hat, torch.Tensor):
                for k, v in self.compute_metrics(y_hat, y, prefix).items():
                    result.log(k, v, on_epoch=True, on_step=True, prog_bar=True)

            elif isinstance(y_hat, dict):

                for k, v in self.compute_metrics(y_hat["out"], y, prefix).items():
                    result.log(k, v, on_epoch=True, on_step=True, prog_bar=True)

                if "aux" in y_hat:
                    loss_aux, loss_dict_aux = self.compute_loss(y_hat["aux"], y, prefix=prefix)

                    for k, v in self.compute_metrics(y_hat["aux"], y, prefix).items():
                        result.log(k + "_aux", v, on_epoch=True, on_step=True, prog_bar=True)
            else:
                raise TypeError

        return result

    @property
    def default_input_norm(self) -> object:
        """A default imagenet normalization

        Raises:
            ImportError: if torchvision is not installed

        Returns:
            object:A tensor normalization
        """
        try:
            from torchvision.transforms import Normalize

            return Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        except ImportError:
            raise ImportError(
                "Torchvision is not installed please install it following the guides on"
                + "https://pytorch.org/get-started/locally/"
            )
