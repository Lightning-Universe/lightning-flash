import math
from typing import (
    Any,
    Callable,
    Mapping,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)
from copy import deepcopy
import warnings
import torch
from torch.optim import Optimizer
import torchvision
from pl_flash.core import Flash


class ImageClassifier(Flash):
    """Image classification task

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
        linear_hiddens: If given, it specifies the number of linear layers as well as their hidden dimensions for the
            new classification head. Has no effect for custom models. Defaults to None.

    Raises:
        RuntimeError: If a custom model was provided and we need to extract the classifcation head
            (which is not supported for custom models). Or the last layer to repalce with the correct number of
            classes cannot be inferred.
        ValueError: If a not supported model was requested from torchvision or bolts
        NotImplementedError: When trying to request a model from bolts (which is not yet implemented)
        ImportError: if torchvision is required but not installed

    """

    _available_models_torchvision = (
        "alexnet",
        "vgg11",
        "vgg11_bn",
        "vgg13",
        "vgg13_bn",
        "vgg16",
        "vgg16_bn",
        "vgg19",
        "vgg19_bn",
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
        "squeezenet1_0",
        "squeezenet1_1",
        "densenet121",
        "densenet169",
        "densenet161",
        "densenet201",
        "inception_v3",
        "googlenet",
        "shufflenet_v2_x0_5",
        "shufflenet_v2_x1_0",
        "shufflenet_v2_x1_5",
        "shufflenet_v2_x2_0",
        "mobilenet_v2",
        "resnext50_32x4d",
        "resnext101_32x8d",
        "wide_resnet50_2",
        "wide_resnet101_2",
        "mnasnet0_5",
        "mnasnet0_75",
        "mnasnet1_0",
        "mnasnet1_3",
    )

    _available_models_bolts = ()

    available_models: tuple = tuple(list(_available_models_torchvision) + list(_available_models_bolts))

    def __init__(
        self,
        num_classes: int,
        model: Union[str, torch.nn.Module] = "resnet18",
        loss: Union[Callable, torch.nn.Module, Mapping, Sequence] = "cross_entropy",
        metrics: Union[Callable, torch.nn.Module, Mapping, Sequence, None] = None,
        learning_rate: float = 1e-3,
        optimizer: Union[Type[Optimizer], str] = "Adam",
        pretrained: bool = True,
        in_channels: int = 3,
        linear_hiddens: Optional[Sequence[int]] = None,
        **kwargs,
    ) -> None:

        if isinstance(model, str):
            assert model in self.available_models

        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            learning_rate=learning_rate,
            optimizer=optimizer,
        )

        self.num_classes = num_classes
        self.in_channels = in_channels

        if isinstance(self.model, str) and self.model in self._available_models_torchvision:
            self.model, self.example_input_array = self._model_from_torchvision(model, pretrained=pretrained, **kwargs)
            self._origin = "torchvision"

        elif isinstance(self.model, str) and self.model in self._available_models_bolts:
            self.model, self.example_input_array = self._model_from_bolts(
                model, pretrained=pretrained, num_classes=num_classes, **kwargs
            )
            self._origin = "bolts"

        else:
            self._origin = "custom"

        if self._origin != "custom":

            try:
                import torchvision

                if hasattr(self.model, "AuxLogits") and isinstance(
                    self.model.AuxLogits, torchvision.models.inception.InceptionAux
                ):
                    self.model.AuxLogits = torchvision.models.inception.InceptionAux(
                        self.model.AuxLogits.conv0.conv.in_channels,
                        self.num_classes,
                        conv_block=self.model.AuxLogits.conv0.__class__,
                    )

            except ImportError:
                pass

            if self.in_channels != self.example_input_array.size(1):
                self._repace_input_layer()

            if linear_hiddens is None:
                new_head = self._replace_last_layer_only()
                self.classification_head = new_head
            else:

                new_head = self._compute_new_head_from_hidden_list(linear_hiddens)
                self.classification_head = new_head

    def _determine_classification_head_attr_name(self) -> str:
        """Determines the classification head

        Raises:
            RuntimeError: the model is a custom one.
            ValueError: Unknown model origin

        Returns:
            str: the name of the classification head attribute
        """
        if self._origin == "custom":
            raise RuntimeError("Cannot infer classification head from custom model autmatically")

        elif self._origin == "torchvision":
            return self._determine_classification_head_attr_name_torchvision(self.model)

        elif self._origin == "bolts":
            return self._determine_classification_head_attr_name_bolts(self.model)
        else:
            raise ValueError("Unknown model origin (neither torchvision nor bolts nor custom)")

    @staticmethod
    def _determine_classification_head_attr_name_bolts(model: torch.nn.Module) -> str:
        """Determines the classification head for bolts models

        Args:
            model: the model to extract the head from

        Raises:
            NotImplementedError: currently this is not yet implemented.

        Returns:
            str: the name of the classification head attribute
        """
        raise NotImplementedError

    @staticmethod
    def _determine_classification_head_attr_name_torchvision(
        model: torch.nn.Module,
    ) -> str:
        """Determines the classification head for torchsivion models

        Args:
            model: the model to extract the head from

        Raises:
            ImportError: torchvision is not Installed
            ValueError: Not supported torchvision module
            AssertionError: The model does not have the determined attribute

        Returns:
            str: the name of the classification head attribute
        """
        try:
            import torchvision.models

        except ImportError:
            raise ImportError(
                "Torchvision is not installed please install it following the guides on"
                + "https://pytorch.org/get-started/locally/"
            )

        if isinstance(
            model,
            (
                torchvision.models.AlexNet,
                torchvision.models.VGG,
                torchvision.models.MobileNetV2,
                torchvision.models.MNASNet,
                torchvision.models.SqueezeNet,
                torchvision.models.DenseNet,
            ),
        ):
            clf_head = "classifier"

        elif isinstance(
            model,
            (
                torchvision.models.ResNet,
                torchvision.models.Inception3,
                torchvision.models.GoogLeNet,
                torchvision.models.ShuffleNetV2,
            ),
        ):

            clf_head = "fc"

        else:
            raise ValueError("A not supported model was requested from torchvision")

        assert hasattr(model, clf_head)
        return clf_head

    @property
    def classification_head(self) -> torch.nn.Module:
        """Returns the classification head of the model.

        Raises:
            RuntimeError: the model is a custom one.
            ValueError: Unknown model origin


        Returns:
            torch.nn.Module: the model's classification head
        """
        return getattr(self.model, self._determine_classification_head_attr_name())

    def __setattr__(self, name: str, value: Union[torch.nn.Module, Any]) -> None:
        """Handles the setting of attributes. Especially the updating if the classification head.

        Args:
            name: the attribute Name
            value: the value to update it with

        """
        # can't do this as a setter, since __setattr__ is called before the actual setter
        # and torch.nn.Module filters out all __setattr__ calls that have nn.Modules as values
        if name == "classification_head":
            return setattr(self.model, self._determine_classification_head_attr_name(), value)

        else:
            return super().__setattr__(name, value)

    def _infer_num_features_clf_head_input(self, part_before_new_layer: torch.nn.Module) -> int:
        """Infers the number of input features for the new part of the classification head

        Args:
            part_before_new_layer: the part of the classification head
                that is retained and combined with the new part of the head.

        Returns:
            int: number of input features
        """
        old_head = deepcopy(self.classification_head)

        setattr(
            self,
            "classification_head",
            torch.nn.Sequential(part_before_new_layer, torch.nn.Flatten()),
        )

        with torch.no_grad():
            output = self.model(self.example_input_array)

            if not isinstance(output, torch.Tensor):
                output = output[0]

            num_features = output.size(-1)

        setattr(self, "classification_head", old_head)

        return num_features

    def _repace_input_layer(self) -> None:
        """Determines the classification head

        Raises:
            RuntimeError: the model is a custom one.
            NotImplementedError: the model is from bolts
            ValueError: Unknown model origin

        Returns:
            str: the name of the classification head attribute
        """
        if self._origin == "custom":
            raise RuntimeError("Cannot infer classification head from custom model autmatically")

        elif self._origin == "torchvision":
            return self._replace_input_layer_torchvision(self.in_channels)

        elif self._origin == "bolts":
            raise NotImplementedError("Replacing input layer for bolts models is not yet implemented")
        else:
            raise ValueError("Unknown model origin (neither torchvision nor bolts nor custom)")

    def _replace_input_layer_torchvision(self, in_channels: int) -> int:
        try:
            import torchvision.models

        except ImportError:
            raise ImportError(
                "Torchvision is not installed please install it following the guides on"
                + "https://pytorch.org/get-started/locally/"
            )

        def replace_conv(old_conv: torch.nn.Conv2d):
            return type(old_conv)(
                in_channels=in_channels,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                dilation=old_conv.dilation,
                bias=old_conv.bias is not None,
            )

        if isinstance(
            self.model,
            (
                torchvision.models.AlexNet,
                torchvision.models.VGG,
                torchvision.models.SqueezeNet,
                torchvision.models.DenseNet,
            ),
        ):
            feature_extractor = self.model.features

            if isinstance(feature_extractor, torch.nn.Sequential):
                new_feat_extractor = torch.nn.Sequential(replace_conv(feature_extractor[0]), *feature_extractor[1:])
            else:
                raise TypeError(
                    f"Expected feature extractor to be of type torch.nn.Sequential, "
                    + f"got {type(feature_extractor).__name__}"
                )

            self.model.features = new_feat_extractor

        elif isinstance(self.model, (torchvision.models.ResNet,)):
            if isinstance(self.model.conv1, torch.nn.Conv2d):
                self.model.conv1 = replace_conv(self.model.conv1)
            else:
                raise TypeError(
                    f"Expected conv1 to be of type torch.nn.Conv2d, " + f"got {type(self.model.conv1).__name__}"
                )

        elif isinstance(self.model, torchvision.models.Inception3):
            if isinstance(self.model.conv1.conv, torch.nn.Conv2d):
                self.model.conv1.conv = replace_conv(self.model.conv1.conv)
            else:
                raise TypeError(
                    f"Expected conv1.conv to be of type torch.nn.Conv2d, "
                    + f"got {type(self.model.conv1.conv).__name__}"
                )

        elif isinstance(self.model, torchvision.models.GoogLeNet):
            if isinstance(self.model.conv1.conv, torch.nn.Conv2d):
                self.model.conv1.conv = replace_conv(self.model.conv1.conv)
            else:
                raise TypeError(
                    f"Expected conv1.conv to be of type torch.nn.Conv2d, "
                    + f"got {type(self.model.conv1.conv).__name__}"
                )

        elif isinstance(self.model, torchvision.models.ShuffleNetV2):

            if isinstance(self.model.conv1, torch.nn.Sequential):

                if isinstance(self.model.conv1[0], torch.nn.Conv2d):
                    self.model.conv1 = torch.nn.Sequential(replace_conv(self.model.conv1[0]), *self.model.conv1[1:])
                else:
                    raise TypeError(
                        f"Expected conv1[0] to be of type torch.nn.Conv2d, "
                        + f"got {type(self.model.conv1[0]).__name__}"
                    )
            else:
                raise TypeError(
                    f"Expected conv1 to be of type torch.nn.Sequential, " + f"got {type(self.model.conv1).__name__}"
                )

        elif isinstance(self.model, torchvision.models.MobileNetV2):

            if isinstance(self.model.features, torch.nn.Sequential):

                if isinstance(self.model.features[0], torch.nn.Sequential):

                    if isinstance(self.model.features[0][0], torch.nn.Conv2d):
                        self.model.features = torch.nn.Sequential(
                            torch.nn.Sequential(
                                replace_conv(self.model.features[0][0]),
                                *self.model.features[0][1:],
                            ),
                            self.model.features[1:],
                        )
                    else:
                        raise TypeError(
                            f"Expected features[0][0] to be of type torch.nn.Conv2d, "
                            + f"got {type(self.model.features[0][0]).__name__}"
                        )
                else:
                    raise TypeError(
                        f"Expected features[0] to be of type torch.nn.Sequential, "
                        + f"got {type(self.model.features[0]).__name__}"
                    )

            else:
                raise TypeError(
                    f"Expected features to be of type torch.nn.Sequential, "
                    + f"got {type(self.model.features).__name__}"
                )
        elif isinstance(self.model, torchvision.models.MNASNet):

            if isinstance(self.model.layers, torch.nn.Sequential):

                if isinstance(self.model.layers[0], torch.nn.Conv2d):
                    self.model.layers = torch.nn.Sequential(replace_conv(self.model.layers[0]), *self.model.layers[1:])

                else:
                    raise TypeError(
                        f"Expected layers[0] to be of type torch.nn.Conv2d, "
                        + f"got {type(self.model.layers[0]).__name__}"
                    )

            else:
                raise TypeError(
                    f"Expected layers to be of type torch.nn.Sequential, " + f"got {type(self.model.layers).__name__}"
                )
        else:
            raise ValueError(f"Not supported model type found: {type(self.model).__name__}")

        self.example_input_array = torch.rand(
            self.example_input_array.size(0),
            in_channels,
            *self.example_input_array.shape[2:],
        )

    def _replace_last_layer_only(self) -> Union[torch.nn.Linear, torch.nn.Sequential]:
        """replaces the last layer to support a classification task with a number of class
        different then to 1000 imagenet classes.

        Raises:
            RuntimeError: the last layer to replace cannot be inferred
                (currently only Sequential and linear layers are supported as heads)

        Returns:
            Union[torch.nn.Linear, torch.nn.Sequential]: the new classification head
        """
        head = self.classification_head

        if isinstance(head, torch.nn.Linear):
            return torch.nn.Linear(head.in_features, self.num_classes, bias=head.bias is not None)

        elif isinstance(head, torch.nn.Sequential):
            new_layer = None
            for idx, layer in enumerate(reversed(head)):
                if isinstance(layer, torch.nn.Linear):
                    new_layer = torch.nn.Linear(layer.in_features, self.num_classes, bias=layer.bias is not None)

                elif isinstance(layer, torch.nn.Conv2d):
                    new_layer = torch.nn.Conv2d(
                        layer.in_channels,
                        self.num_classes,
                        kernel_size=layer.kernel_size,
                        padding=layer.padding,
                        stride=layer.stride,
                        dilation=layer.dilation,
                        bias=layer.bias is not None,
                    )

                if new_layer is not None:
                    idx = len(head) - idx - 1
                    before = head[:idx]
                    after = head[idx + 1 :]
                    return torch.nn.Sequential(*before, new_layer, *after)

        else:
            raise RuntimeError("cannot infer last layer to replace")

    def _compute_new_head_from_hidden_list(self, hiddens: Sequence) -> torch.nn.Sequential:
        """Computes the head for a given list of hidden dimensions

        Args:
            hiddens: a list of hidden dimensions.
                The length of the list determines the number of layers.
                If the last element is not equal to the number of classes, an additional layer is added.

        Returns:
            torch.nn.Sequential: the new classification head
        """

        hiddens = list(hiddens)

        if hiddens[-1] != self.num_classes:
            hiddens += [self.num_classes]

        before, after = self._infer_options_before_and_after_new_head()

        curr_in_feats = self._infer_num_features_clf_head_input(before)

        # if before replacing head there was a pooling in after part: remove it and add pooling to before part
        add_pool = False
        new_after = []
        for layer in after:
            if isinstance(layer, torch.nn.AdaptiveAvgPool2d):
                add_pool = True
                continue

            new_after.append(layer)

        after = torch.nn.Sequential(*new_after)

        new_clf = []

        if add_pool:
            _sqrt_feats = int(math.sqrt(curr_in_feats))
            new_clf = new_clf + [torch.nn.AdaptiveAvgPool2d(_sqrt_feats, int(curr_in_feats / _sqrt_feats))]
        new_clf = new_clf + [
            torch.nn.Flatten(),
        ]

        for curr_out_feats in hiddens:
            new_clf += [torch.nn.Linear(curr_in_feats, curr_out_feats), torch.nn.ReLU()]
            curr_in_feats = curr_out_feats

        # remove last relu
        new_clf = new_clf[:-1]

        return torch.nn.Sequential(before, *new_clf, after)

    def _infer_options_before_and_after_new_head(
        self,
    ) -> Tuple[torch.nn.Sequential, torch.nn.Sequential]:
        """Infers which parts of the old classification head to keep and wrap the new head with.

        Returns:
            Tuple[torch.nn.Sequential, torch.nn.Sequential]: the parts before and after the new classification head
        """
        head = self.classification_head

        before, after = [], []

        if isinstance(head, torch.nn.Sequential):

            last_idx = 0

            for idx, layer in enumerate(head):
                if isinstance(layer, (torch.nn.Linear, torch.nn.modules.conv._ConvNd)):
                    break

                before.append(layer)
                last_idx = idx

            for layer in reversed(head[last_idx:]):
                if isinstance(layer, (torch.nn.Linear, torch.nn.modules.conv._ConvNd)):
                    break

                if isinstance(layer, torch.nn.AdaptiveAvgPool2d):
                    continue

                after = [layer] + after

        return torch.nn.Sequential(*before), torch.nn.Sequential(*after)

    @staticmethod
    def _model_from_torchvision(model: str, pretrained: bool, **kwargs) -> Tuple[torch.nn.Module, torch.Tensor]:
        """Gets a (possibly pretrained) model from torchvision

        Args:
            model: the name of the model in torchvision.
            pretrained: if the model should be pretrained

        Raises:
            ImportError: torchvision is not installed
            ValueError: the requested model is not supported

        Returns:
            Tuple[torch.nn.Module, torch.Tensor]: the module and it's corresponding example input array
        """
        try:
            import torchvision.models

        except ImportError:
            raise ImportError(
                "Torchvision is not installed please install it following the guides on "
                + "https://pytorch.org/get-started/locally/"
            )

        try:
            pretrained_model = getattr(torchvision.models, model)(pretrained=pretrained, **kwargs)
        except (ValueError, NotImplementedError):
            warnings.warn(f"{model} is not available as pretrained model. Switching pretrained to False instead!")
            pretrained_model = getattr(torchvision.models, model)(pretrained=False, **kwargs)

        if isinstance(
            pretrained_model,
            (
                torchvision.models.AlexNet,
                torchvision.models.VGG,
                torchvision.models.MobileNetV2,
                torchvision.models.MNASNet,
                torchvision.models.ResNet,
                torchvision.models.ShuffleNetV2,
                torchvision.models.SqueezeNet,
                torchvision.models.DenseNet,
                torchvision.models.GoogLeNet,
            ),
        ):
            example_input_tensor = torch.rand(1, 3, 224, 224)

        elif isinstance(pretrained_model, torchvision.models.Inception3):
            example_input_tensor = torch.rand(2, 3, 299, 299)

        else:
            raise ValueError("A Non supported model was requested from torchvision")

        return pretrained_model, example_input_tensor

    @staticmethod
    def _model_from_bolts(
        model: str, pretrained: bool, num_classes: int, **kwargs
    ) -> Tuple[torch.nn.Module, torch.Tensor]:
        """Gets a model from bolts

        Args:
            model: the name of the model in bolts
            pretrained: whether this model should be pretrained
            num_classes: the number of classes the model should have

        Raises:
            NotImplementedError: This is not yet implemented for bolts

        Returns:
            Tuple[torch.nn.Module, torch.Tensor]: the model and it's corresponding example input
        """
        raise NotImplementedError

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
