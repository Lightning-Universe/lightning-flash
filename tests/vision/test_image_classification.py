from collections import namedtuple
import pytest

import torch
from torch.utils.data import DataLoader
import torchvision

from pl_flash.vision.image_classification import ImageClassifier
from pl_flash import Trainer

from tests.core.test_flash import DummyDataset


@pytest.mark.parametrize("model", ImageClassifier.available_models)
def test_models_num_classes(model: str):

    classification_model = ImageClassifier(num_classes=10, model=model)

    with torch.no_grad():
        output = classification_model(classification_model.example_input_array)

    if isinstance(output, torchvision.models.inception.InceptionOutputs):
        for _output in output:
            assert _output.size(-1) == 10
    else:
        assert output.size(-1) == 10


def test_model_training(tmpdir):
    data = DataLoader(DummyDataset((3, 224, 224), 10, 500), batch_size=64, shuffle=True,)

    model = ImageClassifier(
        num_classes=10,
        model=torch.nn.Sequential(
            torch.nn.Conv2d(3, 1, kernel_size=1),
            torch.nn.AdaptiveAvgPool2d(32),
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 32, 10),
        ),
    )

    Trainer(fast_dev_run=True, default_root_dir=tmpdir, max_steps=2).fit(model, data)


@pytest.mark.parametrize("model", ImageClassifier.available_models)
def test_classification_head_attr_name(model):
    model = ImageClassifier(num_classes=10)

    assert model._determine_classification_head_attr_name() in ["fc", "classifier"]


@pytest.mark.parametrize("model", ImageClassifier.available_models)
def test_models_different_head(model: str):

    classification_model = ImageClassifier(num_classes=10, model=model, linear_hiddens=[1, 2, 3])

    with torch.no_grad():
        output = classification_model(classification_model.example_input_array)

    if isinstance(output, torchvision.models.inception.InceptionOutputs):
        for _output in output:
            assert _output.size(-1) == 10
    else:
        assert output.size(-1) == 10

    head = classification_model.classification_head
    assert isinstance(head, torch.nn.Sequential)

    for in_feats, out_feats in [(1, 2), (2, 3), (3, 10)]:
        assert any(
            [
                isinstance(layer, torch.nn.Linear) and layer.in_features == in_feats and layer.out_features == out_feats
                for layer in head
            ]
        )


def test_determine_head_error():
    classification_model = ImageClassifier(num_classes=10, model="resnet18")

    classification_model._origin = "custom"
    with pytest.raises(RuntimeError):
        classification_model._determine_classification_head_attr_name()

    classification_model._origin = "abcdefgtest123"

    with pytest.raises(ValueError):
        classification_model._determine_classification_head_attr_name()

    classification_model._origin = "bolts"
    with pytest.raises(NotImplementedError):
        classification_model._determine_classification_head_attr_name()

    classification_model._origin = "torchvision"
    classification_model.model = namedtuple("name", "a")(5.0)

    with pytest.raises(ValueError):
        classification_model._determine_classification_head_attr_name()


def test_determine_head_error():
    classification_model = ImageClassifier(num_classes=10, model="resnet18")

    setattr(classification_model, "classification_head", torch.nn.Flatten())

    with pytest.raises(RuntimeError):
        classification_model._replace_last_layer_only()


def test_from_bolts():
    with pytest.raises(NotImplementedError):
        ImageClassifier._model_from_bolts("abc", True, 3)


def test_normalize():
    assert isinstance(ImageClassifier(10).default_input_norm, torchvision.transforms.Normalize)
