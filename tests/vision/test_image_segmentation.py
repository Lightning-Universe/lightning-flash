import pytest

import torch
from torch.utils.data import DataLoader
import torchvision

from pl_flash.vision.image_segmentation import SemanticSegmenter
from pl_flash import Trainer


class DummySegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, shape: tuple, num_classes: int, length: int):

        super().__init__()

        self.shape = shape
        self.num_classes = num_classes
        self.length = length

    def __getitem__(self, index: int):
        return torch.rand(self.shape), torch.randint(self.num_classes, size=self.shape[1:])

    def __len__(self):
        return self.length


class DummySegmentationModule(torch.nn.Module):
    def __init__(self, aux: bool, num_classes: int):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, num_classes, 1)
        self.aux = aux

    def forward(self, *args, **kwargs):
        x = self.conv(*args, **kwargs)
        out = {"out": x}

        if self.aux:
            out["aux"] = x

        return out


@pytest.mark.parametrize("model", SemanticSegmenter.available_models)
def test_models_num_classes(model: str):

    segmentation_model = SemanticSegmenter(num_classes=10, model=model)

    with torch.no_grad():
        output = segmentation_model(segmentation_model.example_input_array)

    if isinstance(output, dict):
        for _output in output.values():
            assert _output.size(1) == 10
    else:
        assert output.size(1) == 10


def test_model_training_tensor_output(tmpdir):
    data = DataLoader(DummySegmentationDataset((3, 224, 224), 10, 500), batch_size=64, shuffle=True,)

    model = SemanticSegmenter(num_classes=10, model=torch.nn.Sequential(torch.nn.Conv2d(3, 10, kernel_size=1),),)

    Trainer(fast_dev_run=True, default_root_dir=tmpdir, max_steps=2).fit(model, data)


@pytest.mark.parametrize("aux", [True, False])
def test_model_training_dict_output(tmpdir, aux: bool):
    data = DataLoader(DummySegmentationDataset((3, 224, 224), 10, 500), batch_size=64, shuffle=True,)

    model = SemanticSegmenter(10, model=DummySegmentationModule(aux=aux, num_classes=10))

    Trainer(fast_dev_run=True, default_root_dir=tmpdir, max_steps=2).fit(model, data)


def test_from_bolts():
    with pytest.raises(NotImplementedError):
        SemanticSegmenter._model_from_bolts("abc", True)


def test_normalize():
    assert isinstance(SemanticSegmenter(10).default_input_norm, torchvision.transforms.Normalize)


@pytest.mark.parametrize("model", SemanticSegmenter.available_models)
def test_models_in_channels(model: str):

    segmentation_model = SemanticSegmenter(num_classes=10, in_channels=1)

    assert segmentation_model.example_input_array.size(1) == 1

    with torch.no_grad():
        output = segmentation_model(segmentation_model.example_input_array)

    if isinstance(output, dict):
        for _output in output.values():
            assert _output.size(1) == 10
    else:
        assert output.size(1) == 10
