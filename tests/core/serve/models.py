from pathlib import Path

import pytorch_lightning as pl
import torch

from flash.core.serve import expose, ModelComponent
from flash.core.serve.types import Image, Label, Number, Repeated
from flash.core.utilities.imports import _TORCHVISION_AVAILABLE

if _TORCHVISION_AVAILABLE:
    from torchvision.models import squeezenet1_1

CWD = Path(__file__).parent.joinpath("data").absolute()


class LightningSqueezenet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = squeezenet1_1(pretrained=True).eval()

    def forward(self, x):
        return self.model(x)


class LightningSqueezenetServable(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


def _func_from_exposed(arg):
    return ("func", arg)


class ClassificationInference(ModelComponent):
    def __init__(self, model):  # skipcq: PYL-W0621
        self.model = model

    @expose(
        inputs={"img": Image(extension="JPG")},
        outputs={"prediction": Label(path=str(CWD / "imagenet_labels.txt"))},
    )
    def classify(self, img):
        img = img.float() / 255
        mean = torch.tensor([[[0.485, 0.456, 0.406]]]).float()
        std = torch.tensor([[[0.229, 0.224, 0.225]]]).float()
        img = (img - mean) / std
        img = img.permute(0, 3, 2, 1)
        out = self.model(img)

        method_res = self.method_from_exposed(42)
        assert method_res == ("method", 42)
        func_res = _func_from_exposed("DouglasAdams")
        assert func_res == ("func", "DouglasAdams")

        return out.argmax()

    @staticmethod
    def never_should_run():
        raise RuntimeError()

    @staticmethod
    def method_from_exposed(arg):
        return ("method", arg)


try:

    class ClassificationInferenceRepeated(ModelComponent):
        def __init__(self, model):
            self.model = model

        @expose(
            inputs={"img": Repeated(Image(extension="JPG"))},
            outputs={
                "prediction": Repeated(Label(path=str(CWD / "imagenet_labels.txt"))),
                "other": Number(),
            },
        )
        def classify(self, img):
            img = img[0].float() / 255
            mean = torch.tensor([[[0.485, 0.456, 0.406]]]).float()
            std = torch.tensor([[[0.229, 0.224, 0.225]]]).float()
            img = (img - mean) / std
            img = img.permute(0, 3, 2, 1)
            out = self.model(img)
            return ([out.argmax(), out.argmax()], torch.Tensor([21]))


except TypeError:
    ClassificationInferenceRepeated = None

try:

    class ClassificationInferenceModelSequence(ModelComponent):
        def __init__(self, model):
            self.model1 = model[0]
            self.model2 = model[1]

        @expose(
            inputs={"img": Image(extension="JPG")},
            outputs={"prediction": Label(path=str(CWD / "imagenet_labels.txt"))},
        )
        def classify(self, img):
            img = img.float() / 255
            mean = torch.tensor([[[0.485, 0.456, 0.406]]]).float()
            std = torch.tensor([[[0.229, 0.224, 0.225]]]).float()
            img = (img - mean) / std
            img = img.permute(0, 3, 2, 1)
            out = self.model1(img)
            out2 = self.model2(img)
            assert out.argmax() == out2.argmax()
            return out.argmax()


except TypeError:
    ClassificationInferenceRepeated = None

try:

    class ClassificationInferenceModelMapping(ModelComponent):
        def __init__(self, model):
            self.model1 = model["model_one"]
            self.model2 = model["model_two"]

        @expose(
            inputs={"img": Image(extension="JPG")},
            outputs={"prediction": Label(path=str(CWD / "imagenet_labels.txt"))},
        )
        def classify(self, img):
            img = img.float() / 255
            mean = torch.tensor([[[0.485, 0.456, 0.406]]]).float()
            std = torch.tensor([[[0.229, 0.224, 0.225]]]).float()
            img = (img - mean) / std
            img = img.permute(0, 3, 2, 1)
            out = self.model1(img)
            out2 = self.model2(img)
            assert out.argmax() == out2.argmax()
            return out.argmax()


except TypeError:
    ClassificationInferenceModelMapping = None

try:

    class ClassificationInferenceComposable(ModelComponent):
        def __init__(self, model):
            self.model = model

        @expose(
            inputs={
                "img": Image(extension="JPG"),
                "tag": Label(path=str(CWD / "imagenet_labels.txt")),
            },
            outputs={
                "predicted_tag": Label(path=str(CWD / "imagenet_labels.txt")),
                "cropped_img": Image(),
            },
        )
        def classify(self, img, tag):
            im_div = img.float() / 255
            mean = torch.tensor([[[0.485, 0.456, 0.406]]]).float()
            std = torch.tensor([[[0.229, 0.224, 0.225]]]).float()
            img_new = (im_div - torch.mean(mean)) / torch.mean(std)
            img_new = img_new.permute(0, 3, 2, 1)
            out = self.model(img_new)

            return out.argmax(), img


except TypeError:
    ClassificationInferenceComposable = None

try:

    class SeatClassifier(ModelComponent):
        def __init__(self, model, config):
            self.sport = config["sport"]

        @expose(
            inputs={
                "section": Number(),
                "isle": Number(),
                "row": Number(),
                "stadium": Label(path=str(CWD / "imagenet_labels.txt")),
            },
            outputs={
                "seat_number": Number(),
                "team": Label(path=str(CWD / "imagenet_labels.txt")),
            },
        )
        def predict(self, section, isle, row, stadium):
            seat_num = section.item() * isle.item() * row.item() * stadium * len(self.sport)
            stadium_idx = torch.tensor(1000)
            return torch.Tensor([seat_num]), stadium_idx


except TypeError:
    SeatClassifier = None
