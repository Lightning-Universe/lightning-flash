import torch

from flash import Trainer
from flash.vision import ImageClassificationData, ImageClassifier


def _dummy_image_loader(_):
    return torch.rand(3, 224, 224)


def test_classification(tmpdir):
    data = ImageClassificationData.from_filepaths(
        train_filepaths=["a", "b"],
        train_labels=[0, 1],
        train_transform=lambda x: x,
        loader=_dummy_image_loader,
        num_workers=0,
        batch_size=2,
    )
    model = ImageClassifier(2, backbone="resnet18")
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.finetune(model, datamodule=data, strategy="freeze")
