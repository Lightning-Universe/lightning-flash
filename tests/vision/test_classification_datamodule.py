from pl_flash.vision import ImageClassificationData
import torch


def _dummy_image_loader(filepath):
    return torch.rand(3, 64, 64)


def test_from_filepaths(tmpdir):
    img_data = ImageClassificationData.from_filepaths(
        train_filepaths=["a", "b"],
        train_labels=[0, 1],
        loader=_dummy_image_loader,
        batch_size=1,
    )

    imgs, labels = next(iter(img_data.train_dataloader()))
    assert imgs.shape == (1, 3, 64, 64)
    assert labels.shape == (1,)

    assert img_data.val_dataloader() is None
    assert img_data.test_dataloader() is None

    img_data = ImageClassificationData.from_filepaths(
        train_filepaths=["a", "b"],
        train_labels=[0, 1],
        valid_filepaths=["c", "d"],
        valid_labels=[0, 1],
        test_filepaths=["e", "f"],
        test_labels=[0, 1],
        loader=_dummy_image_loader,
        batch_size=1,
    )

    imgs, labels = next(iter(img_data.val_dataloader()))
    assert imgs.shape == (1, 3, 64, 64)
    assert labels.shape == (1,)

    imgs, labels = next(iter(img_data.test_dataloader()))
    assert imgs.shape == (1, 3, 64, 64)
    assert labels.shape == (1,)
