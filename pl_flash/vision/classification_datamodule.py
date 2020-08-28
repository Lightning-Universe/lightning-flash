from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule

from torchvision.datasets import ImageFolder
from PIL import Image


def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class _FilepathDataset(Dataset):
    def __init__(self, filepaths, labels, transform=None, loader=pil_loader):
        self.fnames = filepaths
        self.labels = labels
        self.transform = transform
        self.loader = loader

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        img = self.loader(self.fnames[index])
        if self.transform:
            img = self.transform(img)
        return img, self.labels[index]


class ImageClassificationData(LightningDataModule):
    """Data module for image classification tasks."""

    def __init__(
        self,
        train_ds: Dataset,
        valid_ds: Dataset = None,
        test_ds: Dataset = None,
        batch_size=64,
        num_workers=4,
    ):
        """
        Initialize ImageClassificationData

        Args:
            train_ds: Dataset for training.
            valid_ds (optional): Dataset for validation.
            test_ds (optional): Dataset for testing.
            batch_size (optional): Batch size for data loading.
            num_workers (optional): Number of workers for dataloaders, see
                https://pytorch.org/docs/stable/data.html#multi-process-data-loading
        """
        super().__init__()
        self._train_ds = train_ds
        self._valid_ds = valid_ds
        self._test_ds = test_ds

        if self._valid_ds is not None:
            self.val_dataloader = self._val_dataloader

        if self._test_ds is not None:
            self.test_dataloader = self._test_dataloader

        self._batch_size = batch_size
        self._num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self._train_ds,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=True,
        )

    def _val_dataloader(self):
        return DataLoader(
            self._valid_ds,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=True,
        )

    def _test_dataloader(self):
        return DataLoader(
            self._test_ds,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            pin_memory=True,
        )

    @classmethod
    def from_filepaths(
        cls,
        train_filepaths,
        train_labels,
        train_transform=None,
        valid_filepaths=None,
        valid_labels=None,
        valid_transform=None,
        test_filepaths=None,
        test_labels=None,
        loader=pil_loader,
        batch_size=64,
    ):
        """
        Creates a ImageClassificationData object from lists of image filepaths and labels

        Args:
            train_filepaths (list): list of image filepaths for training set.
            train_labels (list): List of labels for training set.
            train_transform (callable, optional): Image transform to use for training set.
            valid_filepaths (list, optional): List of image filepaths for validation set.
            valid_labels (list, optional): List of labels for validation set.
            valid_transform (callable, optional): Image transform to use for validation and test set.
            test_filepaths (list, optional): List of image filepaths for test set.
            test_labels (list, optional): List of labels for test set.
            loader (callbable, optional): A function to load an image given its path.
            batch_size (optional): Batch size for data loading.

        Examples::
            >>> img_data = ImageClassificationData.from_filepaths(["a.png", "b.png"], [0, 1])

        """

        train_ds = _FilepathDataset(train_filepaths, train_labels, train_transform, loader)
        valid_ds = (
            _FilepathDataset(valid_filepaths, valid_labels, valid_transform, loader)
            if valid_filepaths is not None
            else None
        )

        test_ds = (
            _FilepathDataset(test_filepaths, test_labels, valid_transform, loader)
            if test_filepaths is not None
            else None
        )

        return cls(train_ds, valid_ds, test_ds, batch_size)

    @classmethod
    def from_folders(
        cls,
        train_folder,
        train_transform=None,
        valid_folder=None,
        valid_transform=None,
        test_folder=None,
        loader=pil_loader,
        batch_size=64,
    ):
        """
        Creates a ImageClassificationData object from folders of images arranged in this way:

        ::
        train/dog/xxx.png
        train/dog/xxy.png
        train/dog/xxz.png

        train/cat/123.png
        train/cat/nsdf3.png
        train/cat/asd932{_}.png

        Args:
            train_folder (path): Path to training folder.
            train_transform (callable, optional): Image transform to use for training set.
            valid_folder (path, optional): Path to validation folder.
            valid_transform (callable, optional): Image transform to use for validation and test set.
            test_folder (path, optional): Path to test folder.
            loader (callbable, optional): A function to load an image given its path.
            batch_size (optional): Batch size for data loading.

        Examples::
            >>> img_data = ImageClassificationData.from_folders("train/") # doctest: +SKIP

        """

        train_ds = ImageFolder(train_folder, train_transform, loader=loader)
        valid_ds = ImageFolder(valid_folder, valid_transform, loader=loader) if valid_folder is not None else None

        test_ds = ImageFolder(test_folder, valid_transform, loader=loader) if test_folder is not None else None

        return cls(train_ds, valid_ds, test_ds, batch_size)
