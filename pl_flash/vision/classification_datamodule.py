from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule

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
    def __init__(
        self,
        train_ds: Dataset,
        valid_ds: Dataset = None,
        test_ds: Dataset = None,
        batch_size=64,
        num_workers=4,
    ):
        super().__init__()
        self.train_ds = train_ds
        self.valid_ds = valid_ds
        self.test_ds = test_ds

        if self.valid_ds is not None:
            self.val_dataloader = self._val_dataloader

        if self.test_ds is not None:
            self.test_dataloader = self._test_dataloader

        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def _val_dataloader(self):
        return DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def _test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
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
        # TODO: DOCSTRING
        train_ds = _FilepathDataset(
            train_filepaths, train_labels, train_transform, loader
        )
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
