import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Dataset

from flash.vision import ImageDetector


def collate_fn(batch):
    return tuple(zip(*batch))


class DummyDetectionDataset(Dataset):

    def __init__(self, img_shape, num_boxes, num_classes, length):
        super().__init__()
        self.img_shape = img_shape
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.length = length

    def __len__(self):
        return self.length

    def _random_bbox(self):
        c, h, w = self.img_shape
        xs = torch.randint(w - 1, (2, ))
        ys = torch.randint(h - 1, (2, ))
        return [min(xs), min(ys), max(xs) + 1, max(ys) + 1]

    def __getitem__(self, idx):
        img = torch.rand(self.img_shape)
        boxes = torch.tensor([self._random_bbox() for _ in range(self.num_boxes)])
        labels = torch.randint(self.num_classes, (self.num_boxes, ))
        return img, {"boxes": boxes, "labels": labels}


def test_init():
    model = ImageDetector(num_classes=2)
    model.eval()

    batch_size = 2
    ds = DummyDetectionDataset((3, 224, 224), 1, 2, 10)
    dl = DataLoader(ds, collate_fn=collate_fn, batch_size=batch_size)
    img, target = next(iter(dl))

    out = model(img)

    assert len(out) == batch_size
    assert {"boxes", "labels", "scores"} <= out[0].keys()


def test_training(tmpdir):
    model = ImageDetector(num_classes=2, model="fasterrcnn_resnet50_fpn")
    ds = DummyDetectionDataset((3, 224, 224), 1, 2, 10)
    dl = DataLoader(ds, collate_fn=collate_fn)
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    trainer.fit(model, dl)
