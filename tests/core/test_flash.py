from pl_flash import Flash
from pytorch_lightning.metrics import functional as FM

import torch
import torch.nn.functional as F

import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as T


def test_init():
    mlp = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10), nn.LogSoftmax())

    model = Flash(mlp, loss=F.nll_loss)


def test_train(tmpdir):
    data = DataLoader(MNIST(tmpdir, download=True, transform=T.ToTensor()), batch_size=64, shuffle=True,)
    mlp = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 10), nn.LogSoftmax(),)
    model = Flash(mlp, loss=F.nll_loss, metrics=[FM.accuracy])
    model.fit(data, fast_dev_run=True, default_root_dir=tmpdir, max_steps=2)
