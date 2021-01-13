import sys
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning import _logger as log
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from pl_flash import ClassificationLightningTask


def train_generic_task_on_mnist(args):

    # model
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    # data
    dataset = datasets.MNIST('./data_folder', download=True, transform=transforms.ToTensor())
    train, val, test = random_split(dataset, [50000, 5000, 5000])

    # task
    classifier = ClassificationLightningTask(model, loss_fn=nn.functional.cross_entropy, optimizer=optim.Adam)

    # train
    trainer = pl.Trainer(**vars(args))
    trainer.fit(classifier, DataLoader(train), DataLoader(val))
    results = trainer.test(classifier, test_dataloaders=DataLoader(test))
    log.info(results)


if __name__ == "__main__":
    parser = ArgumentParser(description="")
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # when running `python pl_flash_examples/torchvision_classifier.py`
    if len(sys.argv) == 1:
        args = parser.parse_args("--max_epochs 1 --fast_dev_run 1 --num_processes 2 --accelerator ddp_cpu".split(" "))

    train_generic_task_on_mnist(args)
