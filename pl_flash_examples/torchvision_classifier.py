# import our libraries
import sys
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning import _logger as log
from pandas import DataFrame

from pl_flash.data import download_data
from pl_flash.vision import ImageClassificationData, ImageClassifier


def train_image_classifier(args):

    # download data
    download_data("https://download.pytorch.org/tutorial/hymenoptera_data.zip", 'data/')

    # 1. organize our data
    data = ImageClassificationData.from_folders(
        train_folder="data/hymenoptera_data/train/",
        valid_folder="data/hymenoptera_data/val/",
        test_folder="data/hymenoptera_data/val/",
        num_workers=0
    )

    # 2. build our model
    num_classes = 2
    model = ImageClassifier(backbone="resnet18", num_classes=num_classes)

    trainer = pl.Trainer(**vars(args))
    trainer.fit(model, data)
    results = trainer.test(model, datamodule=data)
    df = DataFrame(results[0]["predictions"])
    log.info(df)


if __name__ == "__main__":
    parser = ArgumentParser(description="Training ImageClassifier on Ants / Bees Dataset")
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # when running `python pl_flash_examples/torchvision_classifier.py`
    if len(sys.argv) == 1:
        args = parser.parse_args("--max_epochs 1 --fast_dev_run 1 --num_processes 2 --accelerator ddp_cpu".split(" "))

    train_image_classifier(args)
