# import our libraries
import sys
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning import _logger as log
from pandas import DataFrame

from pl_flash.data import download_data
from pl_flash.text import TextClassificationData, TextClassifier


def train_text_classification(args):

    # download data
    download_data("https://pl-flash-data.s3.amazonaws.com/imdb.zip", 'data/')

    # 1. organize our data
    datamodule = TextClassificationData.from_files(
        backbone="bert-base-cased",
        train_file="data/imdb/train.csv",
        valid_file="data/imdb/valid.csv",
        test_file="data/imdb/test.csv",
        text_field="review",
        label_field="sentiment",
    )

    # 2. build model
    model = TextClassifier(backbone="bert-base-cased", num_classes=2)

    trainer = pl.Trainer(**vars(args))
    trainer.fit(model, datamodule=datamodule)
    results = trainer.test(model, datamodule=datamodule)
    df = DataFrame(results[0]["predictions"])
    log.info(df)


if __name__ == "__main__":
    parser = ArgumentParser(description="Training ImageClassifier on Ants / Bees Dataset")
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # when running `python pl_flash_examples/torchvision_classifier.py`
    if len(sys.argv) == 1:
        args = parser.parse_args("--max_epochs 1 --fast_dev_run 1 --num_processes 2 --accelerator ddp_cpu".split(" "))

    train_text_classification(args)
