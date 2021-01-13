# import our libraries
import sys
from argparse import ArgumentParser

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import _logger as log
from pandas import DataFrame
from pytorch_lightning import LightningDataModule, LightningModule
from sklearn.model_selection import train_test_split

from pl_flash.data import download_data
from pl_flash.tabular import TabularClassifier, TabularData


def train_tabular_data(args):

    # download data
    download_data("https://pl-flash-data.s3.amazonaws.com/titanic.csv", "titanic.csv")

    data = pd.read_csv("titanic.csv")
    train_df, valid_df = train_test_split(data, test_size=0.5)
    valid_df, test_df = train_test_split(valid_df, test_size=0.5)

    # 1. organize our data
    datamodule: TabularData = TabularData.from_df(
        train_df,
        categorical_cols=["Sex", "Age", "SibSp", "Parch", "Ticket", "Cabin", "Embarked"],
        numerical_cols=["Fare"],
        target_col="Survived",
        num_workers=0,
        batch_size=8,
        valid_df=valid_df,
        test_df=test_df,
    )

    # 2. build model
    model: LightningModule = TabularClassifier(
        num_classes=2,
        num_columns=8,
        embedding_sizes=datamodule.emb_sizes,
    )

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

    train_tabular_data(args)
