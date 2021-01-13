import platform
from argparse import ArgumentParser

import pandas as pd
import pytest
import pytorch_lightning as pl

from pl_flash.tabular import TabularClassifier, TabularData
from pl_flash_examples.tabular_data import train_tabular_data

TEST_DF_1 = pd.DataFrame(
    data={
        "category": ["a", "b", "c", "a", None, "c"],
        "scalar_a": [0.0, 1.0, 2.0, 3.0, None, 5.0],
        "scalar_b": [5.0, 4.0, 3.0, 2.0, None, 1.0],
        "label": [0, 1, 0, 1, 0, 1],
    }
)


def test_classification(tmpdir):
    train_df = TEST_DF_1.copy()
    valid_df = TEST_DF_1.copy()
    test_df = TEST_DF_1.copy()
    data = TabularData.from_df(
        train_df,
        categorical_cols=["category"],
        numerical_cols=["scalar_b", "scalar_b"],
        target_col="label",
        valid_df=valid_df,
        test_df=test_df,
        num_workers=0,
        batch_size=2,
    )
    model = TabularClassifier(num_columns=3, num_classes=2, embedding_sizes=data.emb_sizes)
    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, data)


@pytest.mark.skipif(platform.system() == "Windows", reason="Distributed training is not supported on Windows")
def test_tabular_data_example(tmpdir):
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    cmd = "--max_epochs 1 --fast_dev_run 1 --num_processes 2 --accelerator ddp_cpu"
    args = parser.parse_args(cmd.split(" "))
    train_tabular_data(args)
