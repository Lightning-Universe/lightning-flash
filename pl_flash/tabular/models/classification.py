from typing import Callable, Dict, List, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pandas.core.frame import DataFrame
from pytorch_lightning.metrics import Metric
from torch import nn
from torch.utils.data import DataLoader

from pl_flash.model import ClassificationLightningTask
from pl_flash.tabular.data.dataset import PandasDataset, _pre_transform


class TabularClassifier(ClassificationLightningTask):
    """LightningTask that classifies table rows.

    Args:
        num_features: Number of columns in table (not including target column).
        num_classes: Number of classes to classify.
        embedding_sizes: List of (num_classes, emb_dim) to form categorical embeddings.
        hidden: Hidden dimension sizes.
        loss_fn: Loss function for training, defaults to cross entropy.
        optimizer: Optimizer to use for training, defaults to `torch.optim.Adam`.
        metrics: Metrics to compute for training and evaluation.
        learning_rate: Learning rate to use for training, defaults to `1e-3`
    """

    def __init__(
        self,
        num_classes: int,
        num_features: int,
        embedding_sizes: List[Tuple] = None,
        codes: Dict = None,
        mean: DataFrame = None,
        std: DataFrame = None,
        cat_cols: List = None,
        num_cols: List = None,
        hidden=[512],
        loss_fn: Callable = F.cross_entropy,
        optimizer=torch.optim.Adam,
        metrics: List[Metric] = None,
        learning_rate: float = 1e-3,
    ):
        super().__init__(
            model=None,
            loss_fn=loss_fn,
            optimizer=optimizer,
            metrics=metrics,
            learning_rate=learning_rate,
        )

        self.save_hyperparameters()

        num_num = self.hparams.num_features - len(self.hparams.embedding_sizes)  # numerical columns
        input_size = num_num + sum(emb_dim for _, emb_dim in self.hparams.embedding_sizes)
        sizes = [input_size] + hidden + [self.hparams.num_classes]

        self.embs = nn.ModuleList([nn.Embedding(n_emb, emb_dim) for n_emb, emb_dim in self.hparams.embedding_sizes])
        self.bn_num = nn.BatchNorm1d(num_num) if num_num > 0 else None
        self.mlp = self._init_mlp(sizes)

    def _init_mlp(self, sizes):
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(
                nn.Sequential(
                    nn.BatchNorm1d(sizes[i]),
                    nn.Linear(sizes[i], sizes[i + 1], bias=False),
                    nn.ReLU(),
                )
            )
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        return nn.Sequential(*layers)

    def forward(self, x_in):
        x_cat, x_num = x_in
        if len(self.embs):
            # concatenate embeddings for each categorical variable
            x = [e(x_cat[:, i]) for i, e in enumerate(self.embs)]
            x = torch.cat(x, dim=1)
        if self.bn_num is not None:
            x_num = self.bn_num(x_num)
            x = torch.cat([x_num, x], dim=1) if len(self.embs) else x_num
        x = self.mlp(x)
        return x

    def predict(self, dfs: List[DataFrame], batch_size: int = 2, num_workers: int = 0, **kwargs):
        """
        This function is used to make prediction directly from raw_data
        """
        self._predict = True

        assert isinstance(dfs, list)
        assert isinstance(dfs[0], DataFrame)

        # pre-transform used during training phase
        dfs = _pre_transform(
            dfs,
            self.hparams.num_cols,
            self.hparams.cat_cols,
            self.hparams.codes,
            self.hparams.mean,
            self.hparams.std
        )

        # create test dataloaders
        test_dataloaders = [
            DataLoader(
                PandasDataset(df, self.hparams.cat_cols, self.hparams.num_cols, None, predict=True),
                batch_size=batch_size,
                num_workers=num_workers,
            ) for df in dfs]

        # create trainaer
        trainer = pl.Trainer(**kwargs)

        # perform inference using test
        results = trainer.test(self, test_dataloaders=test_dataloaders)

        # if predictions are available, convert them into DataFrame
        outputs = []
        if "predictions" in results[0]:
            for r in results:
                outputs.append(pd.json_normalize(r["predictions"], sep='_'))
        else:
            results = outputs
        return outputs
