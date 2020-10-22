<div align="center">

![Logo](https://raw.githubusercontent.com/PyTorchLightning/pytorch-lightning/master/docs/source/_images/logos/lightning_logo.svg)

# Flash

[![CI testing](https://github.com/PyTorchLightning/pytorch-lightning-flash/workflows/CI%20testing/badge.svg)](https://github.com/PyTorchLightning/pytorch-lightning-flash/actions?query=workflow%3A%22CI+testing%22)
![Check Code formatting](https://github.com/PyTorchLightning/pytorch-lightning-flash/workflows/Check%20Code%20formatting/badge.svg)
[Docs](https://pytorchlightning.github.io/pytorch-lightning-flash/)

</div>

## Installation

### Install with pip

```bash
pip install git+https://github.com/PyTorchLightning/pytorch-lightning-flash/
```

### Install from source

``` bash
git clone https://github.com/PyTorchLightning/python-lightning-flash.git
cd pytorch-lightning-flash 
pip install -e .
```



## Vision

### Image Classification

```python
from pl_flash.vision import ImageClassifier, ImageClassificationData
import pytorch_lightning as pl

model = ImageClassifier(backbone="resnet18", num_classes=2)

data = ImageClassificationData.from_folders(
    train_folder="train/",
    valid_folder="val/"
)

pl.Trainer().fit(model, data)
```

## Text

### Text Classification

```python
from pl_flash.text import TextClassifier, TextClassificationData
import pytorch_lightning as pl

model = TextClassifier(backbone="bert-base-cased", num_classes=2)

data = TextClassificationData.from_files(
    backbone="bert-base-cased",
    train_file="train.csv",
    valid_file="val.csv",
    text_field="sentence",
    label_field="label",
)

pl.Trainer().fit(model, data)
```

## Tabular Classification

```python
from pl_flash.tabular import TabularClassifier, TabularData

import pytorch_lightning as pl
import pandas as pd


train_df = pd.read_csv("train.csv")

data = TabularData.from_df(
    train_df=train_df,
    categorical_cols=["category"],
    numerical_cols=["scalar_b", "scalar_b"],
    target_col="target",
)

model = TabularClassifier(
    num_classes=2,
    num_columns=3,
    embedding_sizes=data.emb_sizes,
)

pl.Trainer().fit(model, data)
```

## Custom
```python
import torch
from torch import nn
import torch.nn.functional as F

from pl_flash import Task
import pytorch_lightning as pl

mlp = nn.Sequential(...)
model = Task(mlp, loss=F.cross_entropy)

pl.Trainer().fit(model, torch.utils.data.DataLoader(...))
```
