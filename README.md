<div align="center">

![Logo](https://raw.githubusercontent.com/PyTorchLightning/pytorch-lightning/master/docs/source/_images/logos/lightning_logo-name.svg)

# Flash

[![CI testing](https://github.com/PyTorchLightning/pytorch-lightning-flash/workflows/CI%20testing/badge.svg)](https://github.com/PyTorchLightning/pytorch-lightning-flash/actions?query=workflow%3A%22CI+testing%22)
![Check Code formatting](https://github.com/PyTorchLightning/pytorch-lightning-flash/workflows/Check%20Code%20formatting/badge.svg)
[![Docs](https://github.com/PyTorchLightning/pytorch-lightning-flash/workflows/Docs/badge.svg)](https://pytorchlightning.github.io/pytorch-lightning-flash/)

</div>

## Installation

Pip

```bash
pip install pytorch-lightning-flash
```

Source

``` bash
git clone https://github.com/PyTorchLightning/pytorch-lightning-flash.git
cd pytorch-lightning-flash 
pip install -e .
```

## Train any PyTorch model

```python
from pl_flash import Task
from torch import nn
from torch.optim import Adam
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# data
dataset = MNIST('./data_folder', download=True, transform=transforms.ToTensor())
train, val = random_split(dataset, [55000, 5000])

classifier = Task(model, loss_fn=nn.functional.cross_entropy, optimizer=Adam)
pl.Trainer().fit(classifier, DataLoader(train), DataLoader(val))
```

## Vision example:

```python
from pl_flash.vision import ImageClassifier, ImageClassificationData
import pytorch_lightning as pl

# 1. build our model
model = ImageClassifier(backbone="resnet18", num_classes=2)

# 2. organize our data
data = ImageClassificationData.from_folders(
    train_folder="train/",
    valid_folder="validation/"
)

# 3. train!
pl.Trainer().fit(model, data)
```

## Text

```python
import pytorch_lightning as pl

# build our model
model = TextClassifier(backbone="bert-base-cased", num_classes=2)

# structure our data
data = TextClassificationData.from_files(
    backbone="bert-base-cased",
    train_file="train.csv",
    valid_file="val.csv",
    text_field="sentence",
    label_field="label",
)

# train
pl.Trainer().fit(model, data)
```

## Tabular

```python
from pl_flash.tabular import TabularClassifier, TabularData

import pytorch_lightning as pl
import pandas as pd

# stucture data
data = TabularData.from_df(
    pd.read_csv("train.csv"),
    categorical_cols=["years_as_customer", "country"],
    numerical_cols=["money_spent", "purchases"],
    target_col="cancelled",
)

# build model
model = TabularClassifier(
    num_classes=2,
    num_columns=3,
    embedding_sizes=data.emb_sizes,
)

# train
pl.Trainer().fit(model, data)
```
