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

Master
```bash
pip install git+https://github.com/PytorchLightning/pytorch-lightning-flash.git@master --upgrade
```

Source

``` bash
git clone https://github.com/PyTorchLightning/pytorch-lightning-flash.git
cd pytorch-lightning-flash 
pip install -e .
```

## What is Flash
Flash is a task-based deep learning framework for flexible deep learning built on PyTorch Lightning. Tasks can be anything from
text classification to object segmentation.

Although PyTorch Lightning provides ultimate flexibility, for common tasks it does not remove 100% of the boilerplate.

Flash is built for applied researchers, beginners, data scientists, Kagglers or anyone starting out with Deep Learning. But unlike other entry-level frameworks (keras, etc...), Flash users can switch to Lightning trivially when they need added flexibility.

## Example 1: Generic Task for training any nn.Module.
<img src="https://pl-flash-data.s3.amazonaws.com/images/mnist.png" width="200px">

```python
from pl_flash import LightningTask
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import pytorch_lightning as pl

# model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# data
dataset = datasets.MNIST('./data_folder', download=True, transform=transforms.ToTensor())
train, val = random_split(dataset, [55000, 5000])

# task
classifier = LightningTask(model, loss_fn=nn.functional.cross_entropy, optimizer=optim.Adam)

# train
pl.Trainer().fit(classifier, DataLoader(train), DataLoader(val))
```

## Example 2: A task for computer vision.
<img src="https://pl-flash-data.s3.amazonaws.com/images/ant_bee.png" width="300px">
Here we classify ants vs bees.

```python
# import our libraries
from pl_flash.vision import ImageClassifier, ImageClassificationData
import pytorch_lightning as pl
from pl_flash.data import download_data

# download data
download_data("https://download.pytorch.org/tutorial/hymenoptera_data.zip", 'data/')

# 1. organize our data
data = ImageClassificationData.from_folders(
    train_folder="data/hymenoptera_data/train/",
    valid_folder="data/hymenoptera_data/val/"
)

# 2. build our model
model = ImageClassifier(backbone="resnet18", num_classes=2)


# 3. train!
pl.Trainer().fit(model, data)
```

## Example 3: A task for NLP

```python
from pl_flash.text import TextClassifier, TextClassificationData
import pytorch_lightning as pl
from pl_flash.data import download_data

# download data
download_data("https://pl-flash-data.s3.amazonaws.com/imdb.zip", 'data/')

# build our model
model = TextClassifier(backbone="bert-base-cased", num_classes=2)

# structure our data
data = TextClassificationData.from_files(
    backbone="bert-base-cased",
    train_file="data/imdb/train.csv",
    valid_file="data/imdb/valid.csv",
    text_field="review",
    label_field="sentiment",
)

# train
pl.Trainer().fit(model, data)
```

## Example 3: A task for Tabular data.

```python
from pl_flash.tabular import TabularClassifier, TabularData
import pytorch_lightning as pl
import pandas as pd
from pl_flash.data import download_data

# download data
download_data("https://pl-flash-data.s3.amazonaws.com/titanic.csv", "titanic.csv")

# structure data
data = TabularData.from_df(
    pd.read_csv("titanic.csv"),
    categorical_cols=["Sex", "Age", "SibSp", "Parch", "Ticket", "Cabin", "Embarked"],
    numerical_cols=["Fare"],
    target_col="Survived",
    num_workers=0,
    batch_size=8
)

# build model
model = TabularClassifier(
    num_classes=2,
    num_columns=8,
    embedding_sizes=data.emb_sizes,
)

pl.Trainer().fit(model, data)
```
