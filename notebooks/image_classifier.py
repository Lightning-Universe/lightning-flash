# -*- coding: utf-8 -*-
# +
import os
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import pytorch_lightning as pl
import torch

from flash.vision import ImageClassificationData, ImageClassifier

# -

# First we'll download our data:

with urlopen("https://download.pytorch.org/tutorial/hymenoptera_data.zip") as resp:
    with ZipFile(BytesIO(resp.read())) as file:
        file.extractall('data/')

# Our data is sorted by class in train and val folders:
# ```
# hymenoptera_data
# ├── train
# │   ├── ants
# │   └── bees
# └── val
#     ├── ants
#     └── bees
# ```
# We can create a `pl.DataModule` from this like so:

data = ImageClassificationData.from_folders(
    train_folder="data/hymenoptera_data/train/",
    valid_folder="data/hymenoptera_data/val/",
    batch_size=4,
)

model = ImageClassifier(
    backbone="resnet18",
    num_classes=2,
    metrics=pl.metrics.Accuracy(),
    optimizer=torch.optim.SGD,
    learning_rate=0.001,
)

trainer = pl.Trainer(max_epochs=25, fast_dev_run=os.getenv("TEST_ENV", False))
trainer.fit(model, data)
