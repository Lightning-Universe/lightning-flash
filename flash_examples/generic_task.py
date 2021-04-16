# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import pytorch_lightning as pl
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from flash import ClassificationTask
from flash.data.utils import download_data

_PATH_ROOT = os.path.dirname(os.path.dirname(__file__))

# 1. Download the data
download_data("https://www.di.ens.fr/~lelarge/MNIST.tar.gz", os.path.join(_PATH_ROOT, 'data'))

# 2. Load a basic backbone
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

# 3. Load a dataset
dataset = datasets.MNIST(os.path.join(_PATH_ROOT, 'data'), download=False, transform=transforms.ToTensor())

# 4. Split the data randomly
train, val, test = random_split(dataset, [50000, 5000, 5000])  # type: ignore

# 5. Create the model
classifier = ClassificationTask(model, loss_fn=nn.functional.cross_entropy, optimizer=optim.Adam, learning_rate=10e-3)

# 6. Create the trainer
trainer = pl.Trainer(
    max_epochs=10,
    limit_train_batches=128,
    limit_val_batches=128,
)

# 7. Train the model
trainer.fit(classifier, DataLoader(train), DataLoader(val))

# 8. Test the model
results = trainer.test(classifier, test_dataloaders=DataLoader(test))
