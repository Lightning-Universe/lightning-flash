import pytorch_lightning as pl
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from flash import ClassificationTask

# 1 . create model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

# 2. create dataset
dataset = datasets.MNIST('./data', download=True, transform=transforms.ToTensor())

# 3. randomly split the data
train, val, test = random_split(dataset, [50000, 5000, 5000])

# 4. create model
classifier = ClassificationTask(model, loss_fn=nn.functional.cross_entropy, optimizer=optim.Adam, learning_rate=10e-3)

# 5. create trainer
trainer = pl.Trainer(
    max_epochs=10,
    limit_train_batches=128,
    limit_val_batches=128,
)

# 6. train model
trainer.fit(classifier, DataLoader(train), DataLoader(val))

# 7. test model
results = trainer.test(classifier, test_dataloaders=DataLoader(test))
