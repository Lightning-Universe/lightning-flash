import pytorch_lightning as pl
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from flash import ClassificationTask

# 1. Load a basic backbone
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

# 2. Load a dataset
dataset = datasets.MNIST('./data', download=True, transform=transforms.ToTensor())

# 3. Split the data randomly
train, val, test = random_split(dataset, [50000, 5000, 5000])

# 4. Create the model
classifier = ClassificationTask(model, loss_fn=nn.functional.cross_entropy, optimizer=optim.Adam, learning_rate=10e-3)

# 5. Create the trainer
trainer = pl.Trainer(
    max_epochs=10,
    limit_train_batches=128,
    limit_val_batches=128,
)

# 6. Train the model
trainer.fit(classifier, DataLoader(train), DataLoader(val))

# 7. Test the model
results = trainer.test(classifier, test_dataloaders=DataLoader(test))
