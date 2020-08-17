# pytorch-lightning-tasks
Simplified lightning and predefined steps

## Demo

```python
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as T

from pl_flash import Flash
from pytorch_lightning.metrics import functional as FM

data = DataLoader(MNIST(".", download=True, transform=T.ToTensor()), batch_size=32)

mlp = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
    nn.LogSoftmax(),
)
model = Flash(mlp, loss=F.nll_loss, metrics=[FM.accuracy])

model.fit(data)
```
