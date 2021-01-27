# import our libraries
import pytorch_lightning as pl
import torch

from pytorch_lightning import _logger as log

# 1. create trainer
model = torch.load("tabular_model.pt")

# 2. create trainer
trainer = pl.Trainer()

# 3. Predict over a path to a csv file
predictions = model.predict("./data/titanic/predict.csv")
log.info(predictions)
