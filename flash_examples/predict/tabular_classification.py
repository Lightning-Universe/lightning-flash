# import our libraries
import torch
from pytorch_lightning import _logger as log

# 1. Load model from checkpoint
model = torch.load("tabular_classification_model.pt")

# 2. Predict over a path to a `.csv` file
predictions = model.predict("./data/titanic/predict.csv")
print(predictions)
