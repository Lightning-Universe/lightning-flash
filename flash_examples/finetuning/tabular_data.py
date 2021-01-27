# import our libraries
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.metrics.classification import Accuracy, Precision, Recall

from flash.tabular import TabularClassifier, TabularData
from flash.tabular.classification.data.dataset import titanic_data_download

# 1. Download data
titanic_data_download("./data/titanic")

# 2. Organize our data - create a LightningDataModule
datamodule = TabularData.from_df(
    pd.read_csv("./data/titanic/titanic.csv"),
    categorical_input=["Sex", "Age", "SibSp", "Parch", "Ticket", "Cabin", "Embarked"],
    numerical_input=["Fare"],
    target="Survived",
    val_size=0.25,
)

# 3. Build model
model = TabularClassifier.from_data(datamodule, metrics=[Accuracy(), Precision(), Recall()])

# 4. Create trainer
trainer = pl.Trainer(max_epochs=1)

# 5. Train model
trainer.fit(model, datamodule=datamodule)

# 6. Save model
torch.save(model, "tabular_model.pt")
