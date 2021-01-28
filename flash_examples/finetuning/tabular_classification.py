# import our libraries
import torch
from pytorch_lightning.metrics.classification import Accuracy, Precision, Recall

import flash
from flash.core.data import download_data
from flash.tabular import TabularClassifier, TabularData

# 1. Download data
download_data("https://pl-flash-data.s3.amazonaws.com/titanic.zip", 'data/')

# 2. Organize our data - create a LightningDataModule
datamodule = TabularData.from_csv(
    "./data/titanic/titanic.csv",
    categorical_input=["Sex", "Age", "SibSp", "Parch", "Ticket", "Cabin", "Embarked"],
    numerical_input=["Fare"],
    target="Survived",
    val_size=0.25,
)

# 3. Build model
model = TabularClassifier.from_data(datamodule, metrics=[Accuracy(), Precision(), Recall()])

# 4. Create trainer
trainer = flash.Trainer(max_epochs=1)

# 5. Train model
trainer.fit(model, datamodule=datamodule)

# 6. Save model
torch.save(model, "tabular_classification_model.pt")
