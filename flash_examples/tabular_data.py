# import our libraries
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import _logger as log
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics.classification import Accuracy, IoU, Precision, Recall

from flash.data import download_data
from flash.tabular import TabularClassifier, TabularData

# 1. download data
download_data("https://pl-flash-data.s3.amazonaws.com/titanic.csv", "titanic.csv")

# 2. organize our data - create a LightningDataModule
datamodule = TabularData.from_df(
    pd.read_csv("titanic.csv"),
    categorical_cols=["Sex", "Age", "SibSp", "Parch", "Ticket", "Cabin", "Embarked"],
    numerical_cols=["Fare"],
    target_col="Survived",
    num_workers=0,
    batch_size=8,
    val_size=0.25,
    test_size=0.25,
)

# 3. create metrics
metrics = [Accuracy(), Precision(), Recall(), IoU(datamodule.num_classes)]

# 4. build model
model: LightningModule = TabularClassifier(**datamodule.data_config, metrics=metrics)

# 5. create trainer
trainer = pl.Trainer(max_epochs=1, limit_test_batches=8)

# 6. train mode
trainer.fit(model, datamodule=datamodule)

# 7. Helper for test - In reality, you would provide your own test DataFrame.
test_df = datamodule.test_df

# 8. Predict over provided list of DataFrame.
predictions = model.predict([test_df])
log.info(predictions)
