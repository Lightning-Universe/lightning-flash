# import our libraries
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning import _logger as log

from pl_flash.data import download_data
from pl_flash.tabular import TabularClassifier, TabularData

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

# 3. build model
model: LightningModule = TabularClassifier(
    num_classes=datamodule.num_classes,
    num_features=datamodule.num_features,
    embedding_sizes=datamodule.emb_sizes,
)

# 4. create trainer
trainer = pl.Trainer(max_epochs=10, limit_test_batches=8)

# 5. train mode
trainer.fit(model, datamodule=datamodule)

# 6. test model
results = trainer.test(model, datamodule=datamodule)
log.info(results)
