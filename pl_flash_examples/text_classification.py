# import our libraries
import pytorch_lightning as pl
from pytorch_lightning import _logger as log

from pl_flash.data import download_data
from pl_flash.text import TextClassificationData, TextClassifier

download_data("https://pl-flash-data.s3.amazonaws.com/imdb.zip", 'data/')

backbone = "bert-base-uncased"

# 1. organize our data
datamodule = TextClassificationData.from_files(
    backbone=backbone,
    train_file="data/imdb/train.csv",
    valid_file="data/imdb/valid.csv",
    test_file="data/imdb/test.csv",
    text_field="review",
    label_field="sentiment",
    batch_size=1,
)

# 2. build model
model = TextClassifier(backbone=backbone, num_classes=datamodule.num_classes)

# 3. train model
trainer = pl.Trainer(
    max_epochs=1,
    limit_train_batches=8,
    limit_val_batches=8,
    limit_test_batches=8
)

# 4. train model
trainer.fit(model, datamodule=datamodule)

# 5. test model
results = trainer.test(model, datamodule=datamodule)

# 6. log for introspection
log.info(results)
