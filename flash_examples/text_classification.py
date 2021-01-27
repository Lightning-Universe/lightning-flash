# import our libraries
import pytorch_lightning as pl
from pytorch_lightning import _logger as log

from flash.core.data import download_data
from flash.text import TextClassificationData, TextClassifier

RUN_TRAINING = False

download_data("https://pl-flash-data.s3.amazonaws.com/imdb.zip", 'data/')

backbone = "bert-base-uncased"

# 1. organize our data
datamodule = TextClassificationData.from_files(
    backbone=backbone,
    train_file="data/imdb/train.csv",
    valid_file="data/imdb/valid.csv",
    text_field="review",
    label_field="sentiment",
)


# 2. build model
model = TextClassifier(backbone, num_classes=datamodule.num_classes)

# 3. train model
trainer = pl.Trainer(
    max_epochs=1,
    limit_train_batches=8,
    limit_val_batches=8,
    limit_test_batches=8,
)

# 4. train model
if RUN_TRAINING:
    trainer.fit(model, datamodule=datamodule)

# 5.1 Perform inference from text
predictions = model.predict([
    "Turgid dialogue, feeble characterization - Harvey Keitel a judge? He plays more like an off-duty hitman - and a tension-free plot conspire to make one of the unfunniest films of all time. You feel sorry for the cast as they try to extract comedy from a dire and lifeless script. Avoid!",  # noqa E501
    "The worst movie in the history of cinema. I don't know if it was trying to be funny or sad, poignant or droll, but the end result was unwatchable. Everyone from Key Grip, to Robin Williams, and back down to Best Boy should be ashamed to be a part of this film!",  # noqa E501
    "Well , I come from Bulgaria where it 's almost impossible to have a tornado but my imagination tells me to be "
    "very , very afraid"
    "!!!This guy (Devon Sawa) has done a great job with this movie!I don't know exactly how old he was but he didn't act like a child (WELL DONE)!Now about the tornado-it wasn't very realistic but frightens you!If you want to have a nice time in front of the telly - this is the movie!",  # noqa E501
])

log.info(predictions)

# 5.2 Perform inference from csv file
datamodule = TextClassificationData.from_file(
    "data/imdb/test.csv",
    "review",
)
predictions = trainer.predict(model, datamodule=datamodule)
log.info(predictions)
