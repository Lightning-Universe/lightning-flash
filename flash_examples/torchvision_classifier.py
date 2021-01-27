# import our libraries
import os

import pytorch_lightning as pl
from pytorch_lightning import _logger as log
from pytorch_lightning.callbacks import BackboneLambdaFinetuningCallback

from flash.core.data import download_data
from flash.vision import ImageClassificationData, ImageClassifier
from flash.vision.classification.dataset import hymenoptera_data_download

RUN_TRAINING = True

# 1. make reproducible
pl.seed_everything(42)

# 2. download data
hymenoptera_data_download('data/')

# 3. organize our data
datamodule = ImageClassificationData.from_folders(
    train_folder="data/hymenoptera_data/train/", valid_folder="data/hymenoptera_data/val/", batch_size=2, num_workers=0
)

# 4. build our model
model = ImageClassifier(backbone="resnet18", num_classes=datamodule.num_classes)

# 5. Create trainer
trainer = pl.Trainer(
    max_epochs=1,
    limit_train_batches=1,
    limit_val_batches=1,
    limit_test_batches=1,
)

# 6. Optional: train the model
if RUN_TRAINING:
    trainer.fit(model, datamodule=datamodule)

# 7. predict our model on list of images
predictions = model.predict([
    "data/hymenoptera_data/val/bees/65038344_52a45d090d.jpg",
    "data/hymenoptera_data/val/bees/590318879_68cf112861.jpg",
    "data/hymenoptera_data/val/ants/540543309_ddbb193ee5.jpg",
])

# 8. Check prediction.
log.info(predictions)

# 9. Inference on an entire folder.
datamodule = ImageClassificationData.from_folder(folder="data/hymenoptera_data/predict/", num_workers=1)
predictions = trainer.predict(model, datamodule=datamodule)
log.info(predictions)
