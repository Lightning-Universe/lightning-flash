# import our libraries
import pytorch_lightning as pl
from pytorch_lightning import _logger as log
from pytorch_lightning.callbacks import BackboneLambdaFinetuningCallback

from pl_flash.data import download_data
from pl_flash.vision import ImageClassificationData, ImageClassifier

download_data("https://download.pytorch.org/tutorial/hymenoptera_data.zip", 'data/')

# 1. organize our data
datamodule = ImageClassificationData.from_folders(
    train_folder="data/hymenoptera_data/train/",
    valid_folder="data/hymenoptera_data/val/",
    test_folder="data/hymenoptera_data/val/",
    batch_size=2,
)

# 2. build our model
model = ImageClassifier(backbone="resnet18", num_classes=datamodule.num_classes)

# 3. create finetunning schedule
finetuning_callback = BackboneLambdaFinetuningCallback(unfreeze_backbone_at_epoch=2)

# 4. create Lightning trainer
trainer = pl.Trainer(
    max_epochs=5,
    limit_train_batches=128,
    limit_val_batches=32,
    callbacks=[finetuning_callback])

# 5. train our model
trainer.fit(model, datamodule=datamodule)

# 6. test our model
results = trainer.test(model, datamodule=datamodule)
log.info(results)
