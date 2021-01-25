# import our libraries
import pytorch_lightning as pl
from pytorch_lightning import _logger as log
from pytorch_lightning.callbacks import BackboneLambdaFinetuningCallback

from flash.data import download_data
from flash.vision import ImageClassificationData, ImageClassifier

# 0. make reproducible
pl.seed_everything(42)

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
    max_epochs=1,
    limit_train_batches=64,
    limit_val_batches=2,
    limit_test_batches=2,
    callbacks=[finetuning_callback],
)

# 5. train our model
trainer.fit(model, datamodule=datamodule)

# 6. predict our model on list of images
predictions = model.predict(
    [
        "data/hymenoptera_data/val/bees/65038344_52a45d090d.jpg",
        "data/hymenoptera_data/val/bees/590318879_68cf112861.jpg",
        "data/hymenoptera_data/val/ants/540543309_ddbb193ee5.jpg",
    ],
    transform=datamodule.default_valid_transforms,
)

# 7. Check prediction.
log.info(predictions)
"""
Out:
[                                                  id                                predictions
0  data/hymenoptera_data/val/bees/65038344_52a45d...   [0.2558616101741791, 0.7441383600234985]
1  data/hymenoptera_data/val/bees/590318879_68cf1...  [0.17213837802410126, 0.8278616666793823]
2  data/hymenoptera_data/val/ants/540543309_ddbb1...     [0.653433620929718, 0.346566379070282]]
"""
