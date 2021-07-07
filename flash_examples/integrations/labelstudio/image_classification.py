from itertools import chain

from flash.core.data.utils import download_data

import flash
from flash.core.classification import Labels
from flash.core.finetuning import FreezeUnfreeze
from flash.image import ImageClassificationData, ImageClassifier

# 1 Download data
download_data("https://label-studio-testdata.s3.us-east-2.amazonaws.com/lightning-flash/data.zip")

# 1. Load export data
datamodule = ImageClassificationData.from_labelstudio(
    export_json='data/project.json',
    img_folder='data/upload/',
    val_split=0.8,
)

# 2. Fine tune a model
model = ImageClassifier(
    backbone="resnet18",
    num_classes=datamodule.num_classes,
)
trainer = flash.Trainer(max_epochs=3)

trainer.finetune(
    model,
    datamodule=datamodule,
    strategy=FreezeUnfreeze(unfreeze_epoch=1),
)
trainer.save_checkpoint("image_classification_model.pt")

# 3. Predict from checkpoint
model = ImageClassifier.load_from_checkpoint("image_classification_model.pt")
model.serializer = Labels()

predictions = model.predict([
    "data/test/1.jpg",
    "data/test/2.jpg",
])
