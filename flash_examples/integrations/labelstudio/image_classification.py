from itertools import chain

import flash
from flash.core.classification import Labels
from flash.core.finetuning import FreezeUnfreeze
from flash.image import ImageClassificationData, ImageClassifier

# 1. Load export data
datamodule = ImageClassificationData.from_labelstudio(
    export_json='project.json',
    img_folder=r'C:\Users\MI\AppData\Local\label-studio\label-studio\media\upload',
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
model = ImageClassifier.load_from_checkpoint("../../image_classification_model.pt")
model.serializer = Labels()

predictions = trainer.predict(model, datamodule=datamodule)
predictions = list(chain.from_iterable(predictions))

# 5 Visualize predictions in FiftyOne App
