import flash
from flash.core.classification import FiftyOneLabels, Labels, Probabilities
from flash.core.data.utils import download_data
from flash.core.finetuning import FreezeUnfreeze
from flash.core.integrations.fiftyone import fiftyone_launch_app
from flash.image import ImageClassificationData, ImageClassifier

# 1 Download data
download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip")

# 2 Load data using fiftyone
datamodule = ImageClassificationData.fiftyone_from_dir(
    train_dir="data/hymenoptera_data/train/",
    val_dir="data/hymenoptera_data/val/",
    test_dir="data/hymenoptera_data/test/",
    predict_dir="data/hymenoptera_data/predict/",
)

# 3 Fine tune a model
model = ImageClassifier(backbone="resnet18", num_classes=datamodule.num_classes, serializer=Labels())
trainer = flash.Trainer(max_epochs=1, limit_train_batches=1, limit_val_batches=1)
trainer.finetune(model, datamodule=datamodule, strategy=FreezeUnfreeze(unfreeze_epoch=1))
trainer.save_checkpoint("image_classification_model.pt")

# 4 Predict from checkpoint
model = ImageClassifier.load_from_checkpoint("https://flash-weights.s3.amazonaws.com/image_classification_model.pt")
model.serializer = FiftyOneLabels()
predictions = trainer.predict(model, datamodule=datamodule)

# 5. Visualize predictions in FiftyOne for 2 minutes.
session = fiftyone_launch_app(predictions)
