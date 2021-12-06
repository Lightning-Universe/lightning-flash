import flash
from flash.core.data.utils import download_data
from flash.core.integrations.labelstudio.visualizer import launch_app
from flash.text import TextClassificationData, TextClassifier

# 1. Create the DataModule
download_data("https://label-studio-testdata.s3.us-east-2.amazonaws.com/lightning-flash/text_data.zip", "./data/")

backbone = "prajjwal1/bert-medium"

datamodule = TextClassificationData.from_labelstudio(
    export_json="data/project.json",
    val_split=0.2,
    backbone=backbone,
)

# 2. Build the task
model = TextClassifier(backbone=backbone, num_classes=datamodule.num_classes)

# 3. Create the trainer and finetune the model
trainer = flash.Trainer(max_epochs=3)
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

# 4. Classify a few sentences! How was the movie?
predict_datamodule = TextClassificationData.from_lists(
    predict_data=[
        "Turgid dialogue, feeble characterization - Harvey Keitel a judge?.",
        "The worst movie in the history of cinema.",
        "I come from Bulgaria where it 's almost impossible to have a tornado.",
    ]
)
predictions = trainer.predict(model, datamodule=predict_datamodule)

# 5. Save the model!
trainer.save_checkpoint("text_classification_model.pt")

# 6. Visualize predictions
app = launch_app(datamodule)
print(app.show_predictions(predictions))
