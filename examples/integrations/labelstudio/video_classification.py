import os

import flash
from flash.core.data.utils import download_data
from flash.core.integrations.labelstudio.visualizer import launch_app
from flash.video import VideoClassificationData, VideoClassifier

# 1 Download data
download_data("https://label-studio-testdata.s3.us-east-2.amazonaws.com/lightning-flash/video_data.zip")

# 2. Load export data
datamodule = VideoClassificationData.from_labelstudio(
    export_json="data/project.json",
    data_folder="data/upload/",
    val_split=0.2,
    clip_sampler="uniform",
    clip_duration=1,
    decode_audio=False,
)

# 3. Build the task
model = VideoClassifier(
    backbone="slow_r50",
    num_classes=datamodule.num_classes,
)

# 4. Create the trainer and finetune the model
trainer = flash.Trainer(max_epochs=3)
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

# 5. Make a prediction
datamodule = VideoClassificationData.from_folders(predict_folder=os.path.join(os.getcwd(), "data/test"))
predictions = trainer.predict(model, datamodule=datamodule)

# 6. Save the model!
trainer.save_checkpoint("video_classification.pt")

# 7. Visualize predictions
app = launch_app(datamodule)
print(app.show_predictions(predictions))
