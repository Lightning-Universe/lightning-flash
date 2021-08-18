import os

import flash
from flash.core.data.utils import download_data
from flash.video import VideoClassificationData, VideoClassifier

# 1 Download data
download_data("https://label-studio-testdata.s3.us-east-2.amazonaws.com/lightning-flash/video_data.zip")

# 1. Load export data
datamodule = VideoClassificationData.from_labelstudio(
    export_json="data/project.json",
    data_folder="data/upload/",
    data_type="video",
    val_split=0.8,
    clip_sampler="uniform",
    clip_duration=1,
    decode_audio=False,
)

# 2. Build the task
model = VideoClassifier(
    backbone="slow_r50",
    num_classes=datamodule.num_classes,
)

# 3. Create the trainer and finetune the model
trainer = flash.Trainer(max_epochs=3)
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

# 4. Make a prediction
predictions = model.predict(os.path.join(os.getcwd(), "data/test"))
print(predictions)

# 5. Save the model!
trainer.save_checkpoint("video_classification.pt")
