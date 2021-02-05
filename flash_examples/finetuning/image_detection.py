import flash
from flash.vision.detection.data import ImageDetectionData
from flash.vision.detection.dataset import coco128_data_download
from flash.vision.detection.model import ImageDetector

# 1. Download the data
coco128_data_download("data/")

# 2. Load the Data
datamodule = ImageDetectionData.from_coco(
    train_folder="data/coco128/images/train2017/",
    train_ann_file="data/coco128/annotations/instances_train2017.json",
    batch_size=2
)

# 3. Build the model
model = ImageDetector(num_classes=datamodule.num_classes)

# 4. Create the trainer. Run twice on data
trainer = flash.Trainer(max_epochs=2)

# 5. Finetune the model
trainer.finetune(model, datamodule)

# 6. Save it!
trainer.save_checkpoint("image_detection_model.pt")
