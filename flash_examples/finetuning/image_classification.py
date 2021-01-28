# import our libraries
import torch

import flash
from flash.core.data import download_data
from flash.vision import ImageClassificationData, ImageClassifier

if __name__ == "__main__":

    # 1. Download data
    download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", 'data/')

    # 2. Organize our data
    datamodule = ImageClassificationData.from_folders(
        train_folder="data/hymenoptera_data/train/",
        valid_folder="data/hymenoptera_data/val/",
    )

    # 3. Build a model
    model = ImageClassifier(num_classes=datamodule.num_classes)

    # 4. Create trainer
    trainer = flash.Trainer(max_epochs=2)

    # 5. Train the model
    trainer.finetune(model, datamodule=datamodule, unfreeze_milestones=(0, 1))

    # 6. Save model
    torch.save(model, "image_classification_model.pt")
