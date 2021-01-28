import flash
from flash.core.data import download_data
from flash.vision import ImageClassificationData, ImageClassifier

if __name__ == "__main__":

    # 1. Download the data
    download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", 'data/')

    # 2. Load the data
    datamodule = ImageClassificationData.from_folders(
        train_folder="data/hymenoptera_data/train/",
        valid_folder="data/hymenoptera_data/val/",
    )

    # 3. Build the model
    model = ImageClassifier(num_classes=datamodule.num_classes)

    # 4. Create the trainer. Run once on data
    trainer = flash.Trainer(max_epochs=1)

    # 5. Train the model
    trainer.finetune(model, datamodule=datamodule, unfreeze_milestones=(0, 1))

    # 6. Save it!
    trainer.save_checkpoint("image_classification_model.pt")
