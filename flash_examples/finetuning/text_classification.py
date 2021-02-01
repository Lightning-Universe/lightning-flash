import flash
from flash.core.data import download_data
from flash.text import TextClassificationData, TextClassifier

if __name__ == "__main__":

    # 1. Download the data
    download_data("https://pl-flash-data.s3.amazonaws.com/imdb.zip", 'data/')

    # 2. Load the data
    datamodule = TextClassificationData.from_files(
        train_file="data/imdb/train.csv",
        valid_file="data/imdb/valid.csv",
        test_file="data/imdb/test.csv",
        input="review",
        target="sentiment",
        batch_size=512
    )

    # 3. Build the model
    model = TextClassifier(num_classes=datamodule.num_classes)

    # 4. Create the trainer. Run once on data
    trainer = flash.Trainer(max_epochs=1)

    # 5. Fine-tune the model
    trainer.finetune(model, datamodule=datamodule, finetune_strategy='never_freeze')

    # 6. Test model
    trainer.test()

    # 7. Save it!
    trainer.save_checkpoint("text_classification_model.pt")
