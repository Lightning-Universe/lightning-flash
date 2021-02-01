import flash
from flash.core.data import download_data
from flash.text import TranslationData, TranslationTask

if __name__ == "__main__":
    # 1. Download the data
    download_data("https://pl-flash-data.s3.amazonaws.com/wmt_en_ro.zip", 'data/')

    # 2. Load the data
    datamodule = TranslationData.from_files(
        train_file="data/wmt_en_ro/train.csv",
        valid_file="data/wmt_en_ro/valid.csv",
        test_file="data/wmt_en_ro/test.csv",
        input="input",
        target="target",
    )

    # 3. Build the model
    model = TranslationTask()

    # 4. Create the trainer. Run once on data
    trainer = flash.Trainer(max_epochs=5, gpus=1, precision=16)

    # 5. Fine-tune the model
    trainer.finetune(model, datamodule=datamodule)

    # 6. Test model
    trainer.test()

    # 7. Save it!
    trainer.save_checkpoint("translation_model_en_ro.pt")
