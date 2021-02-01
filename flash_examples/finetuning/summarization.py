import flash
from flash import download_data
from flash.text import SummarizationData, SummarizationTask

if __name__ == "__main__":
    # 1. Download the data
    download_data("https://pl-flash-data.s3.amazonaws.com/xsum.zip", 'data/')

    # 2. Load the data
    datamodule = SummarizationData.from_files(
        train_file="data/xsum/train.csv",
        valid_file="data/xsum/valid.csv",
        test_file="data/xsum/test.csv",
        input="input",
        target="target"
    )

    # 3. Build the model
    model = SummarizationTask()

    # 4. Create the trainer. Run once on data
    trainer = flash.Trainer(max_epochs=1, gpus=1)

    # 5. Fine-tune the model
    trainer.finetune(model, datamodule=datamodule)

    # 6. Test model
    trainer.test()

    # 7. Save it!
    trainer.save_checkpoint("summarization_model_xsum.pt")
