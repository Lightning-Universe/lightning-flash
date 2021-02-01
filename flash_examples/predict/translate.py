from pytorch_lightning import Trainer

from flash.core.data import download_data
from flash.text import TranslationData, TranslationTask

if __name__ == "__main__":

    # 1. Download the data
    download_data("https://pl-flash-data.s3.amazonaws.com/wmt_en_ro.zip", 'data/')

    # 2. Load the model from a checkpoint
    model = TranslationTask.load_from_checkpoint("https://flash-weights.s3.amazonaws.com/translation_model_en_ro.pt")

    # 2a. Translate a few sentences!
    predictions = model.predict([
        "BBC News went to meet one of the project's first graduates.",
        "A recession has come as quickly as 11 months after the first rate hike and as long as 86 months.",
    ])
    print(predictions)

    # 2b. Or generate translations from a sheet file!
    datamodule = TranslationData.from_file(
        predict_file="data/wmt_en_ro/predict.csv",
        input="input",
    )
    predictions = Trainer().predict(model, datamodule=datamodule)
    print(predictions)
