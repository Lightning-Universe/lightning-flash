# import our libraries
from pytorch_lightning import Trainer

from flash.core.data import download_data
from flash.core.model import download_model
from flash.text import TextClassificationData, TextClassifier

if __name__ == "__main__":

    # 1. Download data
    download_data("https://pl-flash-data.s3.amazonaws.com/imdb.zip", 'data/')

    # 2 Load finetuned model
    download_model("text_classification_model.pt")
    model = TextClassifier.load_from_checkpoint("text_classification_model.pt")

    # 2.1 Perform inference from list of sequences
    predictions = model.predict([
        "Turgid dialogue, feeble characterization - Harvey Keitel a judge?.",
        "The worst movie in the history of cinema.",
        "I come from Bulgaria where it 's almost impossible to have a tornado."
        "Very, very afraid"
        "This guy has done a great job with this movie!",
    ])
    print(predictions)

    # 2.2 Or perform inference from `.csv` file
    datamodule = TextClassificationData.from_file(
        predict_file="data/imdb/predict.csv",
        input="review",
    )
    predictions = Trainer().predict(model, datamodule=datamodule)
    print(predictions)
