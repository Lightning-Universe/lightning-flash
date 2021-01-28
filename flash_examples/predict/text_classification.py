# import our libraries
import torch
from pytorch_lightning import Trainer

from flash.core.data import download_data
from flash.core.model import download_model
from flash.text import TextClassificationData

if __name__ == "__main__":

    # 1. Load finetuned model - Run `python flash_examples/finetuning/text_classification.py` first.
    model = torch.load("text_classification_model.pt", map_location=torch.device('cpu'))

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
        predict_file="data/imdb/test.csv",
        input="review",
    )
    predictions = Trainer().predict(model, datamodule=datamodule)
    print(predictions)
