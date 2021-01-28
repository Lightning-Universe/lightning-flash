# import our libraries
import torch
from flash.core.model import download_model


if __name__ == "__main__":

    # 1. Download and load model from checkpoint
    download_model("tabular_classification_model.pt")
    model = torch.load("tabular_classification_model.pt", map_location=torch.device('cpu'))

    # 2. Predict over a path to a `.csv` file
    predictions = model.predict("data/titanic/titanic.csv")
    print(predictions)
