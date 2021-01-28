# import our libraries
from flash.core.data import download_data
from flash.core.model import download_model
from flash.tabular import TabularClassifier

if __name__ == "__main__":

    # 1. Download data
    download_data("https://pl-flash-data.s3.amazonaws.com/titanic.zip", 'data/')

    # 2. Download and load model from checkpoint
    download_model("tabular_classification_model.pt")
    model = TabularClassifier.load_from_checkpoint("tabular_classification_model.pt")

    # 3. Predict over a path to a `.csv` file
    predictions = model.predict("data/titanic/titanic.csv")
    print(predictions)
