# import our libraries
from flash.core.data import download_data
from flash.tabular import TabularClassifier

if __name__ == "__main__":

    # 1. Download data
    download_data("https://pl-flash-data.s3.amazonaws.com/titanic.zip", 'data/')

    # 2. Download and load model from checkpoint
    model = TabularClassifier.load_from_checkpoint(
        "https://flash-weights.s3.amazonaws.com/tabular_classification_model.pt"
    )

    # 3. Predict over a path to a `.csv` file
    predictions = model.predict("data/titanic/titanic.csv")
    print(predictions)
