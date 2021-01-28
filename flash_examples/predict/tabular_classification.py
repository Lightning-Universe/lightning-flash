from flash.core.data import download_data
from flash.tabular import TabularClassifier

if __name__ == "__main__":

    # 1. Download the data
    download_data("https://pl-flash-data.s3.amazonaws.com/titanic.zip", 'data/')

    # 2. Load the model from a checkpoint
    model = TabularClassifier.load_from_checkpoint(
        "https://flash-weights.s3.amazonaws.com/tabular_classification_model.pt"
    )

    # 3. Generate predictions from a sheet file! Who would survive?
    predictions = model.predict("data/titanic/titanic.csv")
    print(predictions)
