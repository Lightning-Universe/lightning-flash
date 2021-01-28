# import our libraries
from flash import Trainer
from flash.core.data import download_data
from flash.core.model import download_model
from flash.vision import ImageClassificationData, ImageClassifier

if __name__ == "__main__":

    # 1. Download data
    download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", 'data/')

    # 2. Download and load model from checkpoint
    download_model("image_classification_model.pt")
    model = ImageClassifier.load_from_checkpoint("image_classification_model.pt")

    # 3.1 Make predictions on list of image paths
    predictions = model.predict([
        "data/hymenoptera_data/val/bees/65038344_52a45d090d.jpg",
        "data/hymenoptera_data/val/bees/590318879_68cf112861.jpg",
        "data/hymenoptera_data/val/ants/540543309_ddbb193ee5.jpg",
    ])
    print(predictions)

    # 3.2. Make predictions on folder of images
    datamodule = ImageClassificationData.from_folder(folder="data/hymenoptera_data/predict/")
    predictions = Trainer().predict(model, datamodule=datamodule)
    print(predictions)
