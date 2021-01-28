from pytorch_lightning.metrics.classification import Accuracy, Precision, Recall

import flash
from flash.core.data import download_data
from flash.tabular import TabularClassifier, TabularData

if __name__ == "__main__":

    # 1. Download the data
    download_data("https://pl-flash-data.s3.amazonaws.com/titanic.zip", 'data/')

    # 2. Load the data
    datamodule = TabularData.from_csv(
        "./data/titanic/titanic.csv",
        categorical_input=["Sex", "Age", "SibSp", "Parch", "Ticket", "Cabin", "Embarked"],
        numerical_input=["Fare"],
        target="Survived",
        val_size=0.25,
    )

    # 3. Build the model
    model = TabularClassifier.from_data(datamodule, metrics=[Accuracy(), Precision(), Recall()])

    # 4. Create the trainer. Run 10 times on data
    trainer = flash.Trainer(max_epochs=10)

    # 5. Train the model
    trainer.fit(model, datamodule=datamodule)

    # 6. Save it!
    trainer.save_checkpoint("tabular_classification_model.pt")
