.. _remote_data_loading:

*******************
Remote Data Loading
*******************

Where possible, all file loading in Flash uses the `fsspec library <https://github.com/fsspec/filesystem_spec>`_.
As a result, file references can use any of the protocols returned by ``fsspec.available_protocols()``.

For example, you can load :class:`~flash.tabular.classification.data.TabularClassificationData` from a URL to a CSV file:

.. testcode:: tabular

    from flash.tabular import TabularClassificationData

    datamodule = TabularClassificationData.from_csv(
        categorical_fields=["Sex", "Age", "SibSp", "Parch", "Ticket", "Cabin", "Embarked"],
        numerical_fields="Fare",
        target_fields="Survived",
        train_file="https://pl-flash-data.s3.amazonaws.com/titanic.csv",
        val_split=0.1,
        batch_size=8,
    )

Here's another example, showing how you can load :class:`~flash.image.classification.data.ImageClassificationData` for prediction using images found on the web:

.. testcode:: image

    from flash.image import ImageClassificationData

    datamodule = ImageClassificationData.from_files(
        predict_files=[
            "https://pl-flash-data.s3.amazonaws.com/images/ant_1.jpg",
            "https://pl-flash-data.s3.amazonaws.com/images/ant_2.jpg",
            "https://pl-flash-data.s3.amazonaws.com/images/bee_1.jpg",
            "https://pl-flash-data.s3.amazonaws.com/images/bee_2.jpg",
        ],
        batch_size=4,
    )
