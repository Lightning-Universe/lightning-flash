.. _tabular_classification:

######################
Tabular Classification
######################

********
The task
********

Tabular classification is the task of assigning a class to samples of structured or relational data. The Flash Tabular Classification task can be used for multi-class classification, or classification of samples in more than two classes. In the following example, the Tabular data is structured into rows and columns, where columns represent properties or features. The task will learn to predict a single target column.

-----

**********
Finetuning
**********

Say we want to build a model to predict if a passenger survived on the
Titanic. We can organize our data in ``.csv`` files
(exportable from Excel, but you can find the kaggle dataset `here <https://www.kaggle.com/c/titanic-dataset/data>`_):


.. code-block::

    PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
    1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
    3,1,3,"Heikkinen, Miss. Laina",female,26,0,0,STON/O2. 3101282,7.925,,S
    5,0,3,"Allen, Mr. William Henry",male,35,0,0,373450,8.05,,S
    6,0,3,"Moran, Mr. James",male,,0,0,330877,8.4583,,Q
    ...

We can use the Flash Tabular classification task to predict the probability a passenger survived (1 means survived, 0 otherwise), using the feature columns.

We can create :class:`~flash.tabular.TabularData` from csv files using the :func:`~flash.tabular.TabularData.from_csv` method. We will pass in:

* **train_csv**- csv file containing the training data converted to a Pandas DataFrame
* **categorical_input**- a list of the names of columns that contain categorical data (strings or integers)
* **numerical_input**- a list of the names of columns that contain numerical continuous data (floats)
* **target**- the name of the column we want to predict


Next, we create the :class:`~flash.tabular.TabularClassifier` task, using the Data module we created.

.. code-block:: python

    import flash
    from flash import download_data
    from flash.tabular import TabularClassifier, TabularData
    from pytorch_lightning.metrics.classification import Accuracy, Precision, Recall

    # 1. Download the data
    download_data("https://pl-flash-data.s3.amazonaws.com/titanic.zip", 'data/')

    # 2. Load the data
    datamodule = TabularData.from_csv(
        "./data/titanic/titanic.csv",
        test_csv="./data/titanic/test.csv",
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

    # 6. Test model
    trainer.test()

    # 7. Save it!
    trainer.save_checkpoint("tabular_classification_model.pt")

    # 8. Predict!
    predictions = model.predict("data/titanic/titanic.csv")
    print(predictions)

-----

*********
Inference
*********

You can make predcitions on a pretrained model, that has already been trained for the titanic task:

.. code-block:: python


    from flash.core.data import download_data
    from flash.tabular import TabularClassifier

    # 1. Download the data
    download_data("https://pl-flash-data.s3.amazonaws.com/titanic.zip", 'data/')

    # 2. Load the model from a checkpoint
    model = TabularClassifier.load_from_checkpoint(
        "https://flash-weights.s3.amazonaws.com/tabnet_classification_model.pt"
    )

    # 3. Generate predictions from a sheet file! Who would survive?
    predictions = model.predict("data/titanic/titanic.csv")
    print(predictions)


Or you can finetune your own model and use that for prediction:

.. code-block:: python

    import flash
    from flash import download_data
    from flash.tabular import TabularClassifier, TabularData

    # 1. Load the data
    datamodule = TabularData.from_csv(
        "my_data_file.csv",
        test_csv="./data/titanic/test.csv",
        categorical_input=["Sex", "Age", "SibSp", "Parch", "Ticket", "Cabin", "Embarked"],
        numerical_input=["Fare"],
        target="Survived",
        val_size=0.25,
    )

    # 3. Build the model
    model = TabularClassifier.from_data(datamodule, metrics=[Accuracy(), Precision(), Recall()])

    # 4. Create the trainer
    trainer = flash.Trainer()

    # 5. Train the model
    trainer.fit(model, datamodule=datamodule)

    # 6. Test model
    trainer.test()

    predictions = model.predict("data/titanic/titanic.csv")
    print(predictions)

------

*************
API reference
*************

.. _tabular_classifier:

TabularClassifier
-----------------

.. autoclass:: flash.tabular.TabularClassifier
    :members:
    :exclude-members: forward

.. _tabular_data:

TabularData
-----------

.. autoclass:: flash.tabular.TabularData

.. automethod:: flash.tabular.TabularData.from_csv

.. automethod:: flash.tabular.TabularData.from_df

