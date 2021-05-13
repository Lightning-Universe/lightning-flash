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

* **cat_cols**- a list of the names of columns that contain categorical data (strings or integers)
* **num_cols**- a list of the names of columns that contain numerical continuous data (floats)
* **target**- the name of the column we want to predict
* **train_csv**- csv file containing the training data converted to a Pandas DataFrame

Next, we create the :class:`~flash.tabular.TabularClassifier` task, using the Data module we created.

.. literalinclude:: ../../../flash_examples/finetuning/tabular_classification.py
    :language: python
    :lines: 14-

-----

*********
Inference
*********

You can make predictions on a pretrained model, that has already been trained for the titanic task:

.. literalinclude:: ../../../flash_examples/predict/tabular_classification.py
    :language: python
    :lines: 14-

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

.. automethod:: flash.tabular.TabularData.from_data_frame
