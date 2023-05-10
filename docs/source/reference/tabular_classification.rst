.. customcarditem::
   :header: Tabular Classification
   :card_description: Learn to classify tabular records with Flash and build an example model to predict survival rates on the Titanic.
   :image: https://pl-flash-data.s3.amazonaws.com/assets/thumbnails/tabular_classification.svg
   :tags: Tabular,Classification

.. _tabular_classification:

######################
Tabular Classification
######################

********
The Task
********

Tabular classification is the task of assigning a class to samples of structured or relational data.
The :class:`~flash.tabular.classification.model.TabularClassifier` task can be used for classification of samples in more than two classes (multi-class classification).

------

*******
Example
*******

Let's look at training a model to predict if passenger survival on the Titanic using `the classic Kaggle data set <https://www.kaggle.com/c/titanic-dataset/data>`_.
The data is provided in CSV files that look like this:

.. code-block::

    PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
    1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
    3,1,3,"Heikkinen, Miss. Laina",female,26,0,0,STON/O2. 3101282,7.925,,S
    5,0,3,"Allen, Mr. William Henry",male,35,0,0,373450,8.05,,S
    6,0,3,"Moran, Mr. James",male,,0,0,330877,8.4583,,Q
    ...

We can create the :class:`~flash.tabular.classification.data.TabularData` from our CSV files using the :func:`~flash.tabular.classification.data.TabularData.from_csv` method.
From :meth:`the API reference <flash.tabular.classification.data.TabularData.from_csv>`, we need to provide:

* **cat_cols**- A list of the names of columns that contain categorical data (strings or integers).
* **num_cols**- A list of the names of columns that contain numerical continuous data (floats).
* **target**- The name of the column we want to predict.
* **train_csv**- A CSV file containing the training data converted to a Pandas DataFrame

Next, we create the :class:`~flash.tabular.classification.model.TabularClassifier` and finetune on the Titanic data.
We then use the trained :class:`~flash.tabular.classification.model.TabularClassifier` for inference.
Finally, we save the model.
Here's the full example:

.. literalinclude:: ../../../examples/tabular/tabular_classification.py
    :language: python
    :lines: 14-

To learn how to view the available backbones / heads for this task, see :ref:`backbones_heads`.

------

**********
Flash Zero
**********

The tabular classifier can be used directly from the command line with zero code using :ref:`flash_zero`.
You can run the above example with:

.. code-block:: bash

    flash tabular_classifier

To view configuration options and options for running the tabular classifier with your own data, use:

.. code-block:: bash

    flash tabular_classifier --help

------

*******
Serving
*******

The :class:`~flash.tabular.classification.model.TabularClassifier` is servable.
This means you can call ``.serve`` to serve your :class:`~flash.core.model.Task`.
Here's an example:

.. literalinclude:: ../../../examples/serve/tabular_classification/inference_server.py
    :language: python
    :lines: 14-

You can now perform inference from your client like this:

.. literalinclude:: ../../../examples/serve/tabular_classification/client.py
    :language: python
    :lines: 14-
