Quick Start
===========

Image Classification
--------------------

Lets say you wanted to develope a model that could classify between **dogs** and **cats**. 
We only need a ``train`` and ``validation`` folder, each with examples of images of dogs and cats like so: 

.. code-block::

    train
    ├── dogs 
    │   ├── 0013035.jpg
    │   ├── 1030023514_aad5c608f9.jpg
    │   ├── 1095476100_3906d8afde.jpg
    |   ...
    └── cats
        ├── 1092977343_cb42b38d62.jpg
        ├── 1093831624_fb5fbe2308.jpg
        ├── 1097045929_1753d1c765.jpg
        ...

Now all we need is three lines of code to build and train our model!

.. code-block:: python

    # import our libraries
    from pl_flash.vision import ImageClassifier, ImageClassificationData
    import pytorch_lightning as pl

    # 1. build our model
    model = ImageClassifier(backbone="resnet18", num_classes=2)

    # 2. organize our data
    data = ImageClassificationData.from_folders(
        train_folder="train/",
        valid_folder="validation/"
    )

    # 3. train!
    pl.Trainer().fit(model, data)


Text Classification
-------------------

Say you wanted to classify a sentence as "happy" or "angry". Simply collect a
``train.csv`` and ``val.csv``, structured like so:

.. code-block::

    sentence,label
    I love puppies!,1
    I am very angry,0
    This sandwich was delicious,1
    "I want a refund on my sandwich, it was awful!",0
    ...

Once again, all we need is three lines of code to train our model!

.. code-block:: python

    from pl_flash.text import TextClassifier, TextClassificationData
    import pytorch_lightning as pl

    # build our model
    model = TextClassifier(backbone="bert-base-cased", num_classes=2)

    # structure our data
    data = TextClassificationData.from_files(
        backbone="bert-base-cased",
        train_file="train.csv",
        valid_file="val.csv",
        text_field="sentence",
        label_field="label",
    )

    # train
    pl.Trainer().fit(model, data)


Tabular Classification
----------------------

Lastly, say we want to build a model to predict if a customer will cancel their
service. Once again we can orgainize our data in ``.csv`` files (exportable
from Excel):


.. code-block::

    years_as_customer,purchases,money_spent,country,cancelled
    5,20,123.45,USA,0
    1,1,55.55,Canada,1
    10,2,23.45,Mexico,0
    ...

And now we train:

.. code-block:: python

    from pl_flash.tabular import TabularClassifier, TabularData

    import pytorch_lightning as pl
    import pandas as pd

    # stucture data
    data = TabularData.from_df(
        pd.read_csv("train.csv"),
        categorical_cols=["years_as_customer", "country"],
        numerical_cols=["money_spent", "purchases"],
        target_col="cancelled",
    )

    # build model
    model = TabularClassifier(
        num_classes=2,
        num_columns=3,
        embedding_sizes=data.emb_sizes,
    )

    # train
    pl.Trainer().fit(model, data)
