Quick Start
===========
Flash is designed for rapid applied deep learning such as.

- finetuning
- solving common problems
- generating embeddings
- ...

In just a few short lines you can leverage the latest state of the art models for any
particular task you want to solve.

Flash is built on top of PyTorch Lightning so you can train across GPUs, TPUs etc without doing
any code changes. Tasks are just custom LightningModules, so when you need more flexibility you can
simply use lightning directly or modify and existing task in just a few lines.

For most applied deep learning
------------------------------

Flash is designed for quickly finetuning and doing applied deep learning. The FlashTask is
very general and can handle the majority of these problems.

.. code-block:: python

    from pl_flash.model import LightningTask
    from torch import nn, optim
    from torch.utils.data import DataLoader, random_split
    from torchvision import transforms, datasets
    import pytorch_lightning as pl

    # model
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    # data
    dataset = datasets.MNIST('./data_folder', download=True, transform=transforms.ToTensor())
    train, val = random_split(dataset, [55000, 5000])

    # task
    classifier = LightningTask(model, loss_fn=nn.functional.cross_entropy, optimizer=optim.Adam)

    # train
    pl.Trainer().fit(classifier, DataLoader(train), DataLoader(val))

----

Image Classification
--------------------

Lets say you wanted to develope a model that could classify between **ants** and **bees**. 
We only need a ``train`` and ``validation`` folder, each with examples of images of **ants** and **bees** like so: 

.. code-block::

    hymenoptera_data
    ├── train
    │   ├── ants
    │   │   ├── 0013035.jpg
    │   │   ├── 1030023514_aad5c608f9.jpg
    │   │   ...
    │   └── bees
    │       ├── 1092977343_cb42b38d62.jpg
    │       ├── 1093831624_fb5fbe2308.jpg
    │       ...
    └── val
        ├── ants
        │   ├── 10308379_1b6c72e180.jpg
        │   ├── 1053149811_f62a3410d3.jpg
        │   ...
        └── bees
            ├── 1032546534_06907fe3b3.jpg
            ├── 10870992_eebeeb3a12.jpg
            ...

Now all we need is three lines of code to build and train our model!

.. code-block:: python

    from pl_flash.vision import ImageClassifier, ImageClassificationData
    import pytorch_lightning as pl
    from pl_flash.data import download_data

    # download data
    download_data("https://download.pytorch.org/tutorial/hymenoptera_data.zip", 'data/')

    # 1. organize our data
    data = ImageClassificationData.from_folders(
        train_folder="data/hymenoptera_data/train/",
        valid_folder="data/hymenoptera_data/val/"
    )

    # 2. build our model
    model = ImageClassifier(backbone="resnet18", num_classes=2)

    # 3. train!
    pl.Trainer().fit(model, data)

------

To run the example:

.. code-block:: python

    python pl_flash_examples/torchvision_classifier.py


Text Classification
-------------------

Say you wanted to classify movie reviews as **positive** or **negative**. From a ``train.csv`` and ``valid.csv``, structured like so:

.. code-block::

    review,sentiment
    "Japanese indie film with humor ... ",positive
    "Isaac Florentine has made some ...",negative
    "After seeing the low-budget ...",negative
    "I've seen the original English version ...",positive
    "Hunters chase what they think is a man through ...",negative
    ...

Once again, all we need is three lines of code to train our model!

.. code-block:: python

    from pl_flash.text import TextClassifier, TextClassificationData
    import pytorch_lightning as pl
    from pl_flash.data import download_data

    # download data
    download_data("https://pl-flash-data.s3.amazonaws.com/imdb.zip", 'data/')

    # build our model
    model = TextClassifier(backbone="bert-base-cased", num_classes=2)

    # structure our data
    data = TextClassificationData.from_files(
        backbone="bert-base-cased",
        train_file="data/imdb/train.csv",
        valid_file="data/imdb/valid.csv",
        text_field="review",
        label_field="sentiment",
    )

    # train
    pl.Trainer().fit(model, data)

------

To run the example:

.. code-block:: python

    python pl_flash_examples/text_classification.py


Tabular Classification
----------------------

Lastly, say we want to build a model to predict if a passenger survived on the
Titanic. Once again we can organize our data in ``.csv`` files
(exportable from Excel):


.. code-block::

    PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
    1,0,3,"Braund, Mr. Owen Harris",male,22,1,0,A/5 21171,7.25,,S
    3,1,3,"Heikkinen, Miss. Laina",female,26,0,0,STON/O2. 3101282,7.925,,S
    5,0,3,"Allen, Mr. William Henry",male,35,0,0,373450,8.05,,S
    6,0,3,"Moran, Mr. James",male,,0,0,330877,8.4583,,Q
    ...

And now we train:

.. code-block:: python

    from pl_flash.tabular import TabularClassifier, TabularData
    import pytorch_lightning as pl
    import pandas as pd
    from pl_flash.data import download_data

    # download data
    download_data("https://pl-flash-data.s3.amazonaws.com/titanic.csv", "titanic.csv")

    # structure data
    data = TabularData.from_df(
        pd.read_csv("titanic.csv"),
        categorical_cols=["Sex", "Age", "SibSp", "Parch", "Ticket", "Cabin", "Embarked"],
        numerical_cols=["Fare"],
        target_col="Survived",
        num_workers=0,
        batch_size=8
    )

    # build model
    model = TabularClassifier(
        num_classes=2,
        num_columns=8,
        embedding_sizes=data.emb_sizes,
    )

    pl.Trainer().fit(model, data)


To run the example:

.. code-block:: python

    python pl_flash_examples/tabular_data.py
