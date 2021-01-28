Quick Start
===========
Flash is a toolkit to solve deep learning challanges using the latest state-of-the-art models in just a few lines of code. You can use Flash to finetune on you own data, generate embeddings, etc.

Flash is built on top of `Pytorch Lightning
<https://github.com/PyTorchLightning/pytorch-lightning>`_, a powerful deep learning research framework for training models at scale. With the power of Lightning, you can train your flash tasks on any hardware: CPUs, GPUs or TPUs without any code changes. 

Continue reading to train your deep learning models in a flash!


Install
-------

You can install flash using pip:

.. code-block:: bash

    pip install pytorch-lightning-flash


Quick Inference
---------------
.. code-block:: python

    # import our libraries
    from flash.text import TextClassifier


    # Load finetuned task
    model = ImageClassifier.load_from_checkpoint("https://flash-weights.s3.amazonaws.com/image_classification_model.pt")

    # 2. Perform inference from list of sequences
    predictions = model.predict([
        "Turgid dialogue, feeble characterization - Harvey Keitel a judge?.",
        "The worst movie in the history of cinema.",
        "I come from Bulgaria where it 's almost impossible to have a tornado."
        "Very, very afraid"
        "This guy has done a great job with this movie!",
    ])
    print(predictions)

Finetune
--------

.. code-block:: python

    import flash
    from flash.core.data import download_data
    from flash.vision import ImageClassificationData, ImageClassifier


    # 1. Download the data
    download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", 'data/')

    # 2. Load the data
    datamodule = ImageClassificationData.from_folders(
        backbone="resnet18",
        train_folder="data/hymenoptera_data/train/",
        valid_folder="data/hymenoptera_data/val/",
        test_folder="data/hymenoptera_data/test/",
    )

    # 3. Build the model
    model = ImageClassifier(num_classes=datamodule.num_classes)

    # 4. Create the trainer. Run once on data
    trainer = flash.Trainer(max_epochs=1)

    # 5. Train the model
    trainer.finetune(model, datamodule=datamodule, unfreeze_milestones=(0, 1))

    # 6. Test the model
    trainer.test()

    # 7. Save it!
    trainer.save_checkpoint("image_classification_model.pt")

Train
-----

Tasks let you focus on solving applied problems without any of the boilerplate. Here's a built-in
task that works for 99% of machine learning problems that data scientists, kagglers and practicioners
encounter.

.. code-block:: python

    import flash
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
    classifier = flash.Task(model, loss_fn=nn.functional.cross_entropy, optimizer=optim.Adam)

    # train
    flash.Trainer().fit(classifier, DataLoader(train), DataLoader(val))


Customize
---------

Tasks can be built in just a few minutes because Flash is built on top of PyTorch Lightning LightningModules, which
are infinitely extensible and let you train across GPUs, TPUs etc without doing any code changes.

---------

Image Classification
--------------------

Flash has an ImageClassification task to tackle any image classification problem.
To illustrate, Let's say we wanted to develop a model that could classify between **ants** and **bees**.
We only need a ``train`` and ``validation`` folder, each with examples of images of ants and bees like so:

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

    from flash.vision import ImageClassifier, ImageClassificationData
    import pytorch_lightning as pl
    from flash.core.data import download_data

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

To run the example:

.. code-block:: python

    python flash_examples/torchvision_classifier.py

---------

Text Classification
-------------------

Flash has a TextClassification task to tackle any text classification problem.
To illustrate, say you wanted to classify movie reviews as **positive** or **negative**. From a ``train.csv`` and ``valid.csv``, structured like so:

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

    from flash.text import TextClassifier, TextClassificationData
    import pytorch_lightning as pl
    from flash.core.data import download_data

    # download data
    download_data("https://pl-flash-data.s3.amazonaws.com/imdb.zip", 'data/')

    # build our model
    model = TextClassifier(backbone="bert-base-cased", num_classes=2)

    # structure our data
    data = TextClassificationData.from_files(
        backbone="bert-base-cased",
        train_file="data/imdb/train.csv",
        valid_file="data/imdb/valid.csv",
        input="review",
        target="sentiment",
    )

    # train
    pl.Trainer().fit(model, data)

To run the example:

.. code-block:: python

    python flash_examples/text_classification.py

-----

Tabular Classification
----------------------

Flash has a TabularClassification task to tackle any tabular classification problem.
To illustrate, say we want to build a model to predict if a passenger survived on the
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

    from flash.tabular import TabularClassifier, TabularData
    import pytorch_lightning as pl
    import pandas as pd
    from flash.core.data import download_data

    # download data
    download_data("https://pl-flash-data.s3.amazonaws.com/titanic.csv", "titanic.csv")

    # structure data
    data = TabularData.from_df(
        pd.read_csv("titanic.csv"),
        categorical_input=["Sex", "Age", "SibSp", "Parch", "Ticket", "Cabin", "Embarked"],
        numerical_input=["Fare"],
        target="Survived",
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

    python flash_examples/tabular_data.py

---------

Task for 99% of ML problems
---------------------------

The Task can be used to tackle the majority of machine learning problems data scientists encounter.

.. code-block:: python

    from flash.model import Task
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
    classifier = Task(model, loss_fn=nn.functional.cross_entropy, optimizer=optim.Adam)

    # train
    pl.Trainer().fit(classifier, DataLoader(train), DataLoader(val))

Other tasks
-----------
The lightning + Flash team is hard at work building more tasks for common deep-learning use cases.
But we're looking for incredible contributors like you to submit new tasks!

Join our `Slack <https://join.slack.com/t/pytorch-lightning/shared_invite/zt-f6bl2l0l-JYMK3tbAgAmGRrlNr00f1A>`_ to get help becoming a contributor!
