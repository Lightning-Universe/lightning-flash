Quick Start
===========

Image Classification
--------------------

.. code-block:: python

    from pl_flash.vision import ImageClassifier, ImageClassificationData
    import pytorch_lightning as pl

    model = ImageClassifier(backbone="resnet18", num_classes=2)

    data = ImageClassificationData.from_folders(
        train_folder="train/",
        valid_folder="val/"
    )

    pl.Trainer().fit(model, data)


Text Classification
-------------------

.. code-block:: python

    from pl_flash.text import TextClassifier, TextClassificationData
    import pytorch_lightning as pl

    model = TextClassifier(backbone="bert-base-cased", num_classes=2)

    data = TextClassificationData.from_files(
        backbone="bert-base-cased",
        train_file="train.csv",
        valid_file="val.csv",
        text_field="sentence",
        label_field="label",
    )

    pl.Trainer().fit(model, data)


Tabular Classification
----------------------

.. code-block:: python

    from pl_flash.tabular import TabularClassifier, TabularData

    import pytorch_lightning as pl
    import pandas as pd

    data = TabularData.from_df(
        pd.read_csv("train.csv"),
        categorical_cols=["category"],
        numerical_cols=["scalar_b", "scalar_b"],
        target_col="target",
    )

    model = TabularClassifier(
        num_classes=2,
        num_columns=3,
        embedding_sizes=data.emb_sizes,
    )

    pl.Trainer().fit(model, data)
