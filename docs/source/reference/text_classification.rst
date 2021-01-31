.. _text_classification:

###################
Text Classification
###################

********
The task
********

Text classification is the task of assigning a piece of text (word, sentence or document) an appropriate class, or category. The categories depend on the chosen dataset and can range from topics. For example, we can use text classification to understand the sentiment of a given sentence- if it is positive or negative.

-----

*********
Inference
*********

The :class:`~flash.core.classification.TextClassificatier` is already pre-trained on `IMDB <https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews>`_, a dataset of highly polarized movie reviews, trained for binary classification- to predict if a given review has a positive or negative sentiment.

Use the :class:`~flash.core.classification.TextClassificatier` pretrained model for inference on any string sequence using :func:`~flash.core.classification.TextClassifier.predict`:

.. code-block:: python

    from pytorch_lightning import Trainer

    from flash.core.data import download_data
    from flash.text import TextClassificationData, TextClassifier


    # 1. Download the data
    download_data("https://pl-flash-data.s3.amazonaws.com/imdb.zip", 'data/')

    # 2. Load the model from a checkpoint
    model = TextClassifier.load_from_checkpoint("https://flash-weights.s3.amazonaws.com/text_classification_model.pt")

    # 2a. Classify a few sentences! How was the movie?
    predictions = model.predict([
        "Turgid dialogue, feeble characterization - Harvey Keitel a judge?.",
        "The worst movie in the history of cinema.",
        "I come from Bulgaria where it 's almost impossible to have a tornado."
        "Very, very afraid"
        "This guy has done a great job with this movie!",
    ])
    print(predictions)

    # 2b. Or generate predictions from a sheet file!
    datamodule = TextClassificationData.from_file(
        predict_file="data/imdb/predict.csv",
        input="review",
    )
    predictions = Trainer().predict(model, datamodule=datamodule)
    print(predictions)

For more advanced inference options, see :ref:`predictions`.

-----

**********
Finetuning
**********

Say you wanted to create a model that can predict whether a movie review is **positive** or **negative**. We will be using the IMDB dataset, that contains a ``train.csv`` and ``valid.csv``, structured like so:

.. code-block::

    review,sentiment
    "Japanese indie film with humor ... ",positive
    "Isaac Florentine has made some ...",negative
    "After seeing the low-budget ...",negative
    "I've seen the original English version ...",positive
    "Hunters chase what they think is a man through ...",negative
    ...

All we need is three lines of code to train our model!

.. code-block:: python

    import flash
    from flash.core.data import download_data
    from flash.text import TextClassificationData, TextClassifier

    # 1. Download the data
    download_data("https://pl-flash-data.s3.amazonaws.com/imdb.zip", 'data/')

    # 2. Load the data
    datamodule = TextClassificationData.from_files(
        train_file="data/imdb/train.csv",
        valid_file="data/imdb/valid.csv",
        test_file="data/imdb/test.csv",
        input="review",
        target="sentiment",
        batch_size=512
    )

    # 3. Build the task (using the default backbone="bert-base-cased")
    model = TextClassifier(num_classes=datamodule.num_classes)

    # 4. Create the trainer. Run once on data
    trainer = flash.Trainer(max_epochs=1)

    # 5. Fine-tune the model
    trainer.finetune(model, datamodule=datamodule, unfreeze_milestones=(0, 1))

    # 6. Test model
    trainer.test()

    # 7. Save it!
    trainer.save_checkpoint("text_classification_model.pt")

----

To run the example:

.. code-block:: bash

    python flash_examples/finetuning/text_classification.py


------

*********************
Changing the backbone
*********************
By default, we use the `bert-base-uncased <https://arxiv.org/abs/1810.04805>`_ model for text classification. You can change the model run by the task to any BERT model from `HuggingFace/transformers <https://huggingface.co/models>`_ by passing in a different backbone.

.. note:: When changing the backbone, make sure you pass in the same backbone to the Task and the Data object!

.. code-block:: python

    datamodule = TextClassificationData.from_files(
    	backbone="bert-base-chinese",
        train_file="data/imdb/train.csv",
        valid_file="data/imdb/valid.csv",
        input="review",
        target="sentiment",
        batch_size=512
    )

    task = TextClassifier(backbone="bert-base-chinese", num_classes=datamodule.num_classes)

------

*************
API reference
*************

.. _text_classifier:

TextClassifier
--------------

.. autoclass:: flash.text.classification.model.TextClassifier
    :members:
    :exclude-members: forward

.. _text_classification_data:

TextClassificationData
----------------------

.. autoclass:: flash.text.classification.data.TextClassificationData

.. automethod:: flash.text.classification.data.TextClassificationData.from_files



