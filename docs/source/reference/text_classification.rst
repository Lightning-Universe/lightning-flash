.. _text_classification:

Text Classification
===================

The task
--------

Text classification is the task of assigning a piece of text (word, sentence or document) an appropriate class, or category. The categories depend on the chosen dataset and can range from topics. For example, we can use text classification to understand the sentiment of a given sentance- if it is positive or negative. 


Inference
---------

The :class:`~flash.text.TextClassificatier` is already pre-trained on [IMDB](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews), a dataset of highly polarized movie reviews, trained for binary classification- to predict if a given review has a positive or negative sentiment.

Use the :class:`~flash.text.TextClassificatier` pretrained model for inference on any string sequence using :func:`~flash.text.TextClassifier.predict`:

.. code-block:: python

	# import our libraries
	from flash.text import TextClassifier


	# Load finetuned task from URL
    model = TextClassifier.load_from_checkpoint("https://flash-weights.s3.amazonaws.com/text_classification_model.pt")

	# 2. Perform inference from list of sequences
	predictions = model.predict([
	    "Turgid dialogue, feeble characterization - Harvey Keitel a judge?.",
	    "The worst movie in the history of cinema.",
	    "I come from Bulgaria where it 's almost impossible to have a tornado."
	    "Very, very afraid"
	    "This guy has done a great job with this movie!",
	])
	print(predictions)

Or on a given dataset:

.. code-block:: python

	# import our libraries

	from flash.core.data import download_data
	from flash.text import TextClassifier


	# 1. Download dataset, save it under 'data' dir
	download_data("https://pl-flash-data.s3.amazonaws.com/imdb.zip", 'data/')

	# 2. Load finetuned task
    model = TextClassifier.load_from_checkpoint("https://flash-weights.s3.amazonaws.com/text_classification_model.pt")

	# 3. Perform inference from a csv file
	predictions = model.predict("data/imdb/test.csv")
	print(predictions)

For more advanced inference options, see :ref:`predictions`.

Finetuning
----------

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

	# import our libraries
	import flash
	from flash.core.data import download_data
	from flash.text import TextClassificationData, TextClassifier

    # 1. Download data
    download_data("https://pl-flash-data.s3.amazonaws.com/imdb.zip", 'data/')

    # Organize the data
    datamodule = TextClassificationData.from_files(
        train_file="data/imdb/train.csv",
        valid_file="data/imdb/valid.csv",
        input="review",
        target="sentiment",
        batch_size=512
    )

    # 2. Build the task (using the defult backbone="bert-base-cased")
    model = TextClassifier(num_classes=datamodule.num_classes)

    # 4. Create trainer
    trainer = flash.Trainer()

    # 5. Finetune the task
    trainer.finetune(model, datamodule=datamodule, unfreeze_milestones=(0, 1))

    # 6. Save trainer task
    trainer.save_checkpoint("text_classification_model.pt")

----

To run the example:

.. code-block:: python

    python flash_examples/finetuning/text_classification.py


------

Changing the backbone
---------------------
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


API reference
-------------

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



