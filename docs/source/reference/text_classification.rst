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

The :class:`~flash.text.classification.model.TextClassifier` is already pre-trained on `IMDB <https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews>`_, a dataset of highly polarized movie reviews, trained for binary classification- to predict if a given review has a positive or negative sentiment.

Use the :class:`~flash.text.classification.model.TextClassifier` pretrained model for inference on any string sequence using :func:`~flash.text.classification.model.TextClassifier.predict`:

.. literalinclude:: ../../../flash_examples/predict/text_classification.py
    :language: python
    :lines: 14-

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

All we need is to train our model!

.. literalinclude:: ../../../flash_examples/finetuning/text_classification.py
    :language: python
    :lines: 14-

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

.. testsetup::

    from flash.data.utils import download_data
    from flash.text import TextClassificationData, TextClassifier

    download_data("https://pl-flash-data.s3.amazonaws.com/imdb.zip", "data/")

.. testcode::

    datamodule = TextClassificationData.from_csv(
        "review",
        "sentiment",
    	backbone="bert-base-chinese",
        train_file="data/imdb/train.csv",
        val_file="data/imdb/valid.csv",
        batch_size=512
    )

    task = TextClassifier(backbone="bert-base-chinese", num_classes=datamodule.num_classes)

.. testoutput::
    :hide:

    ...

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
