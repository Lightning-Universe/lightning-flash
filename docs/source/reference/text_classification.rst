.. _text_classification:

###################
Text Classification
###################

********
The Task
********

Text classification is the task of assigning a piece of text (word, sentence or document) an appropriate class, or category.
The categories depend on the chosen data set and can range from topics.

-----

*******
Example
*******

Let's train a model to classify text as expressing either positive or negative sentiment.
We will be using the IMDB data set, that contains a ``train.csv`` and ``valid.csv``.
Here's the structure:

.. code-block::

    review,sentiment
    "Japanese indie film with humor ... ",positive
    "Isaac Florentine has made some ...",negative
    "After seeing the low-budget ...",negative
    "I've seen the original English version ...",positive
    "Hunters chase what they think is a man through ...",negative
    ...

Once we've downloaded the data using :func:`~flash.core.data.download_data`, we create the :class:`~flash.text.classification.data.TextClassificationData`.
We select a pre-trained backbone to use for our :class:`~flash.text.classification.model.TextClassifier` and finetune on the IMDB data.
The backbone can be any BERT classification model from `HuggingFace/transformers <https://huggingface.co/models?filter=pytorch&pipeline_tag=text-classification>`_.

.. note::

    When changing the backbone, make sure you pass in the same backbone to the :class:`~flash.text.classification.model.TextClassifier` and the :class:`~flash.text.classification.data.TextClassificationData`!

Next, we use the trained :class:`~flash.text.classification.model.TextClassifier` for inference.
Finally, we save the model.
Here's the full example:

.. literalinclude:: ../../../flash_examples/text_classification.py
    :language: python
    :lines: 14-

------

*******
Serving
*******

The :class:`~flash.text.classification.model.TextClassifier` is servable.
This means you can call ``.serve`` to serve your :class:`~flash.core.model.Task`.
Here's an example:

.. literalinclude:: ../../../flash_examples/serve/text_classification/inference_server.py
    :language: python
    :lines: 14-

You can now perform inference from your client like this:

.. literalinclude:: ../../../flash_examples/serve/text_classification/client.py
    :language: python
    :lines: 14-
